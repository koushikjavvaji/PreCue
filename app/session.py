"""The app's brain: keys, documents, and one meeting at a time, for the page in web/app.html.

  page --fetch /api/...--> PreCueApp.get() / post() --> settings.py, the meeting thread
  page <--/events-------- LiveScreen  (progress, transcript, alerts: as they happen)

A meeting goes through these states (each change is sent to the page as a "session" event):

  idle --Start--> preparing --(document ready, mic on)--> live --Stop--> idle
                      └──── (a problem) ────> error (the page shows the message)

The search model takes ~4 s to load, so it is loaded once, in the background, as soon as
the app opens. When you press Start, it is ready, and a document you used before is
ready in well under a second (it is cached).
"""

import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import sounddevice as sd

from app.documents import MAX_FILE_MB, SUPPORTED_FILES
from app.errors import PreCueError
from app.overlay import start_overlay, stop_overlay
from app.settings import check_key, key_status, save_keys

# The search model and LLM libraries (torch, LangChain) take ~4 s just to import.
# They are imported in the background threads below, so the page opens at once.

PROJECT = Path(__file__).parent.parent
UPLOADS = PROJECT / "data" / "uploads"  # data/ is never committed
SAMPLES = PROJECT / "samples"
SAMPLE_DOCUMENTS = ["sample_report.txt", "aurora_report.md"]
SAFE_NAME = re.compile(r"[^A-Za-z0-9._ ()-]")
MODEL_WAIT_SECONDS = 180  # the first start downloads the model (~600 MB); later it loads in ~2 s
KEY_CHECK_SECONDS = 60  # a key check this recent is answered from memory, at once
EARPHONES = re.compile(r"airpods|headphone|headset|earphone|earbud|buds|beats|bose|jabra|wh-|wf-", re.IGNORECASE)


class Cancelled(Exception):
    """Stop was pressed while the meeting was still being prepared."""


class PreCueApp:
    MAX_UPLOAD_BYTES = MAX_FILE_MB * 1024 * 1024

    def __init__(self, load_model=True):
        """load_model=False: for tests, which give the app a fake model instead."""
        self.screen = None  # set by scripts/app.py, after the server is made
        self.state = {"state": "idle", "text": "", "steps": [], "document": "", "warnings": []}
        self.lock = threading.Lock()  # one Start at a time
        self.meeting = None
        self.stt = None
        self.cancel = threading.Event()  # Stop pressed
        self._embedder = None
        self._model_ready = threading.Event()
        self._key_checks = None  # (time, result): the last answer from the providers
        if load_model:
            threading.Thread(target=self._load_model, daemon=True).start()
            threading.Thread(target=self._check_all, daemon=True).start()  # the first check takes ~4 s

    # ---------------- the page asks ----------------

    def get(self, path):
        if path == "/api/state":
            return 200, {
                "keys": key_status(),
                "session": self.state,
                "documents": documents(),
                "devices": output_devices(),
                "overlay_hidden_from_sharing": sys.platform in ("darwin", "win32"),
                "model_ready": self._model_ready.is_set(),
            }
        return 404, {"error": "Not found"}

    def post(self, path, body, headers):
        try:
            if path == "/api/keys":
                return self._save_keys(body.get("keys", {}))
            if path == "/api/keys/check":
                fresh = self._key_checks and time.time() - self._key_checks[0] < KEY_CHECK_SECONDS
                return 200, {"keys": self._key_checks[1] if fresh else self._check_all()}
            if path == "/api/upload":
                return self._upload(headers.get("X-File-Name", ""), body)
            if path == "/api/start":
                return self._start(body)
            if path == "/api/stop":
                self.stop()
                return 200, {"session": self.state}
        except (PreCueError, ValueError) as error:
            return 400, {"error": str(error)}
        return 404, {"error": "Not found"}

    # ---------------- keys and documents ----------------

    def _save_keys(self, keys):
        save_keys({name: value for name, value in keys.items() if str(value).strip()})
        return 200, {"keys": self._check_all()}

    def _check_all(self):
        """Ask all three providers at the same time: about 0.4 s instead of 1.2 s."""
        items = key_status()
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            answers = list(pool.map(lambda item: check_key(item["id"]), items))
        result = [{**item, "works": works, "message": message} for item, (works, message) in zip(items, answers)]
        self._key_checks = (time.time(), result)
        return result

    def _upload(self, raw_name, data):
        name = SAFE_NAME.sub("_", Path(unquote(raw_name)).name).strip(" .")
        if Path(name).suffix.lower() not in SUPPORTED_FILES:
            raise PreCueError("Use a .pdf, .docx, .pptx, .txt or .md file.")
        if not data:
            raise PreCueError("The file is empty.")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        os.chmod(UPLOADS, 0o700)
        path = UPLOADS / name
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(data)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        return 200, {"document": document_entry(path, "uploads"), "documents": documents()}

    # ---------------- the meeting ----------------

    def _start(self, options):
        document = find_document(options.get("document"))
        if document is None:
            raise PreCueError("Choose a document first.")
        missing = [item["name"] for item in key_status() if not item["set"]]
        if missing:
            raise PreCueError(f"Add your {' and '.join(missing)} key first.")
        if not self.lock.acquire(blocking=False):
            raise PreCueError("A meeting is already running.")
        self.screen.clear()
        self.cancel.clear()
        self._set("preparing", f"Getting {document['name']} ready", document=document["name"], steps=[], warnings=[])
        threading.Thread(target=self._run, args=(document, options), daemon=True).start()
        return 200, {"session": self.state}

    def stop(self):
        if self.state["state"] not in ("preparing", "live"):
            return
        self._set("stopping", "Stopping")
        self.cancel.set()  # between preparing steps, the meeting thread sees this and ends
        if self.meeting:
            self.meeting.stop()
        if self.stt:
            self.stt.close()

    def _run(self, document, options):
        from app.copilot import Copilot
        from app.llm import get_live_llm, model_name
        from app.meeting import LiveMeeting, listen
        from app.mic import open_mic
        from app.prepare import prepare
        from app.stt import DeepgramSTT
        from app.voice import Whisperer

        whisperer = overlay = meeting = None
        try:
            self._step("Loading the search model")
            if not self._model_ready.wait(timeout=MODEL_WAIT_SECONDS) or self._embedder is None:
                raise PreCueError("The search model could not be loaded. Check your internet connection and restart.")
            retriever, facts = prepare(document_path(document["id"]), embedder=self._embedder, report=self._report)
            self._step("Connecting the live fact-checker")
            llm = get_live_llm()
            copilot = Copilot(retriever, facts, llm)

            if options.get("voice", True):
                self._step("Setting up whispers")
                whisperer = Whisperer(os.environ["DEEPGRAM_API_KEY"], options.get("device") or None, _volume())
            if options.get("overlay", True):
                overlay = start_overlay(self.screen.url)

            self.stt = DeepgramSTT(os.environ["DEEPGRAM_API_KEY"])
            self._step("Opening the microphone")
            meeting = LiveMeeting(copilot, self.screen, whisperer)
            audio = queue.Queue()
            with open_mic(audio):
                self.meeting = meeting
                self._step("")  # marks the last step done
                if not self.cancel.is_set():
                    self._set("live", f"{len(retriever.chunks)} parts and {len(facts)} facts ready",
                              started=time.time(), model=model_name(llm))
                    self._listen_until_stopped(listen, audio, meeting)
            final = ("idle", "Meeting ended")
        except Cancelled:
            final = ("idle", "Meeting ended")
        except PreCueError as error:
            final = ("error", str(error))
        except Exception as error:  # never leave the page waiting forever
            final = ("error", f"Something went wrong: {error}")
        finally:
            if meeting:
                meeting.stop()
            if self.stt:
                self.stt.close()
            if whisperer:
                whisperer.stop()
            if overlay:
                stop_overlay(overlay)
            self.meeting = self.stt = None
            self.lock.release()
        # Only now, with the mic, pop-up and lock all free, tell the page: it may start again at once.
        self._set(*final)

    def _listen_until_stopped(self, listen, audio, meeting):
        """Listen in a thread of its own, and return as soon as the meeting stops.

        Opening the Deepgram connection sometimes takes up to 8 seconds and cannot be cut short.
        Waiting for it would keep the mic on after you press End. So we do not wait: the mic and
        the pop-up close at once, and a connection still opening closes itself when it is ready.
        """
        errors = []

        def run():
            try:
                listen(self.stt, audio, meeting)
            except PreCueError as error:  # e.g. gave up reconnecting
                errors.append(error)
            finally:
                meeting.stop()  # listening ended (stopped or failed), so the meeting is over

        threading.Thread(target=run, daemon=True).start()
        meeting.stopping.wait()
        if errors:
            raise errors[0]

    # ---------------- progress for the page ----------------

    def _report(self, kind, text):
        if self.cancel.is_set():
            raise Cancelled()  # Stop pressed while the facts table is being built
        if kind == "step":
            self._step(text)
        elif kind == "progress":
            steps = self.state["steps"]
            if steps:
                steps[-1]["progress"] = text
            self._set(self.state["state"], self.state["text"])
        elif kind == "warning":
            self._set(self.state["state"], self.state["text"], warnings=self.state["warnings"] + [text])

    def _step(self, text):
        """The previous step is done; this one starts. Ends the meeting thread if Stop was pressed."""
        if self.cancel.is_set():
            raise Cancelled()
        steps = [{**step, "done": True} for step in self.state["steps"]]
        if text:
            steps.append({"text": text, "done": False})
        self._set(self.state["state"], self.state["text"], steps=steps)

    def _set(self, state, text, **more):
        self.state = {**self.state, **more, "state": state, "text": text}
        if self.screen:
            self.screen.send({"type": "session", **self.state})

    def _load_model(self):
        try:
            from app.embedder import Embedder
            from app.prepare import prepare  # noqa: F401  (imports LangChain now, not when Start is pressed)

            self._embedder = Embedder()
        except PreCueError as error:
            print(f"Could not load the search model: {error}")
        finally:
            self._model_ready.set()


def documents():
    """The documents to choose from: your uploads (newest first), then the samples."""
    uploads = sorted(UPLOADS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if UPLOADS.exists() else []
    found = [document_entry(p, "uploads") for p in uploads if p.suffix.lower() in SUPPORTED_FILES]
    found += [document_entry(SAMPLES / name, "samples") for name in SAMPLE_DOCUMENTS if (SAMPLES / name).exists()]
    return found


def document_entry(path, folder):
    """A document for the page. Only its name: the page never sees or sends a file path."""
    return {"id": f"{folder}/{path.name}", "name": path.name, "sample": folder == "samples",
            "kind": path.suffix.lower().lstrip("."), "size": path.stat().st_size}


def find_document(document_id):
    """The listed document with this id, or None. Only listed files can be opened."""
    return next((d for d in documents() if d["id"] == document_id), None)


def document_path(document_id):
    folder, name = document_id.split("/", 1)
    return (UPLOADS if folder == "uploads" else SAMPLES) / name


def output_devices():
    """The speakers and earphones whispers can play on, with the system default first."""
    try:
        default = sd.query_devices(kind="output")["name"]
        names = [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
    except Exception:  # no audio system (e.g. a server): whispers are simply not offered
        return []
    # Virtual devices (BlackHole, Teams Audio...) are neither: whispers sent there are heard by nobody.
    return [{"name": name, "default": name == default, "speaker": "speaker" in name.lower(),
             "earphones": bool(EARPHONES.search(name))}
            for name in sorted(set(names), key=lambda n: (n != default, not EARPHONES.search(n)))]


def _volume():
    try:
        return float(os.getenv("WHISPER_VOLUME", "0.6"))
    except ValueError:
        raise PreCueError("WHISPER_VOLUME in .env must be a number from 0 to 1, e.g. 0.6.") from None

