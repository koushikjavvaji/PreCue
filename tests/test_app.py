"""The app (scripts/app.py): its page, its /api/ and the meeting it starts.
The search model, LLM, Deepgram, mic and pop-up are replaced by fakes."""

import json
import os
import socket
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

import app.llm
import app.meeting
import app.mic
import app.prepare
import app.session as session
import app.settings as settings
import app.stt
import app.voice
from app.screen import LiveScreen


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(condition, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """Uploads, samples and .env in a temporary folder, and all three keys set."""
    monkeypatch.setattr(session, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(session, "SAMPLES", tmp_path / "samples")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "sample_report.txt").write_text("# Report\n\n## 1. Money\n\nRevenue was ₹10 crore.\n", encoding="utf-8")
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    for name in settings.KEYS:
        monkeypatch.setenv(name, "test-key")
    monkeypatch.setattr(session, "check_key", lambda name: (True, "Connected"))
    return tmp_path


@pytest.fixture
def server(folders):
    precue = session.PreCueApp(load_model=False)
    precue._embedder = object()  # a stand-in: the fake prepare below never uses it
    precue._model_ready.set()
    screen = LiveScreen(port=free_port(), api=precue)
    precue.screen = screen
    yield precue, screen
    precue.stop()
    screen.server.shutdown()


def call(screen, path, body=None, origin="own", kind="application/json", headers=None):
    """GET (body None) or POST to the app, like the page does. Returns (status, data)."""
    all_headers = dict(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        all_headers["Content-Type"] = kind
        if origin:
            all_headers["Origin"] = screen.url if origin == "own" else origin
    request = urllib.request.Request(screen.url + path, data=data, headers=all_headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        text = error.read()
        try:
            return error.code, json.loads(text)
        except ValueError:
            return error.code, {}


# ---------------- the page ----------------

def test_app_page_is_served_and_safe(server):
    _, screen = server
    with urllib.request.urlopen(screen.url + "/") as response:
        page = response.read().decode()
        assert "default-src 'none'" in response.headers["Content-Security-Policy"]
    assert "/api/start" in page and "EventSource" in page
    assert "innerHTML" not in page.replace("never innerHTML", "")  # names and speech are never run as code
    assert "EventSource" in urllib.request.urlopen(screen.url + "/live").read().decode()  # the plain live page


def test_state_never_shows_keys_or_file_paths(server, folders):
    _, screen = server
    status, data = call(screen, "/api/state")
    assert status == 200
    text = json.dumps(data)
    assert "test-key" not in text and str(folders) not in text
    assert [k["set"] for k in data["keys"]] == [True, True, True]
    assert data["documents"][0]["id"] == "samples/sample_report.txt"


# ---------------- other websites cannot use the API ----------------

@pytest.mark.parametrize("origin, kind", [
    (None, "application/json"),                     # no Origin (not from a browser page)
    ("http://evil.example", "application/json"),    # another website
    ("http://127.0.0.1:1", "application/json"),     # another program on this computer
    ("own", "text/plain"),                          # a type a plain web form can send without asking
    ("own", "application/x-www-form-urlencoded"),
])
def test_changes_only_from_our_own_page(server, origin, kind):
    _, screen = server
    assert call(screen, "/api/stop", {}, origin=origin, kind=kind)[0] == 403


def test_wrong_host_name_is_refused(server):
    _, screen = server
    status, _ = call(screen, "/api/state", headers={"Host": "evil.example:8765"})
    assert status == 403


def test_too_big_body_is_refused(server, monkeypatch):
    precue, screen = server
    monkeypatch.setattr(precue, "MAX_UPLOAD_BYTES", 10)
    status, data = call(screen, "/api/upload", b"x" * 100, kind="application/octet-stream",
                        headers={"X-File-Name": "big.txt"})
    assert status == 413


# ---------------- keys ----------------

def test_keys_are_saved_privately_and_other_lines_kept(folders):
    env = folders / ".env"
    env.write_text("# my settings\nGROQ_API_KEY=old\nWHISPER_VOLUME=0.4\n", encoding="utf-8")
    settings.save_keys({"GROQ_API_KEY": "gsk_new", "GEMINI_API_KEY": "AIza_new"}, env_file=env)
    assert env.read_text() == "# my settings\nGROQ_API_KEY=gsk_new\nWHISPER_VOLUME=0.4\nGEMINI_API_KEY=AIza_new\n"
    if sys.platform != "win32":  # Windows has no such permission bits
        assert stat.S_IMODE(os.stat(env).st_mode) == 0o600
    assert os.environ["GROQ_API_KEY"] == "gsk_new"


@pytest.mark.parametrize("keys", [
    {"OPENAI_API_KEY": "x"},                    # not one of ours
    {"GROQ_API_KEY": "two words"},
    {"GROQ_API_KEY": "abc\nLIVE_LLM=gemini"},   # a new line would add another setting
    {"GROQ_API_KEY": "x" * 400},
])
def test_strange_keys_are_refused(folders, keys):
    with pytest.raises(ValueError):
        settings.save_keys(keys, env_file=folders / ".env")
    assert not (folders / ".env").exists()


def test_keys_api_saves_and_reports_without_the_key(server, folders):
    _, screen = server
    status, data = call(screen, "/api/keys", {"keys": {"GROQ_API_KEY": "gsk_secret123"}})
    assert status == 200 and "gsk_secret123" not in json.dumps(data)
    assert "GROQ_API_KEY=gsk_secret123" in (folders / ".env").read_text()


@pytest.mark.parametrize("code, works", [(200, True), (401, False), (403, False), (429, True), (500, False)])
def test_key_check_reads_the_providers_answer(monkeypatch, code, works):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")

    class Answer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return b"{}"

    def fake_open(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer gsk_x"
        if code != 200:
            raise urllib.error.HTTPError(request.full_url, code, "", {}, None)
        return Answer()

    monkeypatch.setattr(settings.urllib.request, "urlopen", fake_open)
    assert settings.check_key("GROQ_API_KEY")[0] is works


# ---------------- documents ----------------

def test_upload_keeps_only_a_safe_name(server, folders):
    _, screen = server
    status, data = call(screen, "/api/upload", "# Plan\n\nWe grew 5%.".encode(), kind="application/octet-stream",
                        headers={"X-File-Name": "..%2F..%2Fevil%20plan%3B.md"})
    assert status == 200
    assert data["document"]["id"] == "uploads/evil plan_.md"
    assert (folders / "uploads" / "evil plan_.md").exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(folders / "uploads" / "evil plan_.md").st_mode) == 0o600
    assert data["documents"][0]["id"] == "uploads/evil plan_.md"  # newest first, before the samples


@pytest.mark.parametrize("name, body", [("run.sh", b"echo hi"), ("report.txt", b""), ("noext", b"text")])
def test_bad_uploads_are_refused(server, name, body):
    _, screen = server
    status, data = call(screen, "/api/upload", body, kind="application/octet-stream", headers={"X-File-Name": name})
    assert status == 400 and data["error"]


def test_only_listed_documents_can_be_started(server):
    _, screen = server
    for document in ["../../.env", "samples/../../.env", "/etc/passwd", "uploads/missing.txt"]:
        status, data = call(screen, "/api/start", {"document": document})
        assert status == 400 and "Choose a document" in data["error"]


def test_start_needs_all_keys(server, monkeypatch):
    _, screen = server
    monkeypatch.delenv("GROQ_API_KEY")
    status, data = call(screen, "/api/start", {"document": "samples/sample_report.txt"})
    assert status == 400 and "Groq" in data["error"]


# ---------------- the meeting ----------------

class FakeSTT:
    """Hears one sentence, then waits until the meeting closes it (like the real Deepgram connection)."""

    def __init__(self, key):
        self.closed = threading.Event()

    def listen(self, audio):
        yield {"type": "transcript", "text": "Revenue was 10 crore.", "is_final": True, "speech_final": True}
        self.closed.wait(10)

    def close(self):
        self.closed.set()


class FakeCopilot:
    def __init__(self, retriever, facts, llm):
        pass

    def handle(self, event, text="", late=False):
        return {"action": "silent", "message": "", "spoken": "", "verdict": {"verdict": "correct"}}


class FakeRetriever:
    chunks = [1, 2, 3]


@contextmanager
def fake_mic(audio):
    yield


@pytest.fixture
def fakes(monkeypatch):
    slow = threading.Event()  # set it to let a "slow" prepare finish
    slow.set()

    def fake_prepare(path, embedder=None, report=None):
        report("step", "Reading and indexing the document")
        report("warning", "'3. People' has text that looks like instructions to an AI.")
        slow.wait(10)
        report("step", "Building the facts table")
        report("progress", "1/1")
        return FakeRetriever(), [1, 2]

    monkeypatch.setattr(app.prepare, "prepare", fake_prepare)
    monkeypatch.setattr(app.llm, "get_live_llm", lambda: "fake-llm")
    monkeypatch.setattr(app.llm, "model_name", lambda llm: "fake-llm")
    monkeypatch.setattr(app.copilot, "Copilot", FakeCopilot, raising=False)
    monkeypatch.setattr(app.stt, "DeepgramSTT", FakeSTT)
    monkeypatch.setattr(app.mic, "open_mic", fake_mic)
    monkeypatch.setattr(session, "start_overlay", lambda url: None)
    monkeypatch.setattr(app.meeting, "output_device_name", lambda device: "AirPods")
    return slow


import app.copilot  # noqa: E402  (after the fakes' helpers, so the fixture can replace Copilot)


def test_meeting_starts_goes_live_and_stops(server, fakes):
    precue, screen = server
    seen = []
    screen.clients.append(_Recorder(seen))

    status, data = call(screen, "/api/start", {"document": "samples/sample_report.txt", "voice": False, "overlay": True})
    assert status == 200 and data["session"]["state"] == "preparing"
    assert wait_for(lambda: precue.state["state"] == "live"), precue.state
    assert precue.state["text"] == "3 parts and 2 facts ready"
    assert all(step["done"] for step in precue.state["steps"])
    assert precue.state["warnings"]  # the injection warning reaches the page

    assert wait_for(lambda: any(e["type"] == "verdict" for e in seen))
    assert {"type": "verdict", "text": "Revenue was 10 crore.", "verdict": "correct"} in seen
    assert call(screen, "/api/start", {"document": "samples/sample_report.txt"})[1]["error"] == "A meeting is already running."

    assert call(screen, "/api/stop", {})[0] == 200
    assert wait_for(lambda: precue.state["state"] == "idle"), precue.state
    assert precue.state["text"] == "Meeting ended"
    assert precue.lock.acquire(blocking=False)  # free for the next meeting
    precue.lock.release()


def test_stop_while_preparing_ends_cleanly(server, fakes):
    precue, screen = server
    fakes.clear()  # prepare is "slow" now
    call(screen, "/api/start", {"document": "samples/sample_report.txt", "voice": False})
    assert wait_for(lambda: len(precue.state["steps"]) >= 2)
    call(screen, "/api/stop", {})
    fakes.set()
    assert wait_for(lambda: precue.state["state"] == "idle"), precue.state
    assert precue.meeting is None


def test_a_problem_is_shown_and_the_app_can_start_again(server, fakes, monkeypatch):
    precue, screen = server

    def broken_mic(audio):
        raise app.mic.MicError("Could not open the microphone: no device.")

    monkeypatch.setattr(app.mic, "open_mic", broken_mic)
    call(screen, "/api/start", {"document": "samples/sample_report.txt", "voice": False})
    assert wait_for(lambda: precue.state["state"] == "error")
    assert "microphone" in precue.state["text"]

    monkeypatch.setattr(app.mic, "open_mic", fake_mic)
    assert call(screen, "/api/start", {"document": "samples/sample_report.txt", "voice": False})[0] == 200
    assert wait_for(lambda: precue.state["state"] == "live")


class _Recorder:
    """Stands in for an open page: records every event the screen sends."""

    def __init__(self, seen):
        self.seen = seen

    def full(self):
        return False

    def put(self, event):
        self.seen.append(event)


def test_a_listening_failure_ends_the_meeting_with_its_message(server, fakes, monkeypatch):
    precue, screen = server

    class RejectedKey(FakeSTT):
        def listen(self, audio):
            raise app.stt.STTError("Deepgram rejected your API key.")
            yield  # a generator, like the real one

    monkeypatch.setattr(app.stt, "DeepgramSTT", RejectedKey)
    call(screen, "/api/start", {"document": "samples/sample_report.txt", "voice": False})
    assert wait_for(lambda: precue.state["state"] == "error")
    assert precue.state["text"] == "Deepgram rejected your API key."
