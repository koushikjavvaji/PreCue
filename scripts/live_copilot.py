"""PreCue live: listens to you in a meeting, checks your numbers, alerts and suggests.

Run from the project folder:
  python -m scripts.live_copilot samples/sample_report.txt
  python -m scripts.live_copilot samples/sample_report.txt --no-voice     (screen only)
  python -m scripts.live_copilot samples/sample_report.txt --no-screen    (terminal + whispers)
  python -m scripts.live_copilot samples/sample_report.txt --no-overlay   (browser tab instead of pop-ups)

Alerts pop up in a small floating window on top of your meeting (overlay.py),
hidden from screen sharing. The full page with the transcript is at the
printed "Live screen" address.

What runs at the same time (threads):
  main thread    : microphone -> Deepgram -> sentences            (never waits for the LLM)
  decide thread  : sentences and pauses -> decision engine -> screen + whispers
  pause thread   : notices when you have been quiet for PAUSE_SECONDS
  whisper thread : plays whispers one at a time            (inside voice.py)
  screen threads : the local web page                      (inside screen.py)
  overlay        : its own process, the floating pop-up window (overlay.py)
"""

import argparse
import os
import queue
import sys
import threading
import time
import webbrowser

from dotenv import load_dotenv

from app.copilot import Copilot
from app.errors import ConnectionLostError, PreCueError
from app.index import MODES
from app.llm import get_live_llm, model_name
from app.mic import open_mic
from app.overlay import start_overlay, stop_overlay
from app.prepare import prepare
from app.screen import LiveScreen
from app.stt import DeepgramSTT
from app.voice import Whisperer, output_device_name

PAUSE_SECONDS = 4  # quiet this long after speaking = maybe stuck, so suggest the next point
MAX_RECONNECTS = 5  # give up after this many failed reconnects in a row
MAX_DELAY_SECONDS = 8  # an alert later than this after the sentence would only confuse


class LiveMeeting:
    """Connects the parts: speech in -> decisions -> screen and whispers out."""

    def __init__(self, copilot, screen, whisperer):
        self.copilot = copilot
        self.screen = screen
        self.whisperer = whisperer
        # On speakers, the mic hears our own whispers, so we ignore the mic while one plays.
        # With earphones there is no echo, and ignoring the mic would lose the speaker's words.
        self.echo_guard = whisperer is not None and "speaker" in output_device_name(whisperer.device).lower()
        self.events = queue.Queue()  # sentences and pauses waiting for the decision engine
        self.sentence = []  # final pieces of the sentence being spoken right now
        self.last_speech = time.time()
        self.waiting_for_pause = False  # True after a sentence, until we report a pause
        threading.Thread(target=self._decide_forever, daemon=True).start()
        threading.Thread(target=self._watch_pauses, daemon=True).start()

    def on_speech(self, event):
        """Called for every event from Deepgram (main thread)."""
        if self.echo_guard and self.whisperer.is_speaking():
            return  # the mic is hearing our own whisper, not the speaker

        if event["type"] == "speech_started":
            self.last_speech = time.time()
        elif event["type"] == "speech_ended":
            self._finish_sentence()  # 1 second of silence: whatever we have is a sentence
        elif not event["is_final"]:
            self.last_speech = time.time()
            self.show({"type": "partial", "text": event["text"]})
        else:
            self.last_speech = time.time()
            self.sentence.append(event["text"])
            if event["speech_final"]:  # Deepgram heard the end of a sentence
                self._finish_sentence()

    def status(self, text, state):
        self.show({"type": "status", "text": text, "state": state})

    def show(self, event):
        """Send an event to the screen and print the important ones in the terminal."""
        if self.screen:
            self.screen.send(event)
        labels = {"final": "you  ", "alert": "CHECK", "suggest": "NEXT ", "skipped": "skip ", "status": "---- "}
        if event["type"] in labels:
            print(f"{labels[event['type']]} | {event['text']}")

    def _finish_sentence(self):
        if not self.sentence:
            return
        text = " ".join(self.sentence)
        self.sentence = []
        self.show({"type": "final", "text": text})
        self.events.put(("sentence", text, time.time()))
        self.waiting_for_pause = True

    def _watch_pauses(self):
        while True:
            time.sleep(0.5)
            if self.waiting_for_pause and time.time() - self.last_speech > PAUSE_SECONDS:
                self.waiting_for_pause = False
                self.events.put(("pause", "", time.time()))

    def _decide_forever(self):
        while True:
            event, text, when = self.events.get()

            # The LLM may be slow or rate-limited, so events can wait in the queue.
            if event == "pause" and time.time() - self.last_speech < PAUSE_SECONDS:
                continue  # the speaker started talking again: they are not stuck any more
            late = event == "sentence" and time.time() - when > MAX_DELAY_SECONDS
            if late:
                self.show({"type": "skipped", "text": f"Not checked (too late): {text}", "time": now()})

            try:
                state = self.copilot.handle(event, text, late=late)
            except Exception as error:  # a bug in one check must never stop the meeting: report it, go on
                self.show({"type": "skipped", "text": f"Internal error: {error}", "time": now()})
                continue
            if state["action"] in ("alert", "suggest", "skipped"):
                self.show({"type": state["action"], "text": state["message"], "time": now()})
            if state["action"] in ("alert", "suggest") and self.whisperer:
                self.whisperer.say(state["spoken"])


def listen(stt, audio_queue, meeting):
    """Run speech-to-text for the whole meeting. Reconnect when the network drops."""
    failures = 0
    while True:
        meeting.status("Listening", "listening")
        try:
            for event in stt.listen(audio_queue):
                failures = 0  # we are receiving again, so the connection is healthy
                meeting.on_speech(event)
            reason = "Deepgram closed the connection"
        except ConnectionLostError as error:
            reason = str(error)

        failures += 1
        if failures > MAX_RECONNECTS:
            raise PreCueError(f"Gave up after {MAX_RECONNECTS} reconnects. Last problem: {reason}")
        wait = min(2**failures, 30)  # 2, 4, 8, 16, 30 seconds
        meeting.status(f"Connection lost, reconnecting in {wait}s...", "reconnecting")
        time.sleep(wait)
        while not audio_queue.empty():  # drop the audio from the outage; we want to be live again
            audio_queue.get_nowait()


def now():
    return time.strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="PreCue live meeting copilot.")
    parser.add_argument("file", help="the document for this meeting (.txt or .md)")
    parser.add_argument("--mode", choices=MODES, default="plain")
    parser.add_argument("--no-voice", action="store_true", help="no whispers, screen only")
    parser.add_argument("--no-screen", action="store_true", help="no web page and no pop-ups")
    parser.add_argument("--no-overlay", action="store_true", help="open a browser tab instead of the pop-up window")
    args = parser.parse_args()

    load_dotenv()
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key or deepgram_key == "your_key_here":
        raise PreCueError("Add your Deepgram key to the .env file (DEEPGRAM_API_KEY=...).")

    print("Preparing the document...")
    retriever, facts = prepare(args.file, args.mode)
    llm = get_live_llm()
    copilot = Copilot(retriever, facts, llm)
    print(f"Ready: {len(retriever.chunks)} chunks, {len(facts)} facts, live LLM {model_name(llm)}")

    screen = None
    overlay = None
    if not args.no_screen:
        screen = LiveScreen()
        print(f"Live screen: {screen.url}")
        if args.no_overlay or sys.platform != "darwin":  # the overlay is macOS only for now
            webbrowser.open(screen.url)
        else:
            overlay = start_overlay(screen.url)

    whisperer = None
    if not args.no_voice:
        try:
            volume = float(os.getenv("WHISPER_VOLUME", "0.6"))
        except ValueError:
            raise PreCueError("WHISPER_VOLUME in .env must be a number from 0 to 1, e.g. 0.6.") from None
        whisperer = Whisperer(deepgram_key, os.getenv("WHISPER_DEVICE"), volume)
        device = output_device_name(whisperer.device)
        print(f"Whispers play on: {device}")
        if "speaker" in device.lower():
            print("WARNING: whispers will play on your speakers, so the meeting may hear them. "
                  "Plug in earphones and set WHISPER_DEVICE in .env (or use --no-voice).")

    meeting = LiveMeeting(copilot, screen, whisperer)
    audio_queue = queue.Queue()
    print("\nSpeak now. Press Ctrl+C to stop.\n")
    try:
        with open_mic(audio_queue):
            listen(DeepgramSTT(deepgram_key), audio_queue, meeting)
    finally:
        if whisperer:
            whisperer.stop()  # end a whisper cleanly instead of cutting the audio device off
        if overlay:
            stop_overlay(overlay)  # close the pop-up window with the app


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}")
    except KeyboardInterrupt:
        print("\nStopped.")
