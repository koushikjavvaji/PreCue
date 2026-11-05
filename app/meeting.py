"""One live meeting: speech in -> decisions -> screen and whispers out.

Used by the app (app/session.py) and by scripts/live_copilot.py.

What runs at the same time (threads):
  listen()       : microphone -> Deepgram -> sentences            (never waits for the LLM)
  decide thread  : sentences and pauses -> decision engine -> screen + whispers
  pause thread   : notices when you have been quiet for PAUSE_SECONDS

stop() ends the meeting cleanly: listen() returns, and both threads end.
"""

import queue
import threading
import time

from app.errors import ConnectionLostError, PreCueError
from app.voice import output_device_name

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
        self.stopping = threading.Event()
        threading.Thread(target=self._decide_forever, daemon=True).start()
        threading.Thread(target=self._watch_pauses, daemon=True).start()

    def stop(self):
        self.stopping.set()
        self.events.put(None)  # wake the decide thread so it can end

    def on_speech(self, event):
        """Called for every event from Deepgram."""
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
        while not self.stopping.is_set():
            time.sleep(0.5)
            if self.waiting_for_pause and time.time() - self.last_speech > PAUSE_SECONDS:
                self.waiting_for_pause = False
                self.events.put(("pause", "", time.time()))

    def _decide_forever(self):
        while not self.stopping.is_set():
            item = self.events.get()
            if item is None:
                break
            event, text, when = item

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
            verdict = state.get("verdict")
            if event == "sentence" and verdict:
                # The page marks the sentence: verified, wrong, or not in the document.
                self.show({"type": "verdict", "text": text, "verdict": verdict["verdict"]})
            if state["action"] in ("alert", "suggest", "skipped"):
                self.show({"type": state["action"], "text": state["message"], "time": now()})
            if state["action"] in ("alert", "suggest") and self.whisperer:
                self.whisperer.say(state["spoken"])


def listen(stt, audio_queue, meeting):
    """Run speech-to-text until the meeting stops. Reconnect when the network drops."""
    failures = 0
    while not meeting.stopping.is_set():
        meeting.status("Listening", "listening")
        try:
            for event in stt.listen(audio_queue):
                failures = 0  # we are receiving again, so the connection is healthy
                meeting.on_speech(event)
            reason = "Deepgram closed the connection"
        except ConnectionLostError as error:
            reason = str(error)
        if meeting.stopping.is_set():
            return  # we closed the connection ourselves

        failures += 1
        if failures > MAX_RECONNECTS:
            raise PreCueError(f"Gave up after {MAX_RECONNECTS} reconnects. Last problem: {reason}")
        wait = min(2**failures, 30)  # 2, 4, 8, 16, 30 seconds
        meeting.status(f"Connection lost, reconnecting in {wait}s...", "reconnecting")
        pause(wait)
        while not audio_queue.empty():  # drop the audio from the outage; we want to be live again
            audio_queue.get_nowait()


def pause(seconds):
    """Wait before reconnecting. (A function of its own, so tests can skip the waiting.)"""
    time.sleep(seconds)


def now():
    return time.strftime("%H:%M:%S")
