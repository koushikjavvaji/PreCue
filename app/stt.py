"""Speech-to-text using Deepgram."""

import json
import queue
import socket
import ssl
import threading
from urllib.parse import urlencode

import certifi
import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from app.errors import ConnectionLostError, STTError

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

# Python on macOS has no trusted certificates by default, so we use certifi's list.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Deepgram closes the connection after ~10 seconds without audio. If the mic
# stops sending (muted, laptop asleep), we send a small "KeepAlive" message instead.
KEEPALIVE_SECONDS = 3

# Deepgram sometimes holds a short phrase like "99 percent." without ending it, and glues
# it to the next sentence. So when OUR audio has been quiet this long after speech, we send
# "Finalize", which makes Deepgram return what it heard right away.
FINALIZE_AFTER_SECONDS = 0.8
SILENCE_LEVEL = 300  # audio quieter than this (average loudness, 16-bit) counts as silence
SAMPLE_RATE = 16000


class DeepgramSTT:
    def __init__(self, api_key, model="nova-3"):
        if not api_key:
            raise STTError("No Deepgram API key given.")
        self.api_key = api_key
        self.model = model
        self.connection = None
        self.closing = False

    def close(self):
        """End the live connection from another thread (when the meeting stops). listen() then ends.

        If the connection is still being opened, listen() sees `closing` right after and ends too.
        """
        self.closing = True
        if self.connection:
            # Cut the network connection: listen() ends at once. A polite close waits for Deepgram's
            # goodbye, which can take 8 seconds, and there is nothing left to hear.
            try:
                self.connection.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already closed

    def listen(self, audio_queue):
        """Send audio to Deepgram and give back events one by one.

        Each event is a dict:
          {"type": "speech_started"}
          {"type": "speech_ended"}
          {"type": "transcript", "text": "...", "is_final": True or False,
           "speech_final": True when the speaker finished a sentence}

        Raises ConnectionLostError if the network fails (worth retrying),
        or STTError for other problems (like a wrong API key).
        """
        connection = self.connection = self._connect()
        if self.closing:  # the meeting stopped while we were connecting
            connection.close()
            return

        # One thread sends audio, while this one receives the text.
        sender = threading.Thread(
            target=self._send_audio, args=(connection, audio_queue), daemon=True
        )
        sender.start()

        try:
            for message in connection:
                event = self._to_event(json.loads(message))
                if event:
                    yield event
        except ConnectionClosed as error:
            raise ConnectionLostError(f"Lost the connection to Deepgram: {error}") from error
        finally:
            connection.close()

    def _connect(self):
        """Open the live connection. Turn common failures into clear messages."""
        try:
            return connect(
                self._make_url(),
                additional_headers={"Authorization": f"Token {self.api_key}"},
                ssl=SSL_CONTEXT,
            )
        except InvalidStatus as error:
            code = error.response.status_code
            if code in (401, 403):
                raise STTError(
                    "Deepgram rejected your API key. Check DEEPGRAM_API_KEY in your .env file."
                ) from error
            raise STTError(f"Deepgram refused the connection (HTTP {code}).") from error
        except OSError as error:  # no internet, DNS problem, timeout, SSL problem
            raise ConnectionLostError(
                f"Could not reach Deepgram: {error}. Check your internet connection."
            ) from error

    def _make_url(self):
        options = [
            ("model", self.model),
            ("encoding", "linear16"),
            ("sample_rate", SAMPLE_RATE),
            ("channels", 1),
            ("smart_format", "true"),  # numbers as digits: "10", not "ten"
            ("interim_results", "true"),  # get partial text while still speaking
            ("vad_events", "true"),  # tell us when speech starts
            ("utterance_end_ms", 1000),  # tell us after 1 second of silence
            ("endpointing", 300),  # a sentence ends after 300 ms of silence (not at every tiny breath)
        ]
        return f"{DEEPGRAM_URL}?{urlencode(options)}"

    def _send_audio(self, connection, audio_queue):
        quiet_samples = 0  # how long the audio has been silent (counted in samples: whole numbers add up exactly)
        spoke = False  # there was speech since the last Finalize
        try:
            while True:
                try:
                    audio = audio_queue.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:  # no audio for a while: tell Deepgram we are still here
                    connection.send(json.dumps({"type": "KeepAlive"}))
                    continue
                connection.send(audio)

                if is_silent(audio):
                    quiet_samples += len(audio) // 2  # 2 bytes per sample
                else:
                    quiet_samples = 0
                    spoke = True
                if spoke and quiet_samples >= FINALIZE_AFTER_SECONDS * SAMPLE_RATE:
                    connection.send(json.dumps({"type": "Finalize"}))
                    spoke = False  # once per pause
        except ConnectionClosed:
            pass  # the connection ended, so we stop sending. The main loop reports why.

    def _to_event(self, data):
        """Turn Deepgram's message into our simple event. Return None to ignore."""
        kind = data.get("type")

        if kind == "SpeechStarted":
            return {"type": "speech_started"}

        if kind == "UtteranceEnd":
            return {"type": "speech_ended"}

        if kind == "Results":
            text = data["channel"]["alternatives"][0]["transcript"]
            if text:
                return {
                    "type": "transcript",
                    "text": text,
                    "is_final": data["is_final"],
                    # After our Finalize, the result is the end of what was said, so it ends the sentence too
                    "speech_final": data.get("speech_final", False) or data.get("from_finalize", False),
                }

        return None


def is_silent(audio):
    """True if a piece of 16-bit audio is quiet (root mean square loudness below SILENCE_LEVEL)."""
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    return samples.size == 0 or np.sqrt(np.mean(samples**2)) < SILENCE_LEVEL
