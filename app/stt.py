"""Speech-to-text using Deepgram."""

import json
import ssl
import threading
from urllib.parse import urlencode

import certifi
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from app.errors import STTError

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

# Python on macOS has no trusted certificates by default, so we use certifi's list.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class DeepgramSTT:
    def __init__(self, api_key, model="nova-3"):
        if not api_key:
            raise STTError("No Deepgram API key given.")
        self.api_key = api_key
        self.model = model

    def listen(self, audio_queue, keywords=()):
        """Send audio to Deepgram and give back events one by one.

        Each event is a dict:
          {"type": "speech_started"}
          {"type": "speech_ended"}
          {"type": "transcript", "text": "...", "is_final": True or False}

        Raises STTError if something goes wrong.
        """
        connection = self._connect(keywords)

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
            raise STTError(f"Lost the connection to Deepgram: {error}") from error
        finally:
            connection.close()

    def _connect(self, keywords):
        """Open the live connection. Turn common failures into clear messages."""
        try:
            return connect(
                self._make_url(keywords),
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
            raise STTError(
                f"Could not reach Deepgram: {error}. Check your internet connection."
            ) from error

    def _make_url(self, keywords):
        options = [
            ("model", self.model),
            ("encoding", "linear16"),
            ("sample_rate", 16000),
            ("channels", 1),
            ("smart_format", "true"),  # numbers as digits: "10", not "ten"
            ("interim_results", "true"),  # get partial text while still speaking
            ("vad_events", "true"),  # tell us when speech starts
            ("utterance_end_ms", 1000),  # tell us after 1 second of silence
        ]
        for word in keywords:  # boost words from the user's documents
            options.append(("keyterm", word))
        return f"{DEEPGRAM_URL}?{urlencode(options)}"

    def _send_audio(self, connection, audio_queue):
        try:
            while True:
                connection.send(audio_queue.get())
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
                return {"type": "transcript", "text": text, "is_final": data["is_final"]}

        return None
