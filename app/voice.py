"""Whispers: turn a short text into speech and play it in the earphones.

  text -> Deepgram Aura (text-to-speech, same API key as speech-to-text) -> audio
       -> played on the output device you choose (your earphones)

Which device? Set WHISPER_DEVICE in .env to part of its name, e.g. "AirPods"
or "Headphones". List your devices with:  python -m sounddevice

Why earphones matter: if the whisper plays on the laptop speakers, the
microphone hears it and the meeting hears it too. So we warn about that,
and (on speakers only) the live script ignores the mic while a whisper plays.

Whispers play one at a time in a background thread, so the meeting
never waits for audio.
"""

import json
import queue
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd

from app.errors import PreCueError
from app.stt import SSL_CONTEXT

TTS_URL = "https://api.deepgram.com/v1/speak"
VOICE = "aura-2-thalia-en"
SAMPLE_RATE = 24000
CHUNK_BYTES = 4800  # 0.1 s of audio (24,000 samples per second x 2 bytes each / 10)
ECHO_GUARD_SECONDS = 0.5  # keep ignoring the mic a little after the whisper ends
MAX_WAIT_SECONDS = 8  # a whisper still waiting after this long is dropped (the moment has passed)


class Whisperer:
    def __init__(self, api_key, device_name=None, volume=0.6):
        self.api_key = api_key
        self.device = find_output_device(device_name)
        self.volume = volume
        self.quiet_until = 0  # time until which the mic should be ignored
        self.stopping = False
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._play_forever, daemon=True)
        self.thread.start()

    def say(self, text):
        """Queue a whisper. Returns at once; the audio plays in the background."""
        if text:
            self.queue.put((text, time.time()))

    def stop(self):
        """Stop playing (at the end of the meeting). Cuts a whisper short and closes the audio device."""
        self.stopping = True
        self.queue.put(None)  # wake the thread if it is waiting for a whisper
        self.thread.join(timeout=2)

    def is_speaking(self):
        """True while a whisper plays (and a moment after), so the mic can ignore it."""
        return time.time() < self.quiet_until or not self.queue.empty()

    def _play_forever(self):
        while not self.stopping:
            item = self.queue.get()
            if item is None:
                break
            text, queued_at = item
            if time.time() - queued_at > MAX_WAIT_SECONDS:
                print(f"[whisper dropped, too late: {text}]")  # an old correction would only confuse
                continue
            try:
                self._speak(text)
            except PreCueError as error:
                print(f"[whisper failed: {error}]")  # a failed whisper must never stop the meeting

    def _speak(self, text):
        """Stream the audio from Deepgram and play each piece as soon as it arrives.

        Waiting for the whole clip takes ~2.3 s; the first piece arrives after ~1.1 s.
        """
        url = f"{TTS_URL}?model={VOICE}&encoding=linear16&sample_rate={SAMPLE_RATE}&container=none"
        request = urllib.request.Request(
            url,
            data=json.dumps({"text": text}).encode(),
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
        )
        self.quiet_until = time.time() + 60  # ignore the mic while we speak (corrected at the end)
        try:
            with (
                urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=10) as response,
                sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=self.device) as speaker,
            ):
                while not self.stopping and (chunk := response.read(CHUNK_BYTES)):
                    samples = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype=np.int16)  # 2 bytes per sample
                    speaker.write((samples * self.volume).astype(np.int16).tobytes())  # quieter = a whisper
        except urllib.error.HTTPError as error:
            raise PreCueError(f"Deepgram text-to-speech refused the request (HTTP {error.code}).") from error
        except sd.PortAudioError as error:  # e.g. the earphones were unplugged
            raise PreCueError(f"Could not play audio on the whisper device: {error}") from error
        except OSError as error:  # no internet, timeout
            raise PreCueError(f"Could not reach Deepgram text-to-speech: {error}") from error
        finally:
            self.quiet_until = time.time() + ECHO_GUARD_SECONDS


def find_output_device(name):
    """Return the device number whose name contains `name`, or None for the system default."""
    if not name:
        return None
    outputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
    for index, device_name in outputs:
        if name.lower() in device_name.lower():
            return index
    names = ", ".join(device_name for _, device_name in outputs)
    raise PreCueError(f"No output device matches WHISPER_DEVICE='{name}'. Available: {names}")


def output_device_name(device):
    """The name of the device whispers will play on (for a startup message)."""
    if device is None:
        device = sd.default.device[1]
    return sd.query_devices(device)["name"]
