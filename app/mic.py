"""Reads audio from the microphone."""

from contextlib import contextmanager

import sounddevice as sd

from app.errors import MicError

SAMPLE_RATE = 16000  # Deepgram needs 16,000 audio samples per second
CHUNK_SECONDS = 0.1  # send audio in small pieces of 0.1 seconds


@contextmanager
def open_mic(audio_queue):
    """Open the mic. Every 0.1 seconds, a piece of audio is put in audio_queue.

    Use it with `with`, so the mic is always closed at the end.
    """

    def on_audio(data, frames, time, status):
        audio_queue.put(bytes(data))

    try:
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,  # mono
            dtype="int16",  # the audio format Deepgram expects
            blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
            callback=on_audio,
        )
        stream.start()
    except sd.PortAudioError as error:
        raise MicError(
            f"Could not open the microphone: {error}. "
            "Check that a mic is connected and your terminal has microphone permission."
        ) from error

    try:
        yield
    finally:
        stream.stop()
        stream.close()
