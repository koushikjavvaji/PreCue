"""PreCue's own errors. They carry a clear message that we can show to the user."""


class PreCueError(Exception):
    """Base for all PreCue errors."""


class STTError(PreCueError):
    """Speech-to-text failed (bad key, no internet, connection lost...)."""


class MicError(PreCueError):
    """The microphone could not be used."""


class RateLimitError(PreCueError):
    """An LLM's free-tier limit was reached. Waiting (a minute, or a day) helps."""


class ConnectionLostError(STTError):
    """The network dropped or Deepgram could not be reached. Worth trying again."""
