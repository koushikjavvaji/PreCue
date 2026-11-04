"""PreCue's own errors. They carry a clear message that we can show to the user."""


class PreCueError(Exception):
    """Base for all PreCue errors."""


class STTError(PreCueError):
    """Speech-to-text failed (bad key, no internet, connection lost...)."""


class MicError(PreCueError):
    """The microphone could not be used."""


class RateLimitError(PreCueError):
    """An LLM's free-tier limit was reached. Waiting (a minute, or a day) helps.

    daily=True: the daily limit, so waiting a minute will not help.
    """

    def __init__(self, message, daily=False):
        super().__init__(message)
        self.daily = daily


class ConnectionLostError(STTError):
    """The network dropped or Deepgram could not be reached. Worth trying again."""
