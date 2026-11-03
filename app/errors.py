"""PreCue's own errors. They carry a clear message that we can show to the user."""


class PreCueError(Exception):
    """Base for all PreCue errors."""


class STTError(PreCueError):
    """Speech-to-text failed (bad key, no internet, connection lost...)."""


class MicError(PreCueError):
    """The microphone could not be used."""
