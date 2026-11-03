"""Creates the LLMs PreCue uses, in one place.

Today: Gemini (free tier) for offline work like contextual retrieval and facts.
Later: Groq for the fast live work. Because every LLM is a LangChain
chat model, the rest of the code does not care which one it is.
"""

import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError

from app.errors import PreCueError

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # fast and cheap; change with GEMINI_MODEL in .env
DEFAULT_REQUESTS_PER_MINUTE = 15  # the free-tier limit; change with GEMINI_REQUESTS_PER_MINUTE

_rate_limiter = None  # one limiter shared by every Gemini call in this run


def get_gemini():
    """Return a Gemini chat model. Reads GEMINI_API_KEY (and optional settings) from .env."""
    global _rate_limiter

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise PreCueError("Add your Gemini key to the .env file (GEMINI_API_KEY=...).")

    if _rate_limiter is None:
        try:
            per_minute = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", DEFAULT_REQUESTS_PER_MINUTE))
        except ValueError:
            raise PreCueError("GEMINI_REQUESTS_PER_MINUTE in .env must be a whole number.") from None
        # waits before each call when needed, so we never go over the per-minute limit
        _rate_limiter = InMemoryRateLimiter(requests_per_second=per_minute / 60, max_bucket_size=1)

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        google_api_key=api_key,
        rate_limiter=_rate_limiter,
        max_retries=3,  # retry when Google is busy (503)
        timeout=60,
    )


def gemini_error(error, task):
    """Turn a Gemini error into a short PreCueError. task: what we were doing, e.g. "adding context"."""
    if isinstance(error, GoogleRateLimitError):
        return PreCueError(
            f"Gemini free-tier limit reached while {task}. Wait a minute (or until tomorrow for the "
            "daily limit) and run again. Finished chunks are cached, so no work is lost."
        )
    return PreCueError(f"Gemini failed while {task}: {str(error)[:300]}")
