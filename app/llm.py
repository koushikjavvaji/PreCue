"""Creates the LLMs PreCue uses, in one place.

  Gemini (free tier): offline work before the meeting (context, facts table).
  Groq (free tier):   live work during the meeting, because it answers fast.

Every LLM is a LangChain chat model, so the rest of the code does not care
which one it is. Set LIVE_LLM=gemini in .env to use Gemini for live work too.
"""

import os

import groq
from langchain_core.exceptions import ModelRateLimitError, OutputParserException
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_groq import ChatGroq
from pydantic import ValidationError

from app.errors import PreCueError

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # change with GEMINI_MODEL in .env
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # change with GROQ_MODEL in .env

# Every error an LLM call can raise that we know how to report nicely.
LLM_ERRORS = (ChatGoogleGenerativeAIError, groq.GroqError, OutputParserException, ValidationError)

_rate_limiters = {}  # one limiter per provider, shared by every call in this run


def get_gemini():
    """Return a Gemini chat model. Reads GEMINI_API_KEY (and optional settings) from .env."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        google_api_key=_api_key("GEMINI_API_KEY", "Gemini"),
        rate_limiter=_rate_limiter("GEMINI_REQUESTS_PER_MINUTE", default=15),
        max_retries=3,  # retry when Google is busy (503)
        timeout=60,
    )


def get_groq():
    """Return a Groq chat model. Reads GROQ_API_KEY (and optional settings) from .env."""
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(
        model=model,
        api_key=_api_key("GROQ_API_KEY", "Groq"),
        rate_limiter=_rate_limiter("GROQ_REQUESTS_PER_MINUTE", default=30),
        # gpt-oss models "think" before answering; a little thinking is enough and much faster
        reasoning_effort="low" if "gpt-oss" in model else None,
        max_retries=2,
        timeout=15,  # live work: better to skip one check than to hang
    )


def get_live_llm():
    """The LLM for live fact-checking: Groq by default, or Gemini with LIVE_LLM=gemini."""
    choice = os.getenv("LIVE_LLM", "groq").lower()
    if choice == "groq":
        return get_groq()
    if choice == "gemini":
        return get_gemini()
    raise PreCueError("LIVE_LLM in .env must be 'groq' or 'gemini'.")


def structured(llm, schema):
    """The LLM, set up to answer in the shape of `schema` (a pydantic class).

    Groq: "strict" JSON schema mode. Groq forces the model to fill in exactly this shape, so the answer always fits.
    The default (a tool call) failed now and then: the model wrote plain JSON instead of calling the tool.
    """
    if isinstance(llm, ChatGroq):
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    return llm.with_structured_output(schema)


def model_name(llm):
    """The model name of a LangChain chat model (Gemini and Groq store it differently)."""
    return getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")


def llm_error(error, task):
    """Turn an LLM error into a short PreCueError. task: what we were doing, e.g. "adding context"."""
    if isinstance(error, (ModelRateLimitError, groq.RateLimitError)):
        return PreCueError(
            f"LLM free-tier limit reached while {task}. Wait a minute (or until tomorrow for the "
            "daily limit) and run again. Finished work is cached, so nothing is lost."
        )
    return PreCueError(f"LLM failed while {task}: {str(error)[:300]}")


def _api_key(name, provider):
    key = os.getenv(name)
    if not key or key == "your_key_here":
        raise PreCueError(f"Add your {provider} key to the .env file ({name}=...).")
    return key


def _rate_limiter(setting, default):
    """A limiter that spaces calls out so we never go over the per-minute limit."""
    if setting not in _rate_limiters:
        try:
            per_minute = int(os.getenv(setting, default))
        except ValueError:
            raise PreCueError(f"{setting} in .env must be a whole number.") from None
        _rate_limiters[setting] = InMemoryRateLimiter(
            requests_per_second=per_minute / 60, max_bucket_size=1
        )
    return _rate_limiters[setting]
