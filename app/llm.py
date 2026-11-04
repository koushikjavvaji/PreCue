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
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_groq import ChatGroq
from pydantic import ValidationError

from app.errors import PreCueError, RateLimitError

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # change with GEMINI_MODEL in .env
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # change with GROQ_MODEL in .env
# Used when the main Groq model hits its free-tier limit: each model has its own limit
# (200,000 tokens a day), so the backup doubles the checks a day. Change with GROQ_BACKUP_MODEL.
DEFAULT_GROQ_BACKUP_MODEL = "openai/gpt-oss-120b"

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


def get_groq(model=None, max_retries=2):
    """Return a Groq chat model. Reads GROQ_API_KEY (and optional settings) from .env."""
    model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(
        model=model,
        api_key=_api_key("GROQ_API_KEY", "Groq"),
        rate_limiter=_rate_limiter("GROQ_REQUESTS_PER_MINUTE", default=30),
        # gpt-oss models "think" before answering; a little thinking is enough and much faster
        reasoning_effort="low" if "gpt-oss" in model else None,
        max_retries=max_retries,
        timeout=15,  # live work: better to skip one check than to hang
    )


def get_live_llm():
    """The LLM for live fact-checking: Groq by default, or Gemini with LIVE_LLM=gemini."""
    choice = os.getenv("LIVE_LLM", "groq").lower()
    if choice == "groq":
        return groq_with_backup()
    if choice == "gemini":
        return get_gemini()
    raise PreCueError("LIVE_LLM in .env must be 'groq' or 'gemini'.")


def groq_with_backup():
    """The Groq model, with the backup model ready for when it fails (usually its free-tier limit)."""
    main_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    backup_model = os.getenv("GROQ_BACKUP_MODEL", DEFAULT_GROQ_BACKUP_MODEL)
    if not backup_model or backup_model == main_model:
        return get_groq(main_model)
    # No retries for the main model: waiting for it would make the alert late, the backup answers at once.
    main = get_groq(main_model, max_retries=0)
    return main.with_fallbacks([get_groq(backup_model)], exceptions_to_handle=(groq.GroqError,))


def structured(llm, schema):
    """The LLM, set up to answer in the shape of `schema` (a pydantic class).

    Groq: "strict" JSON schema mode. Groq forces the model to fill in exactly this shape, so the answer always fits.
    The default (a tool call) failed now and then: the model wrote plain JSON instead of calling the tool.
    A model with a backup: both are set up the same way, and the backup answers when the main one fails.
    """
    if isinstance(llm, RunnableWithFallbacks):
        main = structured(llm.runnable, schema)
        backup = structured(llm.fallbacks[0], schema)

        def ask(prompt):
            try:
                return main.invoke(prompt)
            except groq.GroqError:
                # If the backup fails too, ITS error is raised: it says how long to really wait.
                # (LangChain's own fallbacks raise the main model's error, e.g. "daily limit",
                # when the backup only needs a few seconds.)
                return backup.invoke(prompt)

        return RunnableLambda(ask)
    if isinstance(llm, ChatGroq):
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    return llm.with_structured_output(schema)


def model_name(llm):
    """The model name of a LangChain chat model (Gemini and Groq store it differently)."""
    if isinstance(llm, RunnableWithFallbacks):
        backups = ", ".join(model_name(backup) for backup in llm.fallbacks)
        return f"{model_name(llm.runnable)} (backup: {backups})"
    return getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")


def llm_error(error, task):
    """Turn an LLM error into a short PreCueError. task: what we were doing, e.g. "adding context"."""
    if isinstance(error, (ModelRateLimitError, groq.RateLimitError)):
        if "per day" in str(error):  # Groq says "tokens per day (TPD)" or "requests per day (RPD)"
            return RateLimitError(
                f"LLM daily free-tier limit reached while {task}. It frees up over the next hours "
                "(Groq counts the last 24 hours). Finished work is cached, so nothing is lost.",
                daily=True,
            )
        return RateLimitError(
            f"LLM free-tier limit reached while {task}. Wait a minute and run again. "
            "Finished work is cached, so nothing is lost."
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
