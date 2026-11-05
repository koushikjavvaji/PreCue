"""Creates the LLMs PreCue uses, in one place.

  Gemini (free tier): offline work before the meeting (context, facts table).
  Groq (free tier):   live work during the meeting, because it answers fast.

Every LLM is a LangChain chat model, so the rest of the code does not care
which one it is. Set LIVE_LLM=gemini in .env to use Gemini for live work too.
"""

import os

import groq
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError, OutputParserException
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
LLM_ERRORS = (ChatGoogleGenerativeAIError, ModelAPIError, groq.GroqError, OutputParserException, ValidationError)

_rate_limiters = {}  # one limiter per provider, shared by every call in this run


def get_gemini(live=False):
    """Return a Gemini chat model. Reads GEMINI_API_KEY (and optional settings) from .env.

    live=True: for checks during a meeting. Google sometimes answers "busy" (503); before the
    meeting we can wait and retry for a while, but in a meeting a late alert is useless.
    """
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        google_api_key=_api_key("GEMINI_API_KEY", "Gemini"),
        rate_limiter=_rate_limiter("GEMINI_REQUESTS_PER_MINUTE", default=15),
        max_retries=1 if live else 3,  # retry when Google is busy (503)
        timeout=10 if live else 60,  # 10 s is the shortest Google accepts
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
        return get_gemini(live=True)
    raise PreCueError("LIVE_LLM in .env must be 'groq' or 'gemini'.")


def groq_with_backup():
    """The Groq model, with backups ready for when it fails (usually its free-tier limit):

      Groq main model -> Groq backup model -> Gemini (if you have a Gemini key)

    Each has its own free quota, so together they give the most checks a day. Gemini is last
    because it is slower (a few seconds), but a slow check is better than none.
    """
    main_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    backup_model = os.getenv("GROQ_BACKUP_MODEL", DEFAULT_GROQ_BACKUP_MODEL)
    backups = []
    if backup_model and backup_model != main_model:
        backups.append(get_groq(backup_model))
    if _has_key("GEMINI_API_KEY"):
        backups.append(get_gemini(live=True))
    if not backups:
        return get_groq(main_model)
    # No retries for the main model: waiting for it would make the alert late, the backup answers at once.
    main = get_groq(main_model, max_retries=0)
    return main.with_fallbacks(backups, exceptions_to_handle=(groq.GroqError,))


def structured(llm, schema):
    """The LLM, set up to answer in the shape of `schema` (a pydantic class).

    Groq: "strict" JSON schema mode. Groq forces the model to fill in exactly this shape, so the answer always fits.
    The default (a tool call) failed now and then: the model wrote plain JSON instead of calling the tool.
    A model with a backup: both are set up the same way, and the backup answers when the main one fails.
    """
    if isinstance(llm, RunnableWithFallbacks):
        models = [structured(model, schema) for model in [llm.runnable, *llm.fallbacks]]
        rejected = set()  # models whose key was refused: asking again would only make every check slower

        def ask(prompt):
            for number, model in enumerate(models[:-1]):
                if number in rejected:
                    continue
                try:
                    return model.invoke(prompt)
                except (groq.AuthenticationError, groq.PermissionDeniedError):
                    rejected.add(number)  # a wrong key does not fix itself during the meeting
                except groq.GroqError:
                    continue  # its limit is reached: the next one answers
            # If the last one fails too, ITS error is raised: it says how long to really wait.
            # (LangChain's own fallbacks raise the first model's error, e.g. "daily limit",
            # when the backup only needs a few seconds.)
            return models[-1].invoke(prompt)

        return RunnableLambda(ask)
    if isinstance(llm, ChatGroq):
        shaped = llm.with_structured_output(schema, method="json_schema", strict=True)
        return RunnableLambda(lambda prompt: _ask_twice_if_json_fails(shaped, prompt))
    return llm.with_structured_output(schema)


def _ask_twice_if_json_fails(shaped, prompt):
    """Now and then (about 2 in 100) Groq answers "Failed to validate JSON": the model wrote an
    answer that does not fit the shape. A second try almost always works."""
    try:
        return shaped.invoke(prompt)
    except groq.BadRequestError as error:
        if "validate JSON" not in str(error):
            raise
        return shaped.invoke(prompt)


def model_name(llm):
    """The model name of a LangChain chat model (Gemini and Groq store it differently)."""
    if isinstance(llm, RunnableWithFallbacks):
        backups = ", ".join(model_name(backup) for backup in llm.fallbacks)
        return f"{model_name(llm.runnable)} (backup: {backups})"
    return getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")


def llm_error(error, task):
    """Turn an LLM error into a short PreCueError. task: what we were doing, e.g. "adding context"."""
    if isinstance(error, (ModelRateLimitError, groq.RateLimitError)):
        if "per day" in str(error):  # Groq: "tokens per day (TPD)"; Gemini: "requests per day"
            return RateLimitError(
                f"LLM daily free-tier limit reached while {task}. It frees up over the next hours "
                "(Groq counts the last 24 hours; Gemini resets at midnight Pacific time). Finished work is cached, so nothing is lost.",
                daily=True,
            )
        return RateLimitError(
            f"LLM free-tier limit reached while {task}. Wait a minute and run again. "
            "Finished work is cached, so nothing is lost."
        )
    busy = ("503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED")  # overloaded, or too slow to answer in time
    if any(word in str(error) for word in busy) or isinstance(error, groq.InternalServerError):
        # The provider is overloaded for a moment ("high demand"): like a short rate limit, worth a retry.
        return RateLimitError(f"The LLM is busy right now while {task}. Try again in a moment.")
    return PreCueError(f"LLM failed while {task}: {str(error)[:300]}")


def _has_key(name):
    value = os.getenv(name)
    return bool(value) and value != "your_key_here"


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
        limiter = InMemoryRateLimiter(requests_per_second=per_minute / 60, max_bucket_size=1)
        # It starts empty, so the very first call would wait a whole interval (2 s for Groq, 4 s for
        # Gemini): in a meeting, that is the first alert arriving late. Start with one call ready.
        limiter.available_tokens = 1.0
        _rate_limiters[setting] = limiter
    return _rate_limiters[setting]
