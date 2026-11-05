"""LLM setup (app/llm.py) and the facts shown to the LLM (app/facts.py). No keys or network needed."""

import groq
import pytest
import httpx
import torch
from langchain_core.documents import Document

from app.errors import RateLimitError
from app.facts import FactsTable
from app.llm import groq_with_backup, llm_error, model_name, structured


def rate_limit(message):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.RateLimitError(message, response=response, body=None)


def test_daily_limit_is_told_apart():
    daily = llm_error(rate_limit("Rate limit reached on tokens per day (TPD): Limit 200000"), "checking")
    minute = llm_error(rate_limit("Rate limit reached on tokens per minute (TPM): Limit 8000"), "checking")
    assert isinstance(daily, RateLimitError) and daily.daily
    assert isinstance(minute, RateLimitError) and not minute.daily
    assert "daily" in str(daily)


def test_backup_model_is_named(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_BACKUP_MODEL", raising=False)
    llm = groq_with_backup()
    assert model_name(llm) == "openai/gpt-oss-20b (backup: openai/gpt-oss-120b)"


class Model:
    """A stand-in chat model: gives an answer or raises an error."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.asked = answer, error, 0

    def invoke(self, prompt):
        self.asked += 1
        if self.error:
            raise self.error
        return self.answer


def shaped_pair(monkeypatch, main, backup):
    """structured() on a main + backup pair, with the stand-in models in place of Groq."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # just the two Groq models here
    pair = groq_with_backup()
    models = {id(pair.runnable): main, id(pair.fallbacks[0]): backup}
    monkeypatch.setattr("app.llm.ChatGroq.with_structured_output", lambda self, *a, **k: models[id(self)])
    return structured(pair, dict)


def test_a_rate_limit_goes_to_the_backup(monkeypatch):
    main, backup = Model(error=rate_limit("tokens per day")), Model(answer="backup answer")
    assert shaped_pair(monkeypatch, main, backup).invoke("hi") == "backup answer"
    assert main.asked == 1 and backup.asked == 1


def test_the_backup_is_not_asked_when_the_main_model_answers(monkeypatch):
    main, backup = Model(answer="main answer"), Model(answer="backup answer")
    assert shaped_pair(monkeypatch, main, backup).invoke("hi") == "main answer"
    assert backup.asked == 0


def test_when_both_are_limited_the_backups_error_is_raised(monkeypatch):
    main = Model(error=rate_limit("tokens per day (TPD)"))
    backup = Model(error=rate_limit("tokens per minute (TPM): try again in 4s"))
    with pytest.raises(groq.RateLimitError, match="per minute"):
        shaped_pair(monkeypatch, main, backup).invoke("hi")


def test_no_backup_when_it_is_the_same_model_or_turned_off(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    for backup in ("openai/gpt-oss-20b", ""):
        monkeypatch.setenv("GROQ_BACKUP_MODEL", backup)
        llm = groq_with_backup()
        assert model_name(llm) == "openai/gpt-oss-20b" and llm.max_retries == 2


def test_the_main_model_does_not_retry_when_there_is_a_backup(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_BACKUP_MODEL", raising=False)
    pair = groq_with_backup()
    assert pair.runnable.max_retries == 0 and pair.fallbacks[0].max_retries == 2


def test_gemini_answers_when_both_groq_models_cannot(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    chain = groq_with_backup()
    assert model_name(chain) == "openai/gpt-oss-20b (backup: openai/gpt-oss-120b, gemini-3.5-flash-lite)"
    main, backup, gemini = Model(error=rate_limit("tokens per day")), Model(error=rate_limit("tokens per day")), Model(answer="gemini")
    models = {id(chain.runnable): main, id(chain.fallbacks[0]): backup}
    monkeypatch.setattr("app.llm.ChatGroq.with_structured_output", lambda self, *a, **k: models[id(self)])
    monkeypatch.setattr("app.llm.ChatGoogleGenerativeAI.with_structured_output", lambda self, *a, **k: gemini)
    assert structured(chain, dict).invoke("hi") == "gemini"
    assert (main.asked, backup.asked, gemini.asked) == (1, 1, 1)


def test_a_rejected_groq_key_still_gets_an_answer_from_gemini(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "bad-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    rejected = groq.AuthenticationError("Invalid API Key", response=response, body=None)
    chain = groq_with_backup()
    models = {id(chain.runnable): Model(error=rejected), id(chain.fallbacks[0]): Model(error=rejected)}
    monkeypatch.setattr("app.llm.ChatGroq.with_structured_output", lambda self, *a, **k: models[id(self)])
    monkeypatch.setattr("app.llm.ChatGoogleGenerativeAI.with_structured_output", lambda self, *a, **k: Model(answer="ok"))
    assert structured(chain, dict).invoke("hi") == "ok"


class WordEmbedder:
    """Embeds a text by which of a few words it has: enough to rank facts by meaning in a test."""

    model_name = "test-words"
    WORDS = ["revenue", "margin", "headcount", "cash", "outlets"]

    def embed_chunks(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = torch.tensor([float(word in text.lower()) for word in self.WORDS]) + 0.01
        return (vector / vector.norm()).tolist()


def test_for_claim_caps_chunk_facts_and_adds_closest(monkeypatch, tmp_path):
    monkeypatch.setattr("app.cache.CACHE_FOLDER", tmp_path)
    rows = [{"metric": f"Margin {i}", "value": f"{i}%", "period": "Q2", "section": "Table", "chunk": 1} for i in range(30)]
    rows.append({"metric": "Revenue", "value": "₹148.6 crore", "period": "Q2", "section": "Summary", "chunk": 5})
    table = FactsTable(rows, WordEmbedder())
    found = [Document("table", metadata={"chunk": 1})]

    picked = table.for_claim("Revenue was 148.6 crore", found, from_chunks=10, closest=5)
    from_table = [fact for fact in picked if fact["chunk"] == 1]
    assert len(from_table) == 10  # 30 facts in the chunk, only 10 are sent
    assert picked[10]["metric"] == "Revenue"  # the closest fact, from another chunk, is added
    assert len(picked) == 11  # the other closest facts were already in the list


def test_groups_respect_count_and_size(monkeypatch):
    import app.facts as facts

    monkeypatch.setattr(facts, "CHUNKS_PER_CALL", 3)
    monkeypatch.setattr(facts, "MAX_CHARACTERS_PER_CALL", 100)
    texts = ["a" * 10] * 7 + ["b" * 150, "c" * 60, "d" * 60]
    assert [len(group) for group in facts.groups(texts)] == [3, 3, 1, 1, 1, 1]


def test_facts_from_one_call_go_to_their_own_chunk(monkeypatch, tmp_path):
    from app.facts import Fact, FactList, extract_facts

    monkeypatch.setattr("app.cache.CACHE_FOLDER", tmp_path)

    class Reader:
        prompts = []

        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return FactList(facts=[
                Fact(part=1, metric="Revenue", value="₹10 crore", period="Q2"),
                Fact(part=2, metric="Merchants", value="52,000", period="Q2"),
                Fact(part=1, metric="Merchants", value="52,000", period="Q2"),  # wrong part: not in chunk 1
                Fact(part=2, metric="Made up", value="77", period="Q2"),  # not in the text at all
            ])

    chunks = [Document("Revenue was ₹10 crore.", metadata={"chunk": 1, "section": "2. Revenue", "title": "T"}),
              Document("Merchants: 52,000.", metadata={"chunk": 2, "section": "3. Customers", "title": "T"})]
    reader = Reader()
    table = extract_facts(chunks, "document", reader, model_name="test")
    assert len(reader.prompts) == 1  # both chunks in one call
    assert [(fact["metric"], fact["chunk"]) for fact in table] == [("Revenue", 1), ("Merchants", 2)]
    assert "part" not in table[0]

    extract_facts(chunks, "document", reader, model_name="test")
    assert len(reader.prompts) == 1  # cached: no new call


def bad_json():
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.BadRequestError("Failed to validate JSON. Please adjust your prompt.", response=response, body=None)


class Flaky:
    """Fails with the given errors first, then answers."""

    def __init__(self, *errors):
        self.errors, self.asked = list(errors), 0

    def invoke(self, prompt):
        self.asked += 1
        if self.errors:
            raise self.errors.pop(0)
        return "answer"


def test_groq_json_failure_is_tried_again(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from app.llm import get_groq
    model = Flaky(bad_json())
    monkeypatch.setattr("app.llm.ChatGroq.with_structured_output", lambda self, *a, **k: model)
    assert structured(get_groq(), dict).invoke("hi") == "answer" and model.asked == 2


def test_other_bad_requests_are_not_tried_again(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from app.llm import get_groq
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    model = Flaky(groq.BadRequestError("Prompt too long", response=response, body=None))
    monkeypatch.setattr("app.llm.ChatGroq.with_structured_output", lambda self, *a, **k: model)
    with pytest.raises(groq.BadRequestError):
        structured(get_groq(), dict).invoke("hi")
    assert model.asked == 1


def test_a_rejected_key_is_not_asked_again(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "bad-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    rejected = groq.AuthenticationError("Invalid API Key", response=response, body=None)
    chain = groq_with_backup()
    main, backup, gemini = Model(error=rejected), Model(error=rejected), Model(answer="ok")
    models = {id(chain.runnable): main, id(chain.fallbacks[0]): backup}
    monkeypatch.setattr("app.llm.ChatGroq.with_structured_output", lambda self, *a, **k: models[id(self)])
    monkeypatch.setattr("app.llm.ChatGoogleGenerativeAI.with_structured_output", lambda self, *a, **k: gemini)
    shaped = structured(chain, dict)
    for _ in range(3):
        assert shaped.invoke("hi") == "ok"
    assert (main.asked, backup.asked, gemini.asked) == (1, 1, 3)  # Groq tried once, then skipped


def test_live_gemini_gives_up_quickly_but_offline_gemini_is_patient(monkeypatch):
    from app.llm import get_gemini, get_live_llm

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_LLM", "gemini")
    assert (get_live_llm().max_retries, get_live_llm().timeout) == (1, 10)  # Google refuses less than 10 s
    assert (get_gemini().max_retries, get_gemini().timeout) == (3, 60)


def test_the_first_call_does_not_wait_for_the_rate_limiter(monkeypatch):
    import time

    import app.llm

    monkeypatch.setattr(app.llm, "_rate_limiters", {})
    limiter = app.llm._rate_limiter("TEST_REQUESTS_PER_MINUTE", default=6)  # one call every 10 s
    start = time.time()
    assert limiter.acquire(blocking=True)
    assert time.time() - start < 0.5
    assert not limiter.acquire(blocking=False)  # the next one is still spaced out


def test_a_busy_gemini_is_a_short_wait_not_a_crash():
    from langchain_google_genai.chat_models import GoogleAPIError

    from app.llm import LLM_ERRORS

    busy = GoogleAPIError(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}})
    assert isinstance(busy, LLM_ERRORS)  # caught like every other LLM error, never an "Internal error"
    error = llm_error(busy, "checking a claim")
    assert isinstance(error, RateLimitError) and not error.daily and "busy" in str(error)


def test_a_gemini_deadline_is_also_a_short_wait():
    from langchain_google_genai.chat_models import GoogleAPIError

    late = GoogleAPIError(504, {"error": {"code": 504, "message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}})
    error = llm_error(late, "checking a claim")
    assert isinstance(error, RateLimitError) and not error.daily
