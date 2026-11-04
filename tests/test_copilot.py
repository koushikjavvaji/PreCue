"""The decision engine and the fact checker's safety rules."""

import pytest

from app.copilot import Copilot, section_name, states_a_number
from app.errors import PreCueError
from app.fact_checker import check_claim
from tests.fakes import CHUNKS, CORRECT, WRONG, FakeFacts, FakeLLM, FakeRetriever, chunk


def make_copilot(verdict=WRONG, error=None):
    llm = FakeLLM(verdict, error)
    return Copilot(FakeRetriever(), FakeFacts(), llm), llm


# ---- staying silent ----

def test_small_talk_is_silent_and_costs_no_llm_call():
    copilot, llm = make_copilot()
    assert copilot.handle("sentence", "Thanks everyone for joining.")["action"] == "silent"
    assert llm.calls == 0


def test_period_names_are_small_talk():
    copilot, llm = make_copilot()
    copilot.handle("sentence", "Let's talk about Q2 and FY2026 now.")
    assert llm.calls == 0


def test_correct_number_is_silent():
    copilot, _ = make_copilot(CORRECT)
    assert copilot.handle("sentence", "Revenue was 10 crore.")["action"] == "silent"


@pytest.mark.parametrize("text, expected", [
    ("Revenue was 10 crore.", True),
    ("We have fifty two thousand merchants.", True),  # Deepgram sometimes writes numbers as words
    ("Churn is three point two percent.", True),
    ("Let's look at Q2.", False),
    ("Thanks for joining the q two review.", False),  # how Deepgram sometimes writes "Q2"
    ("We onboarded 2.3 lakh users in q two.", True),
    ("One more thing.", False),
    ("Thanks for joining.", False),
])
def test_states_a_number(text, expected):
    assert states_a_number(text) is expected


# ---- alerts ----

def test_wrong_number_alerts_once():
    copilot, _ = make_copilot()
    first = copilot.handle("sentence", "Our revenue was 100 crore.")
    second = copilot.handle("sentence", "Again, revenue was 100 crore.")
    assert first["action"] == "alert"
    assert "100 crore" in first["message"] and first["spoken"]
    assert second["action"] == "silent"


def test_same_mistake_written_differently_alerts_once():
    copilot, llm = make_copilot()
    copilot.handle("sentence", "Our revenue was 100 crore.")
    llm.verdict = {**WRONG, "said": "₹100 crore"}  # the LLM wrote it differently this time
    assert copilot.handle("sentence", "As I said, revenue was 100 crore.")["action"] == "silent"


def test_alert_shows_only_the_section_name():
    copilot, llm = make_copilot()
    llm.verdict = {**WRONG, "section": "[NIMBUS REPORT > 2. Revenue]"}  # the LLM copied the chunk header
    assert copilot.handle("sentence", "Revenue was 100 crore.")["message"].endswith("(2. Revenue).")


@pytest.mark.parametrize("section, expected", [
    ("2. Revenue", "2. Revenue"),
    ("[NIMBUS REPORT > 2. Revenue]", "2. Revenue"),
    ("[Report > 2. Revenue > 2.1 Cities]", "2.1 Cities"),
    ("", ""),
])
def test_section_name(section, expected):
    assert section_name(section) == expected


def test_late_sentence_is_remembered_but_not_checked():
    copilot, llm = make_copilot()
    state = copilot.handle("sentence", "Revenue was 100 crore.", late=True)
    assert state["action"] == "silent" and llm.calls == 0
    assert "Revenue was 100 crore." in copilot.memory["recent"]


def test_llm_failure_is_skipped_and_meeting_goes_on():
    copilot, _ = make_copilot(error=PreCueError("rate limit"))
    state = copilot.handle("sentence", "Revenue was 100 crore.")
    assert state["action"] == "skipped" and "rate limit" in state["message"]
    assert copilot.handle("sentence", "Hello.")["action"] == "silent"


# ---- split sentences ----

def test_split_sentence_is_joined_when_just_said():
    copilot, _ = make_copilot()
    copilot.handle("sentence", "Our revenue this quarter was")
    assert copilot.handle("sentence", "100 crore.")["claim"] == "Our revenue this quarter was 100 crore."


def test_split_sentence_is_not_joined_when_old():
    copilot, _ = make_copilot()
    copilot.handle("sentence", "Our revenue this quarter was")
    copilot.memory["last_sentence_time"] -= 60
    assert copilot.handle("sentence", "100 crore.")["claim"] == "100 crore."


# ---- suggestions ----

def test_pause_before_speaking_is_silent():
    copilot, _ = make_copilot()
    assert copilot.handle("pause")["action"] == "silent"


def test_pause_suggests_next_uncovered_section_once():
    copilot, _ = make_copilot(CORRECT)
    copilot.handle("sentence", "Revenue was 10 crore.")  # covers chunk 2
    first = copilot.handle("pause")
    assert first["action"] == "suggest" and "3. Customers" in first["message"]
    assert copilot.handle("pause")["action"] == "silent"  # same pause: do not repeat


def test_suggestion_shows_each_metric_once():
    class TwoValues(FakeFacts):
        def for_chunks(self, chunks):
            row = {"period": "Q2", "section": "4. Operations", "chunk": 3}
            return [{**row, "metric": "Success rate", "value": "97.4%"},
                    {**row, "metric": "success rate", "value": "87.0%"},
                    {**row, "metric": "Uptime", "value": "99.95%"}]

    copilot = Copilot(FakeRetriever(), TwoValues(), FakeLLM(CORRECT))
    copilot.handle("sentence", "Revenue was 10 crore.")
    assert copilot.handle("pause")["message"].endswith("(Success rate: 97.4%; Uptime: 99.95%)")


def test_suggestions_wrap_back_to_skipped_sections():
    copilot, _ = make_copilot(CORRECT)
    copilot.handle("sentence", "Merchants are 52,000.")  # covers 3
    copilot.handle("sentence", "Headcount is 165.")  # covers 4
    assert "1. Summary" in copilot.handle("pause")["message"]


def test_no_suggestion_when_everything_is_covered():
    copilot, _ = make_copilot(CORRECT)
    copilot.memory["covered"] = [c.metadata["chunk"] for c in CHUNKS]
    copilot.memory["recent"] = ["something"]
    assert copilot.handle("pause")["action"] == "silent"


def test_memory_stays_small():
    copilot, _ = make_copilot(CORRECT)
    for i in range(50):
        copilot.handle("sentence", f"Hello {i}.")
    assert len(copilot.memory["recent"]) == 3


# ---- the fact checker never trusts a correction it cannot find in the document ----

def test_made_up_correction_is_dropped():
    llm = FakeLLM({**WRONG, "document_says": "₹77 crore"})
    verdict = check_claim("Revenue was 100 crore", FakeRetriever().search("x"), FakeFacts(), llm)
    assert verdict["verdict"] == "not_in_document"


def test_correction_hidden_inside_a_real_number_is_dropped():
    # "₹1 crore" is not in the document, even though "1" is part of "10"
    llm = FakeLLM({**WRONG, "document_says": "₹1 crore"})
    results = [chunk(2, "2. Revenue", "Total revenue was ₹10 crore.")]
    verdict = check_claim("Revenue was 100 crore", results, FakeFacts(), llm)
    assert verdict["verdict"] == "not_in_document"


def test_correction_without_a_number_is_dropped():
    llm = FakeLLM({**WRONG, "document_says": "a lot"})
    verdict = check_claim("Revenue was 100 crore", FakeRetriever().search("x"), FakeFacts(), llm)
    assert verdict["verdict"] == "not_in_document"
