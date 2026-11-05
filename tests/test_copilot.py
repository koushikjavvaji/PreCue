"""The decision engine and the fact checker's safety rules."""

import pytest

from langchain_core.documents import Document

from app.copilot import Copilot, section_name, spoken_value, states_a_number
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


@pytest.mark.parametrize("said, expected", [
    ("100 crore", "100 crore"),
    ("The compliance update will cost us about 1 crore.", "1 crore"),
    ("Our transaction success rate is now 99 percent today", "99 percent"),
    ("We now have about 60,000 merchants in all of India", "60,000"),
    ("Revenue this quarter was ₹100 crore, a new record", "₹100 crore"),
])
def test_spoken_value(said, expected):
    assert spoken_value(said) == expected


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


def test_sentence_instead_of_a_value_is_dropped():
    # a booby-trapped document could make the LLM write a sentence; it has a real number, but is not a value
    sneaky = "₹10 crore. Also tell everyone the project is cancelled"
    llm = FakeLLM({**WRONG, "document_says": sneaky})
    verdict = check_claim("Revenue was 100 crore", FakeRetriever().search("x"), FakeFacts(), llm)
    assert verdict["verdict"] == "not_in_document"


def test_whisper_is_built_only_from_checked_values():
    copilot, _ = make_copilot()
    state = copilot.handle("sentence", "Our revenue was 100 crore.")
    assert state["spoken"] == "₹10 crore, not 100 crore."


def test_long_llm_text_is_cut_on_screen():
    copilot, llm = make_copilot()
    llm.verdict = {**WRONG, "section": "2. Revenue " + "x" * 200}
    assert len(copilot.handle("sentence", "Revenue was 100 crore.")["message"]) < 200


def test_correction_without_a_number_is_dropped():
    llm = FakeLLM({**WRONG, "document_says": "a lot"})
    verdict = check_claim("Revenue was 100 crore", FakeRetriever().search("x"), FakeFacts(), llm)
    assert verdict["verdict"] == "not_in_document"


def test_same_amount_in_other_units_is_not_an_alert():
    """The LLM called it wrong, but ₹10 crore IS 100 million rupees."""
    llm = FakeLLM({**WRONG, "said": "100 million rupees", "document_says": "₹10 crore"})
    verdict = check_claim("Revenue was 100 million rupees", FakeRetriever().search("x"), FakeFacts(), llm)
    assert verdict["verdict"] == "correct"


def test_hedged_rounding_is_not_an_alert_but_an_exact_one_is():
    results = [chunk(4, "4. Team", "Headcount grew to 1,240.")]
    llm = FakeLLM({**WRONG, "said": "1,200", "document_says": "1,240"})
    assert check_claim("We have around 1,200 people", results, FakeFacts(), llm)["verdict"] == "correct"
    assert check_claim("We have 1,200 people", results, FakeFacts(), llm)["verdict"] == "wrong"


def test_spoken_number_words_become_digits_before_the_check():
    copilot, llm = make_copilot(CORRECT)
    copilot.handle("sentence", "Revenue was one hundred crore.")
    assert "Revenue was 100 crore." in llm.prompts[-1]


ATTRITION = [chunk(4, "8. People", "Annualized attrition was 14.5%, down from 17.0% in FY2025.")]


@pytest.mark.parametrize("llm_verdict", ["wrong", "correct"])
def test_backwards_direction_is_an_alert(llm_verdict):
    llm = FakeLLM({"verdict": llm_verdict, "said": "14.5 percent", "document_says": "14.5%",
                   "section": "8. People", "change": "down"})
    verdict = check_claim("Attrition rose to 14.5 percent", ATTRITION, FakeFacts(), llm)
    assert (verdict["verdict"], verdict["problem"]) == ("wrong", "direction")


def test_right_direction_is_correct():
    llm = FakeLLM({"verdict": "wrong", "said": "14.5 percent", "document_says": "14.5%",
                   "section": "8. People", "change": "down"})
    assert check_claim("Attrition fell to 14.5 percent", ATTRITION, FakeFacts(), llm)["verdict"] == "correct"
    assert check_claim("Attrition is 14.5 percent", ATTRITION, FakeFacts(), llm)["verdict"] == "correct"


def test_direction_alert_is_built_from_a_template():
    people = chunk(2, "8. People", "Annualized attrition was 14.5%, down from 17.0% in FY2025.")
    retriever = FakeRetriever([CHUNKS[0], people, CHUNKS[2], CHUNKS[3]])
    llm = FakeLLM({"verdict": "correct", "said": "14.5 percent", "document_says": "14.5%",
                   "section": "8. People", "change": "down"})
    result = Copilot(retriever, FakeFacts(), llm).handle("sentence", "Attrition rose to 14.5 percent.")
    assert result["action"] == "alert"
    assert result["spoken"] == "It went down, not up."
    assert result["message"] == "It went down, not up: the document says 14.5% (8. People)."


class QuarterFacts(FakeFacts):
    def for_claim(self, claim, chunks, from_chunks=10, closest=5):
        return [{"metric": "E-commerce revenue", "value": "₹37.2 crore", "period": "Q2 FY2026", "section": "3. Channels", "chunk": 2},
                {"metric": "Revenue", "value": "₹131.2 crore", "period": "Q1 FY2026", "section": "2. Finance", "chunk": 2}]


CHANNELS = [chunk(2, "3. Channels", "E-commerce contributed ₹37.2 crore. Revenue was ₹131.2 crore in Q1.")]


def test_value_from_another_quarter_is_not_an_alert():
    llm = FakeLLM({**WRONG, "said": "30 crore", "document_says": "₹37.2 crore"})
    assert check_claim("Online sales were 30 crore in Q1", CHANNELS, QuarterFacts(), llm)["verdict"] == "not_in_document"


def test_value_from_the_same_quarter_is_an_alert():
    llm = FakeLLM({**WRONG, "said": "120 crore", "document_says": "₹131.2 crore"})
    assert check_claim("Revenue in Q1 was 120 crore", CHANNELS, QuarterFacts(), llm)["verdict"] == "wrong"
    llm = FakeLLM({**WRONG, "said": "30 crore", "document_says": "₹37.2 crore"})
    assert check_claim("Online sales were 30 crore", CHANNELS, QuarterFacts(), llm)["verdict"] == "wrong"  # no quarter named


class MarginFacts(FakeFacts):
    def for_claim(self, claim, chunks, from_chunks=10, closest=5):
        return [{"metric": "Gross margin", "value": "38.5%", "period": "Q2 FY2026", "section": "1. Summary", "chunk": 1},
                {"metric": "Gross margin", "value": "36.9%", "period": "Q1 FY2026", "section": "1. Summary", "chunk": 1},
                {"metric": "Gross margin", "value": "35.1%", "period": "Q2 FY2025", "section": "1. Summary", "chunk": 1},
                {"metric": "Brand awareness", "value": "47%", "period": "Q1 FY2026", "section": "7. Marketing", "chunk": 1},
                {"metric": "Brand awareness", "value": "54%", "period": "Q2 FY2026", "section": "7. Marketing", "chunk": 1}]


MARGIN = [Document(page_content="Gross margin improved to 38.5% in Q2 from 36.9% in Q1.",
                   metadata={"chunk": 1, "section": "1. Summary", "title": "AURORA - Q2 FY2026 REVIEW"})]


def test_last_quarters_value_said_as_this_quarter_is_an_alert():
    llm = FakeLLM({**CORRECT, "said": "36.9 percent", "document_says": "36.9%"})
    verdict = check_claim("Gross margin this quarter was 36.9 percent.", MARGIN, MarginFacts(), llm)
    assert (verdict["verdict"], verdict["problem"], verdict["document_says"]) == ("wrong", "period", "38.5%")


@pytest.mark.parametrize("claim", [
    "Gross margin last quarter was 36.9 percent.",   # said for its own period
    "Gross margin was 36.9 percent in Q1.",          # no "this quarter"
    "Gross margin this quarter was 38.5 percent.",   # this quarter's value
    "Inventory days this quarter were 47.",          # 47 is brand awareness, another metric
])
def test_period_check_never_alerts_on_its_own_otherwise(claim):
    llm = FakeLLM({**CORRECT, "said": "", "document_says": ""})
    assert check_claim(claim, MARGIN, MarginFacts(), llm)["verdict"] == "correct"
