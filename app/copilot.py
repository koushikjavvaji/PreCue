"""The decision engine: for every event, decide to stay silent, alert or suggest.

Events come from the live speech:
  "sentence" - the speaker finished a sentence (final text from Deepgram)
  "pause"    - the speaker has been quiet for a while (maybe stuck)

The steps are a LangGraph graph:

            ┌── sentence ──> listen ──(number, not late?)── yes ──> check ──> decide ──> END
  START ────┤                   └────────────── no ─────────────────────────────────────> END
            └── pause ─────> suggest ─────────────────────────────────────────────────> END

  listen  : write spoken numbers as digits, join a split sentence, search the document,
            remember what was covered
  check   : ask the LLM whether the claim is correct (fact_checker.py)
  decide  : alert only if the claim is wrong AND we have not alerted about it before
  suggest : point to the next section the speaker has not talked about yet
            (never the same suggestion twice in a row)

Memory (recent sentences, covered chunks, alerts already given) is kept
between events, so PreCue never repeats itself.
"""

import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.errors import PreCueError
from app.fact_checker import check_claim
from app.facts import numbers_in
from app.numbers import words_to_digits
from app.retriever import has_number

# "Q2", "H1", "FY2026" name a period, not a fact. "Let's talk about Q2" is small talk.
# Deepgram sometimes writes "Q2" as "q two", so the spelled form counts too.
PERIOD = re.compile(r"\b(?:Q\s?(?:[1-4]|one|two|three|four)|H[12]|FY\s?\d{2,4})\b", re.IGNORECASE)

# Deepgram sometimes writes numbers as words ("fifty two thousand"), so those count too.
# "one" is left out: "one more thing" is small talk.
NUMBER_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
    "lakh", "lakhs", "crore", "crores", "million", "billion", "percent",
}

TOP_K = 3
RECENT_SENTENCES = 3  # how many recent sentences we remember
JOIN_SECONDS = 10  # a sentence is joined to the previous one only if that one ended this recently
FACTS_IN_SUGGESTION = 2
REMEMBERED_VERDICTS = 200  # a claim said again in the same meeting reuses its verdict: no second LLM call


class State(TypedDict, total=False):
    # the new event
    event: str  # "sentence" or "pause"
    text: str
    late: bool  # the sentence waited too long; checking it now would alert too late
    # memory, kept between events
    recent: list  # last few sentences
    covered: list  # chunk numbers the speaker has talked about
    alerted: list  # alerts already given, so we never repeat one
    last_chunk: int  # the chunk the speaker talked about most recently
    suggested: bool  # already suggested during this pause
    last_suggestion: int  # the chunk we suggested last time, so we do not repeat it back to back
    last_sentence_time: float  # when the previous sentence ended
    # worked out while handling this event
    claim: str
    results: list
    verdict: dict
    action: str  # "silent", "alert", "suggest" or "skipped"
    message: str  # shown on screen
    spoken: str  # whispered in the earphones


MEMORY = ["recent", "covered", "alerted", "last_chunk", "suggested", "last_suggestion", "last_sentence_time"]


class Copilot:
    def __init__(self, retriever, facts, llm):
        self.retriever = retriever
        self.facts = facts  # the FactsTable
        self.llm = llm
        self.memory = {
            "recent": [], "covered": [], "alerted": [],
            "last_chunk": 0, "suggested": False, "last_suggestion": 0, "last_sentence_time": 0.0,
        }
        self.verdicts = {}  # claim (normalised) -> verdict, for claims said more than once
        self.graph = self._build_graph()

    def handle(self, event, text="", late=False):
        """Run the graph for one event and return the final state (action, message, verdict...).

        late=True: remember the sentence, but do not check it (the alert would come too late).
        """
        start = {**self.memory, "event": event, "text": text, "late": late,
                 "action": "silent", "message": "", "spoken": ""}
        state = self.graph.invoke(start)
        self.memory = {key: state[key] for key in MEMORY}  # keep the memory for the next event
        return state

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("listen", self._listen)
        graph.add_node("check", self._check)
        graph.add_node("decide", self._decide)
        graph.add_node("suggest", self._suggest)

        graph.add_conditional_edges(START, lambda s: "listen" if s["event"] == "sentence" else "suggest",
                                    ["listen", "suggest"])
        graph.add_conditional_edges("listen", self._needs_check, ["check", END])
        graph.add_edge("check", "decide")
        graph.add_edge("decide", END)
        graph.add_edge("suggest", END)
        return graph.compile()

    def _needs_check(self, state):
        """After listen: check only sentences with a number that are not too late."""
        if states_a_number(state["claim"]) and not state["late"]:
            return "check"
        return END

    # ---- the nodes. Each one reads the state and returns only what it changes. ----

    def _listen(self, state):
        text = words_to_digits(state["text"])  # "ninety six" -> "96": search and the LLM do better with digits
        recent = state["recent"]

        # "100 crore." has a number but no topic. It probably finishes the sentence
        # before it ("Our revenue this quarter was..."), so we check them together,
        # but only if that sentence was just said (not minutes ago).
        just_said = time.time() - state["last_sentence_time"] < JOIN_SECONDS
        if states_a_number(text) and recent and not states_a_number(recent[-1]) and just_said:
            claim = f"{recent[-1]} {text}"
        else:
            claim = text

        results = self.retriever.search(claim, top_k=TOP_K)
        update = {
            "claim": claim,
            "results": results,
            "recent": (recent + [text])[-RECENT_SENTENCES:],
            "suggested": False,  # the speaker is talking again, so a new pause may get a suggestion
            "last_sentence_time": time.time(),
        }
        # Only a sentence that states a fact (has a number) marks its section as covered.
        # Small talk like "let me walk you through our quarter" would match a random section.
        if results and states_a_number(claim):
            top = results[0].metadata["chunk"]
            update["last_chunk"] = top
            if top not in state["covered"]:
                update["covered"] = state["covered"] + [top]
        return update

    def _check(self, state):
        # Same words, any punctuation: "10 crore." and "10 crore!" (a dot inside "148.6" stays).
        key = " ".join(word.strip(".") for word in re.findall(r"[a-z0-9.%₹]+", state["claim"].lower()))
        if key in self.verdicts:
            return {"verdict": dict(self.verdicts[key])}  # said before: the same answer, free and at once
        try:
            verdict = check_claim(state["claim"], state["results"], self.facts, self.llm)
        except PreCueError as error:  # a failed check must never stop the meeting (and is not remembered)
            return {"verdict": None, "action": "skipped", "message": str(error)}
        if len(self.verdicts) >= REMEMBERED_VERDICTS:
            self.verdicts.pop(next(iter(self.verdicts)))  # forget the oldest
        self.verdicts[key] = verdict
        return {"verdict": verdict}

    def _decide(self, state):
        verdict = state.get("verdict")
        if not verdict or verdict["verdict"] != "wrong":
            return {}  # correct, not in the document, or skipped: stay as we are

        # The key is the numbers only: the LLM may write "100 crore" once and "₹100 crore" the next time.
        key = f"{sorted(numbers_in(verdict['said']))} | {sorted(numbers_in(verdict['document_says']))}"
        if key in state["alerted"]:
            return {}  # we already told them about this mistake

        # Built only from short, checked values, never from free LLM text: a booby-trapped
        # document must not be able to make PreCue say anything else in your ear.
        said, real = short(spoken_value(verdict["said"])), short(verdict["document_says"])
        section = short(section_name(verdict["section"]))
        if verdict.get("problem") == "direction":
            went, not_went = verdict["change"], "up" if verdict["change"] == "down" else "down"
            return {
                "action": "alert",
                "message": f"It went {went}, not {not_went}: the document says {real} ({section}).",
                "spoken": f"It went {went}, not {not_went}.",
                "alerted": state["alerted"] + [key],
            }
        return {
            "action": "alert",
            "message": f"You said {said}, the document says {real} ({section}).",
            "spoken": f"{real}, not {said}.",
            "alerted": state["alerted"] + [key],
        }

    def _suggest(self, state):
        if state["suggested"] or not state["recent"]:
            return {}  # already suggested in this pause, or the speaker has not started yet

        chunk = self._next_point(state)
        if chunk is None:
            return {"suggested": True}  # everything is covered
        if chunk.metadata["chunk"] == state["last_suggestion"]:
            return {"suggested": True}  # same as last time; saying it again would only annoy

        section = chunk.metadata["section"]
        name = re.sub(r"^[\d.]+\s*", "", section)  # "3. Customers" -> "Customers"
        details = "; ".join(f"{fact['metric']}: {fact['value']}" for fact in self._key_facts(chunk))
        return {
            "action": "suggest",
            "message": f"Next: {section}" + (f" ({details})" if details else ""),
            "spoken": f"Next, {name}.",
            "suggested": True,
            "last_suggestion": chunk.metadata["chunk"],
        }

    def _key_facts(self, chunk):
        """The first few facts of a chunk, one per metric.

        A chunk often gives the same metric twice ("97.4%, compared to 87.0% in Q1");
        the first one is the main value, the second would only confuse.
        """
        facts = []
        for fact in self.facts.for_chunks([chunk]):
            if fact["metric"].lower() not in [f["metric"].lower() for f in facts]:
                facts.append(fact)
        return facts[:FACTS_IN_SUGGESTION]

    def _next_point(self, state):
        """The first section after the current one that the speaker has not covered yet."""
        chunks = [c for c in self.retriever.chunks if c.metadata["section"]]  # skip text before the first heading
        after = [c for c in chunks if c.metadata["chunk"] > state["last_chunk"]]
        before = [c for c in chunks if c.metadata["chunk"] <= state["last_chunk"]]
        for chunk in after + before:  # look ahead first, then go back to anything skipped
            if chunk.metadata["chunk"] not in state["covered"]:
                return chunk
        return None


# A number with its unit: "1 crore", "₹10 crore", "97.4%", "52,000", "3.2 percent".
VALUE = re.compile(r"[₹$]?\s?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|crores?|lakhs?|thousand|million|billion)?", re.IGNORECASE)


def spoken_value(said):
    """What the speaker said, as a value. The LLM sometimes gives the whole sentence instead:
    "The compliance update will cost us about 1 crore." -> "1 crore"."""
    if len(said.split()) <= 6:
        return said
    match = VALUE.search(said)
    return match.group().strip() if match else said


def short(text, limit=60):
    """Cut LLM text that is much longer than a value or a section name should be."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def section_name(section):
    """The LLM sometimes copies the chunk's header line: "[Report > 2. Revenue]" -> "2. Revenue"."""
    return section.strip("[] ").split(" > ")[-1]


def states_a_number(text):
    """True if the text has a number that could be a fact (not just a period name like "Q2")."""
    text = PERIOD.sub("", text)
    return has_number(text) or any(word in NUMBER_WORDS for word in re.findall(r"[a-z]+", text.lower()))
