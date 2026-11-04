"""Stand-ins for the slow or paid parts (LLM, search, facts), so tests run offline in seconds."""

from langchain_core.documents import Document

from app.fact_checker import Verdict


def chunk(number, section, text):
    return Document(page_content=text, metadata={"chunk": number, "section": section, "original": text, "title": "T"})


# Chunks are numbered from 1, like the real chunker does.
CHUNKS = [
    chunk(1, "1. Summary", "Revenue grew to ₹10 crore."),
    chunk(2, "2. Revenue", "Total revenue for Q2 was ₹10 crore."),
    chunk(3, "3. Customers", "Merchant count grew to 52,000."),
    chunk(4, "4. Team", "Headcount grew from 140 to 165."),
]


class FakeRetriever:
    """Finds the chunk by one keyword: merchants -> 3, headcount -> 4, anything else -> 2."""

    def __init__(self, chunks=CHUNKS):
        self.chunks = chunks

    def search(self, question, top_k=3):
        question = question.lower()
        if "merchant" in question:
            order = [3, 1, 2]
        elif "headcount" in question:
            order = [4, 1, 2]
        else:
            order = [2, 1, 3]
        return [self.chunks[number - 1] for number in order][:top_k]


class FakeFacts:
    def for_chunks(self, chunks):
        return [{"metric": "Revenue", "value": "₹10 crore", "period": "Q2", "section": "2. Revenue", "chunk": 2}]

    def closest(self, claim, top_k=5):
        return []


class FakeLLM:
    """Always gives the same verdict (or raises an error). Counts how often it was asked."""

    def __init__(self, verdict=None, error=None):
        self.verdict = verdict
        self.error = error
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return Verdict(**self.verdict)


WRONG = {"verdict": "wrong", "said": "100 crore", "document_says": "₹10 crore",
         "section": "2. Revenue", "correction": "Revenue was ₹10 crore."}
CORRECT = {"verdict": "correct", "said": "10 crore", "document_says": "₹10 crore",
           "section": "2. Revenue", "correction": ""}
