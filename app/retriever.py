"""Hybrid search: meaning (vectors) + exact words (BM25), merged with RRF.

Why two searches?
  - Vector search understands meaning: "how much money did we make"
    finds the revenue chunk even though the words are different.
  - Keyword search (BM25) is good at exact words and numbers:
    "1,850" or "NBFC" are found reliably, where vectors can be fuzzy.
  Numbers matter a lot for fact-checking, so we use both.

When does keyword search join in? Only when the question has a number.
Our evaluation (scripts/evaluate.py) showed that on normal sentences BM25
matches filler words ("people", "because") and makes results worse, while
on number lookups ("1850", "45 lakh") it finds what vectors miss.

How we merge them (Reciprocal Rank Fusion, RRF):
  each chunk gets 1 / (60 + its rank) from each list, and we add them up.
  Being near the top of either list helps; being near the top of both helps most.
  We only use ranks, so the two very different score scales never get mixed.
"""

import re

import torch
from langchain_core.documents import Document
from langsmith import traceable
from rank_bm25 import BM25Okapi

RRF_K = 60  # the standard value from the RRF paper; bigger = flatter merge

# Very common words carry no meaning for search, so BM25 ignores them.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the",
    "this", "to", "was", "we", "were", "what", "when", "which", "who", "why", "with",
}


def words(text):
    """Split text into lowercase search words. Keeps numbers like "2.3" and "1,850" whole."""
    text = text.lower().replace(",", "")  # "1,850" -> "1850", so it matches a spoken "1850"
    return [word for word in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text) if word not in STOP_WORDS]


class Retriever:
    def __init__(self, chunks, vectors, embedder):
        self.chunks = chunks
        self.vectors = torch.tensor(vectors)  # one row per chunk
        self.embedder = embedder
        self.bm25 = BM25Okapi([words(chunk.page_content) for chunk in chunks])

    @traceable(run_type="retriever", name="hybrid_search")  # shows up in LangSmith when tracing is on
    def search(self, question, top_k=3):
        """Return the top_k best chunks, with their scores and ranks in the metadata."""
        scores, vector_ranking, keyword_ranking = self._merge(question)
        best = sorted(scores, key=scores.get, reverse=True)[:top_k]

        results = []
        for i in best:
            info = {
                **self.chunks[i].metadata,
                "score": round(scores[i], 4),
                "vector_rank": vector_ranking.index(i) + 1,
                "keyword_rank": keyword_ranking.index(i) + 1 if i in keyword_ranking else None,
            }
            results.append(Document(page_content=self.chunks[i].page_content, metadata=info))
        return results

    def hybrid_ranking(self, question):
        """All chunk positions, best first, in the same order search() uses."""
        scores, _, _ = self._merge(question)
        return sorted(scores, key=scores.get, reverse=True)

    def _merge(self, question):
        """Run both searches and merge them with RRF. Returns (scores, vector_ranking, keyword_ranking)."""
        vector_ranking = self.vector_ranking(question)
        keyword_ranking = self.keyword_ranking(question) if has_number(question) else []
        return rrf_scores([vector_ranking, keyword_ranking]), vector_ranking, keyword_ranking

    def vector_ranking(self, question):
        """Chunk positions, best first, by meaning."""
        query = torch.tensor(self.embedder.embed_query(question))
        scores = self.vectors @ query  # dot product with every chunk at once = cosine similarity
        return scores.argsort(descending=True).tolist()

    def keyword_ranking(self, question):
        """Chunk positions, best first, by shared words. Chunks with no shared word are left out."""
        scores = self.bm25.get_scores(words(question))
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranking if scores[i] > 0]


def has_number(text):
    return any(char.isdigit() for char in text)


def rrf_scores(rankings):
    """Merge several rankings: {chunk position: sum of 1 / (RRF_K + rank)}."""
    scores = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0) + 1 / (RRF_K + rank)
    return scores
