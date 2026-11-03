"""Turns text into vectors (embeddings), locally and for free.

Model: nomic-ai/modernbert-embed-base
  - reads up to 8,192 tokens at once (needed for late chunking)
  - Apache 2.0 licence, runs with plain `transformers` (no downloaded custom code)
  - the model expects a prefix: "search_document: " for chunks,
    "search_query: " for questions

Two ways to embed chunks:
  1. Normal: each chunk is read on its own.
  2. Late chunking: the WHOLE document is read once, so every word's
     vector "knows" the rest of the document. Then each chunk's vector is
     the average of the word vectors inside that chunk.
     A document longer than the model's limit is read in windows: as many
     neighbouring chunks as fit in one model run.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from app.errors import PreCueError

MODEL_NAME = "nomic-ai/modernbert-embed-base"
MAX_TOKENS = 8192
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder:
    def __init__(self, model_name=MODEL_NAME):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except OSError as error:
            raise PreCueError(f"Could not load the embedding model '{model_name}': {error}") from error
        self.model_name = model_name
        self.model.eval()  # we only use the model, we never train it
        self.embed_query("warm up")  # the first run is slow (~1 s); do it now, not on the first live question

    def embed_query(self, question):
        """One vector for a search question."""
        vectors, _ = self._word_vectors(QUERY_PREFIX + question)
        return self._average(vectors)

    def embed_chunks(self, chunk_texts):
        """Normal embedding: one vector per chunk, each chunk read on its own."""
        result = []
        for text in chunk_texts:
            vectors, _ = self._word_vectors(DOCUMENT_PREFIX + text)
            result.append(self._average(vectors))
        return result

    def embed_chunks_late(self, document, spans):
        """Late chunking: read the document once, then cut it into chunk vectors.

        spans: (start, end) character positions of each chunk in the document.
        """
        result = []
        for group in self._group_spans(document, spans):
            window_start, window_end = group[0][0], group[-1][1]
            window = document[window_start:window_end]
            local_spans = [(start - window_start, end - window_start) for start, end in group]
            result.extend(self._late_window(window, local_spans))
        return result

    def _group_spans(self, document, spans):
        """Put neighbouring chunks together, as many as fit in one model run.

        A short document becomes one group (the whole document at once).
        """
        groups = []
        for start, end in spans:
            if groups and self._count_tokens(document[groups[-1][0][0] : end]) <= MAX_TOKENS:
                groups[-1].append((start, end))  # still fits: add to the current window
            else:
                groups.append([(start, end)])  # does not fit: start a new window
        return groups

    def _late_window(self, text, spans):
        """Run the model once on text, then average the token vectors inside each span."""
        vectors, offsets = self._word_vectors(DOCUMENT_PREFIX + text)
        shift = len(DOCUMENT_PREFIX)  # positions moved right because of the prefix

        result = []
        for start, end in spans:
            # keep the word pieces (tokens) that sit inside this chunk
            inside = [
                i for i, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start  # skip special tokens, they have no text
                and token_start >= start + shift and token_end <= end + shift
            ]
            if not inside:
                raise PreCueError(f"Late chunking: no tokens found for chunk at {start}-{end}.")
            result.append(self._average(vectors[inside]))
        return result

    def _count_tokens(self, text):
        # verbose=False: counting a long text is fine, so skip the "too long" warning
        return len(self.tokenizer(DOCUMENT_PREFIX + text, verbose=False)["input_ids"])

    def _word_vectors(self, text):
        """Run the model once. Return one vector per token, and each token's text position."""
        tokens = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        if tokens["input_ids"].shape[1] > MAX_TOKENS:
            raise PreCueError(
                f"Text is too long for the embedding model ({tokens['input_ids'].shape[1]} tokens, "
                f"max {MAX_TOKENS})."
            )
        offsets = tokens.pop("offset_mapping")[0].tolist()
        with torch.no_grad():  # no training, so skip the extra memory for gradients
            output = self.model(**tokens)
        return output.last_hidden_state[0], offsets

    def _average(self, vectors):
        """Mean of the token vectors, scaled to length 1 so we can compare with a dot product."""
        mean = vectors.mean(dim=0)
        return torch.nn.functional.normalize(mean, dim=0).tolist()
