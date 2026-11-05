"""Builds the search index for a document: its chunks and their vectors.

This is the "before the meeting" step:
  load the file -> chunk it -> (contextual mode) add Gemini context -> embed

Embedding a long document takes time, so the result is saved in data/cache.
The next run with the same document, mode and models loads it instantly.
"""

from langchain_core.documents import Document

from app.cache import cache_file, read_json, write_json
from app.chunker import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKER_VERSION, load_text, make_chunks
from app.contextualizer import add_context
from app.errors import PreCueError
from app.llm import gemini_models, model_name

MODES = ["plain", "contextual", "late"]


def build_index(path, mode, embedder):
    """Return (chunks, vectors) for the document at path, built in the chosen mode."""
    if mode not in MODES:
        raise PreCueError(f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}.")
    text = load_text(path)

    llm = None
    llm_name = ""
    if mode == "contextual":
        llm = gemini_models()
        llm_name = model_name(llm[0])  # the main model names the cache, so a backup answer does not start a new one

    # Everything the index depends on goes into the cache file name.
    settings = [mode, embedder.model_name, llm_name, str(CHUNK_SIZE), str(CHUNK_OVERLAP), CHUNKER_VERSION]
    path_on_disk = cache_file("index", *settings, text)
    saved = read_json(path_on_disk)
    if saved:
        chunks = [Document(page_content=c["text"], metadata=c["metadata"]) for c in saved["chunks"]]
        return chunks, saved["vectors"]

    chunks = make_chunks(text, source=str(path))

    if mode == "contextual":
        chunks = add_context(chunks, text, llm, model_name=llm_name)

    if mode == "late":
        spans = [(c.metadata["start"], c.metadata["end"]) for c in chunks]
        vectors = embedder.embed_chunks_late(text, spans)
    else:
        vectors = embedder.embed_chunks([c.page_content for c in chunks])

    saved = {
        "chunks": [{"text": c.page_content, "metadata": c.metadata} for c in chunks],
        "vectors": vectors,
    }
    write_json(path_on_disk, saved)
    return chunks, vectors
