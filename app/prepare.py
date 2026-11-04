"""Everything we do with a document BEFORE the meeting, in one call.

  load the embedding model -> build (or load) the index -> build (or load) the facts table
  -> warn if the document seems to hide instructions to an AI (injection.py)

All slow steps are cached, so after the first run this takes a few seconds.
"""

from app.chunker import load_text
from app.embedder import Embedder
from app.facts import FactsTable, extract_facts
from app.index import build_index
from app.injection import suspicious_sections
from app.llm import get_gemini, model_name
from app.retriever import Retriever


def prepare(path, mode="plain"):
    """Return (retriever, facts) for the document at path. facts is a FactsTable."""
    embedder = Embedder()
    chunks, vectors = build_index(path, mode, embedder)
    retriever = Retriever(chunks, vectors, embedder)
    for section in suspicious_sections(chunks):
        print(f"Warning: '{section}' has text that looks like instructions to an AI. "
              "PreCue will not follow it, but check that the document was not tampered with.")

    gemini = get_gemini()
    rows = extract_facts(chunks, load_text(path), gemini, model_name=model_name(gemini))
    return retriever, FactsTable(rows, embedder)
