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
from app.llm import gemini_models, model_name
from app.retriever import Retriever


def prepare(path, mode="plain", embedder=None, report=None):
    """Return (retriever, facts) for the document at path. facts is a FactsTable.

    embedder: a loaded Embedder, to skip loading the model again (the app loads it at start).
    report(kind, text): called with ("step", "Reading the document") and ("warning", "...");
    without it, steps are silent and warnings are printed.
    """
    report = report or _print_warnings
    if embedder is None:
        report("step", "Loading the search model")
        embedder = Embedder()
    report("step", "Reading and indexing the document")
    chunks, vectors = build_index(path, mode, embedder)
    retriever = Retriever(chunks, vectors, embedder)
    for section in suspicious_sections(chunks):
        report("warning", f"'{section}' has text that looks like instructions to an AI. "
               "PreCue will not follow it, but check that the document was not tampered with.")

    report("step", "Building the facts table")
    gemini = gemini_models()  # main + backup model; the main one names the cache
    rows = extract_facts(chunks, load_text(path), gemini, model_name=model_name(gemini[0]),
                         on_progress=lambda done, total: report("progress", f"{done}/{total}"))
    return retriever, FactsTable(rows, embedder)


def _print_warnings(kind, text):
    if kind == "warning":
        print(f"Warning: {text}")
