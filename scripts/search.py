"""Ask questions about a document and see which chunks come back.

Run from the project folder:
  python -m scripts.search samples/sample_report.txt
  python -m scripts.search samples/sample_report.txt --mode late
  python -m scripts.search samples/sample_report.txt --facts   (also show the facts of each result)

Modes (how chunks are embedded):
  plain       each chunk embedded on its own (the baseline)
  contextual  Gemini writes a context line for each chunk, then normal embedding
  late        late chunking: the whole document is embedded at once, then cut

Search is always hybrid: vector search + BM25 keyword search, merged with RRF.
"""

import argparse
import time

from dotenv import load_dotenv

from app.chunker import load_text
from app.embedder import Embedder
from app.errors import PreCueError
from app.facts import extract_facts, facts_for
from app.index import MODES, build_index
from app.llm import get_gemini, model_name
from app.retriever import Retriever

TOP_K = 3


def main():
    parser = argparse.ArgumentParser(description="Search a document with PreCue.")
    parser.add_argument("file", help="a .txt or .md file")
    parser.add_argument("--mode", choices=MODES, default="plain")
    parser.add_argument("--facts", action="store_true", help="show the facts table rows of each result")
    args = parser.parse_args()

    load_dotenv()
    print("Loading the embedding model...")
    embedder = Embedder()

    start = time.time()
    chunks, vectors = build_index(args.file, args.mode, embedder)
    retriever = Retriever(chunks, vectors, embedder)
    print(f"Indexed {len(chunks)} chunks in '{args.mode}' mode in {time.time() - start:.1f}s")

    table = []
    if args.facts:
        llm = get_gemini()
        table = extract_facts(chunks, load_text(args.file), llm, model_name=model_name(llm))
        print(f"Facts table: {len(table)} facts")

    while True:
        question = input("\nAsk a question (empty line to quit): ").strip()
        if not question:
            break
        start = time.time()
        results = retriever.search(question, top_k=TOP_K)
        print(f"({(time.time() - start) * 1000:.0f} ms)")
        for result in results:
            show(result, table)


def show(result, table):
    """Print one search result: scores, section, a preview, and (optionally) its facts."""
    info = result.metadata
    keyword_rank = info["keyword_rank"] or "-"
    preview = info.get("original", result.page_content).replace("\n", " ")
    print(f"  {info['score']:.4f} (vector #{info['vector_rank']}, keyword #{keyword_rank}) "
          f"| {info['section'] or '-'} | {preview[:90]}...")
    for fact in facts_for(table, [result]):
        print(f"      - {fact['metric']}: {fact['value']} ({fact['period'] or 'period not stated'})")


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
