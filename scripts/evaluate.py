"""Measure how well search finds the right section, for every mode and method.

Run from the project folder:
  python -m scripts.evaluate samples/sample_report.txt samples/eval_questions.json
  python -m scripts.evaluate samples/sample_report.txt samples/eval_questions.json --modes plain late

The questions file is a list of {"question": "...", "sections": ["2. Revenue", ...]}.
A result counts as correct if its top-level section is one of "sections".

Scores (higher is better, 1.0 is perfect):
  hit@1  the first result is correct
  hit@3  a correct result is in the top 3
  MRR    1 / position of the first correct result, averaged (1st = 1, 2nd = 0.5, ...)
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from app.embedder import Embedder
from app.errors import PreCueError
from app.index import MODES, build_index
from app.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Evaluate PreCue search.")
    parser.add_argument("file", help="the document (.txt, .md, .pdf, .docx or .pptx)")
    parser.add_argument("questions", help="the questions file (.json)")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    load_dotenv()
    embedder = Embedder()

    print(f"{len(questions)} questions\n")
    print(f"{'mode':<11} {'method':<8} {'hit@1':>6} {'hit@3':>6} {'MRR':>6} {'ms/q':>6}")
    misses = []
    for mode in args.modes:
        chunks, vectors = build_index(args.file, mode, embedder)
        retriever = Retriever(chunks, vectors, embedder)
        methods = {
            "vector": retriever.vector_ranking,
            "keyword": retriever.keyword_ranking,
            "hybrid": retriever.hybrid_ranking,
        }
        for method, rank in methods.items():
            hit1 = hit3 = reciprocal = 0
            start = time.time()
            for item in questions:
                position = first_correct(rank(item["question"]), chunks, item["sections"])
                hit1 += position == 1
                hit3 += position is not None and position <= 3
                reciprocal += 1 / position if position else 0
                if method == "hybrid" and position != 1:
                    misses.append((mode, item["question"], position))
            ms = (time.time() - start) * 1000 / len(questions)
            n = len(questions)
            print(f"{mode:<11} {method:<8} {hit1 / n:>6.2f} {hit3 / n:>6.2f} {reciprocal / n:>6.2f} {ms:>6.0f}")

    print("\nHybrid misses (question -> position of the first correct result):")
    for mode, question, position in misses:
        print(f"  [{mode}] {question} -> {position or 'not found'}")


def load_questions(path):
    try:
        questions = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreCueError(f"Could not read the questions file '{path}': {error}") from error
    if not questions or not all("question" in q and "sections" in q for q in questions):
        raise PreCueError("Each question needs a 'question' and a 'sections' list.")
    return questions


def first_correct(ranking, chunks, sections):
    """Position (1 = first) of the first chunk from a correct section, or None."""
    for position, i in enumerate(ranking, start=1):
        if chunks[i].metadata["section"].split(" > ")[0] in sections:
            return position
    return None


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}") from None
    except KeyboardInterrupt:
        print("\nStopped.")
