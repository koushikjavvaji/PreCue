"""Type what a speaker might say and see PreCue's verdict. No microphone needed.

Run from the project folder:
  python -m scripts.check_claims samples/sample_report.txt
  python -m scripts.check_claims samples/sample_report.txt --eval samples/eval_claims.json

With --eval, every claim in the file is checked and compared with the expected
verdict, and the accuracy is printed. The file is a list of
{"claim": "...", "verdict": "correct" | "wrong" | "not_in_document"}.
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from app.errors import PreCueError
from app.fact_checker import check_claim
from app.index import MODES
from app.llm import get_live_llm, model_name
from app.prepare import prepare

TOP_K = 3


def main():
    parser = argparse.ArgumentParser(description="Check spoken claims against a document.")
    parser.add_argument("file", help="the document (.txt or .md)")
    parser.add_argument("--mode", choices=MODES, default="plain")
    parser.add_argument("--eval", help="a claims file (.json) to measure accuracy")
    args = parser.parse_args()

    load_dotenv()
    print("Preparing the document...")
    retriever, facts = prepare(args.file, args.mode)
    llm = get_live_llm()
    print(f"Ready. Live LLM: {model_name(llm)}")

    if args.eval:
        evaluate(args.eval, retriever, facts, llm)
        return

    while True:
        claim = input("\nSay something (empty line to quit): ").strip()
        if not claim:
            break
        start = time.time()
        results = retriever.search(claim, top_k=TOP_K)
        verdict = check_claim(claim, results, facts, llm)
        print(f"({(time.time() - start) * 1000:.0f} ms)")
        show(verdict)


def show(verdict):
    print(f"  verdict      : {verdict['verdict']}")
    print(f"  said         : {verdict['said'] or '-'}")
    print(f"  document says: {verdict['document_says'] or '-'}  [{verdict['section'] or '-'}]")
    if verdict["correction"]:
        print(f"  correction   : {verdict['correction']}")


def evaluate(path, retriever, facts, llm):
    try:
        claims = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreCueError(f"Could not read the claims file '{path}': {error}") from error

    right = 0
    failed = 0  # the check itself failed (LLM error); the live app would skip these
    false_alerts = 0  # said "wrong" when the speaker was not wrong: the worst mistake
    times = []
    for item in claims:
        start = time.time()
        results = retriever.search(item["claim"], top_k=TOP_K)
        try:
            verdict = check_claim(item["claim"], results, facts, llm)
        except PreCueError as error:
            failed += 1
            print(f"FAIL expected {item['verdict']:<15} {error} | {item['claim']}")
            continue
        times.append(time.time() - start)

        ok = verdict["verdict"] == item["verdict"]
        right += ok
        false_alerts += verdict["verdict"] == "wrong" and item["verdict"] != "wrong"
        mark = "OK  " if ok else "MISS"
        print(f"{mark} expected {item['verdict']:<15} got {verdict['verdict']:<15} | {item['claim']}")

    average = sum(times) / len(times) * 1000 if times else 0
    print(f"\nAccuracy: {right}/{len(claims)}   False alerts: {false_alerts}   Failed checks: {failed}   "
          f"Average time: {average:.0f} ms")


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
