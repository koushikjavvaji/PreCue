"""Type what a speaker might say and see PreCue's verdict. No microphone needed.

Run from the project folder:
  python -m scripts.check_claims samples/sample_report.txt
  python -m scripts.check_claims samples/sample_report.txt --eval samples/eval_claims.json

With --eval, every claim in the file is checked and compared with the expected
verdict, and the accuracy is printed. The file is a list of
{"claim": "...", "verdict": "correct" | "wrong" | "not_in_document", "kind": "..." (optional)}.
With "kind", the accuracy of each kind of claim is printed too.
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from app.errors import PreCueError, RateLimitError
from app.fact_checker import check_claim
from app.index import MODES
from app.llm import get_live_llm, model_name
from app.numbers import words_to_digits
from app.prepare import prepare

TOP_K = 3
RATE_LIMIT_WAIT = 30  # seconds; the free tier's per-minute token budget refills within a minute
RATE_LIMIT_TRIES = 4


def main():
    parser = argparse.ArgumentParser(description="Check spoken claims against a document.")
    parser.add_argument("file", help="the document (.txt, .md, .pdf, .docx or .pptx)")
    parser.add_argument("--mode", choices=MODES, default="plain")
    parser.add_argument("--eval", help="a claims file (.json) to measure accuracy")
    parser.add_argument("--kinds", nargs="+", help="with --eval: check only claims of these kinds (saves free-tier quota)")
    args = parser.parse_args()

    load_dotenv()
    print("Preparing the document...")
    retriever, facts = prepare(args.file, args.mode)
    llm = get_live_llm()
    print(f"Ready. Live LLM: {model_name(llm)}")

    if args.eval:
        evaluate(args.eval, retriever, facts, llm, kinds=args.kinds)
        return

    while True:
        claim = input("\nSay something (empty line to quit): ").strip()
        if not claim:
            break
        start = time.time()
        claim = words_to_digits(claim)
        results = retriever.search(claim, top_k=TOP_K)
        verdict = check_claim(claim, results, facts, llm)
        print(f"({(time.time() - start) * 1000:.0f} ms)")
        show(verdict)


def show(verdict):
    print(f"  verdict      : {verdict['verdict']}")
    print(f"  said         : {verdict['said'] or '-'}")
    print(f"  document says: {verdict['document_says'] or '-'}  [{verdict['section'] or '-'}]")


def check_waiting_for_limits(claim, retriever, facts, llm):
    """Check one claim. An evaluation sends claims back to back, faster than a meeting would,
    so it can hit the free tier's per-minute limit: then wait and try again.
    Returns (verdict, seconds the check itself took)."""
    claim = words_to_digits(claim)  # as the live copilot does
    for attempt in range(RATE_LIMIT_TRIES):
        start = time.time()
        try:
            verdict = check_claim(claim, retriever.search(claim, top_k=TOP_K), facts, llm)
            return verdict, time.time() - start
        except RateLimitError as error:
            if error.daily or attempt == RATE_LIMIT_TRIES - 1:
                raise
            print(f"     (free-tier limit, waiting {RATE_LIMIT_WAIT} s)")
            time.sleep(RATE_LIMIT_WAIT)


def evaluate(path, retriever, facts, llm, kinds=None):
    try:
        claims = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreCueError(f"Could not read the claims file '{path}': {error}") from error
    if kinds:
        claims = [item for item in claims if item.get("kind") in kinds]

    right = 0
    by_kind = {}  # kind -> [right, total]
    false_alert_claims = []
    failed = 0  # the check itself failed (LLM error); the live app would skip these
    false_alerts = 0  # said "wrong" when the speaker was not wrong: the worst mistake
    caught = 0  # wrong claims PreCue alerted about: what the speaker needs
    times = []
    for item in claims:
        try:
            verdict, seconds = check_waiting_for_limits(item["claim"], retriever, facts, llm)
        except PreCueError as error:
            failed += 1
            print(f"FAIL expected {item['verdict']:<15} {error} | {item['claim']}")
            continue
        times.append(seconds)

        ok = verdict["verdict"] == item["verdict"]
        right += ok
        caught += item["verdict"] == "wrong" and verdict["verdict"] == "wrong"
        score = by_kind.setdefault(item.get("kind", "all"), [0, 0])
        score[0] += ok
        score[1] += 1
        if verdict["verdict"] == "wrong" and item["verdict"] != "wrong":
            false_alerts += 1
            false_alert_claims.append(f"{item['claim']}  (said {verdict['said']!r}, document {verdict['document_says']!r})")
        mark = "OK  " if ok else "MISS"
        print(f"{mark} expected {item['verdict']:<15} got {verdict['verdict']:<15} | {item['claim']}")

    if len(by_kind) > 1:
        print()
        for kind, (good, total) in by_kind.items():
            print(f"  {kind:<32} {good}/{total}")
    for claim in false_alert_claims:
        print(f"  FALSE ALERT: {claim}")
    average = sum(times) / len(times) * 1000 if times else 0
    wrong = sum(item["verdict"] == "wrong" for item in claims)
    print(f"\nAccuracy: {right}/{len(claims)}   Failed checks: {failed}   Average time: {average:.0f} ms")
    print(f"Wrong claims caught: {caught}/{wrong}   False alerts: {false_alerts}/{len(claims) - wrong}")


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}") from None
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
