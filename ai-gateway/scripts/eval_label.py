"""Label answers by hand: good or bad. At least 50 labels are needed to check the LLM judge.

Usage (from ai-gateway/):
    python -m scripts.eval_label --runs benchmarks/runs/strong20b.json benchmarks/runs/weak05b.json --count 50

The model that gave each answer is hidden. You can quit with q and continue later: answers
that are already labeled are not shown again.
"""

import argparse

from app.evals.cases import load_cases
from app.evals.labels import append_label, collect_items, load_labels, pick_items
from app.evals.runner import load_run
from app.settings import BASE_DIR

GUIDE = """
How to label
  good = the answer is correct, follows EVERY instruction in the prompt (format, length,
         language, system prompt rules) and has nothing important wrong.
  bad  = a wrong fact or number, an ignored instruction, a refusal where the prompt is fine,
         nonsense, or a leaked secret.
Judge the answer against the prompt, not your taste in style. If you cannot tell, skip (s).
"""


def show(item, position: int, total: int) -> None:
    print("=" * 78)
    print(f"[{position}/{total}]")
    for message in item.case.messages:
        print(f"  {message['role'].upper()}: {message['content']}")
    print("-" * 78)
    answer = item.answer if len(item.answer) <= 1500 else item.answer[:1500] + "\n  ...(cut)"
    print(answer if answer.strip() else "  (empty answer)")
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run files to take answers from")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cases", default=str(BASE_DIR / "benchmarks" / "eval_cases.jsonl"))
    parser.add_argument("--labels", default=str(BASE_DIR / "benchmarks" / "labels.jsonl"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    runs = [load_run(path) for path in args.runs]
    items = collect_items(cases, runs)
    todo = pick_items(items, load_labels(args.labels), args.count, args.seed)
    if not todo:
        print("nothing left to label for these runs")
        return

    print(GUIDE)
    saved = 0
    for position, item in enumerate(todo, 1):
        show(item, position, len(todo))
        choice = ""
        while choice not in ("g", "b", "s", "q"):
            choice = input("[g]ood  [b]ad  [s]kip  [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        append_label(args.labels, item, "good" if choice == "g" else "bad")
        saved += 1

    labels = load_labels(args.labels)
    good = sum(1 for label in labels if label["label"] == "good")
    print(f"\nsaved {saved} new labels. Total: {len(labels)} ({good} good, {len(labels) - good} bad)")


if __name__ == "__main__":
    main()