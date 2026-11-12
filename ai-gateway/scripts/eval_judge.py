"""Grade answers with the LLM judge, and check the judge against your hand labels.

Usage (from ai-gateway/, with EVAL_API_KEY in .env and the gateway running):
    python -m scripts.eval_judge calibrate
    python -m scripts.eval_judge run strong20b
"""

import argparse
import asyncio
import glob
import json
import os
import sys

from app.evals.calibration import Pair, calibrate, collect_labeled, save_calibration
from app.evals.cases import load_cases
from app.evals.judge import RUBRIC_VERSION, JudgeCache, JudgeConfig, judge_many
from app.evals.labels import load_labels
from app.evals.runner import EvalAbort, load_run
from app.evals.scoring import score_run
from app.settings import BASE_DIR

BENCH = BASE_DIR / "benchmarks"


def load_runs(runs_dir) -> dict:
    runs = {}
    for path in sorted(glob.glob(os.path.join(str(runs_dir), "*.json"))):
        try:
            run = load_run(path)
        except (KeyError, ValueError, TypeError):
            continue
        runs[run.name] = run
    return runs


def clip(text: str, size: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= size else text[:size] + "..."


def judge_config(args) -> JudgeConfig:
    return JudgeConfig(model=args.judge_model, base_url=args.gateway, delay_seconds=args.delay)


def progress(index, total, case, result):
    state = "cached" if result.cached else (result.verdict or f"no verdict ({result.error})")
    print(f"[{index:>3}/{total}] {case.id:<22} {state}", flush=True)


def percent(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def do_calibrate(args) -> None:
    api_key = require_key()
    cases = load_cases(args.cases)
    runs = load_runs(args.runs_dir)
    items, stale, missing = collect_labeled(load_labels(args.labels), runs, cases)
    print(f"{len(items)} usable labels ({stale} stale because the answer changed, {missing} missing run or case)")
    if not items:
        sys.exit("no labels to calibrate with. Label answers first: python -m scripts.eval_label")

    config = judge_config(args)
    cache = JudgeCache(args.cache)
    print(f"Judge {config.model}, rubric {config.rubric_version}. Cached verdicts are reused.\n")
    try:
        results = asyncio.run(
            judge_many([(case, answer) for _, case, answer in items], config, api_key, cache, progress=progress)
        )
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")

    pairs = []
    disagreements = []
    unparsable = 0
    for (label, case, answer), result in zip(items, results):
        if result.verdict is None:
            unparsable += 1
            continue
        pair = Pair(label["label"] == "good", result.verdict == "good")
        pairs.append(pair)
        if pair.human_good != pair.judge_good:
            disagreements.append((label, case, answer, result))

    result = calibrate(pairs, config.model, config.rubric_version, unparsable)
    save_calibration(args.out, result)

    print()
    print(f"Judge against your labels ({result.n} answers, {result.unparsable} without a verdict)")
    print("                  judge good   judge bad")
    print(f"  you: good       {result.tp:>9}   {result.fn:>9}")
    print(f"  you: bad        {result.fp:>9}   {result.tn:>9}")
    low, high = result.accuracy_interval
    print(f"  agreement {percent(result.accuracy)} (95% interval {percent(low)} to {percent(high)})")
    if result.kappa is not None:
        interval = "" if result.kappa_interval is None else f" (95% interval {result.kappa_interval[0]:.2f} to {result.kappa_interval[1]:.2f})"
        print(f"  Cohen's kappa {result.kappa:.2f}{interval}   (0 is chance, 0.6 or more is substantial)")
    print(f"  you call {percent(result.human_good_rate)} of the answers good, the judge {percent(result.judge_good_rate)}")
    print(f"  the judge catches {percent(result.specificity)} of the answers you called bad")
    print(f"  the judge accepts {percent(result.sensitivity)} of the answers you called good")
    print()
    if result.trusted:
        print("RESULT: the judge is TRUSTED for this model and rubric.")
    else:
        print("RESULT: the judge is NOT trusted:")
        for reason in result.reasons:
            print(f"  - {reason}")
    print(f"saved to {args.out}")

    if disagreements:
        print(f"\nWhere the judge and you disagree ({len(disagreements)}): check your own label too, you can be the one who is wrong")
        for label, case, answer, result_item in disagreements[:15]:
            print(f"  {case.id}: you {label['label']}, judge {result_item.verdict}: {clip(result_item.reason)}")
            print(f"      prompt: {clip(case.messages[-1]['content'], 110)}")
            print(f"      answer: {clip(answer, 110)}")


def do_run(args) -> None:
    api_key = require_key()
    cases = load_cases(args.cases)
    runs = load_runs(args.runs_dir)
    if args.name not in runs:
        sys.exit(f"no run named '{args.name}' in {args.runs_dir}")
    run = runs[args.name]
    score = score_run(cases, run)
    by_id = {c.id: c for c in cases}
    answers = {r.id: r.answer or "" for r in run.results if r.status == "ok"}
    pairs = [(by_id[c.id], answers[c.id]) for c in score.valid]

    config = judge_config(args)
    try:
        results = asyncio.run(judge_many(pairs, config, api_key, JudgeCache(args.cache), progress=progress))
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")

    good = sum(1 for r in results if r.verdict == "good")
    judged = sum(1 for r in results if r.verdict)
    print(f"\nrun '{run.name}': the judge calls {good} of {judged} answers good ({percent(good / judged if judged else None)})")
    print(f"the checks say {percent(score.pass_rate)} of the cases pass every check")
    print("(the judge is only a trusted signal after: python -m scripts.eval_judge calibrate)")


def require_key() -> str:
    key = os.environ.get("EVAL_API_KEY")
    if not key:
        sys.exit("EVAL_API_KEY is not set. Add EVAL_API_KEY=ap-... to ai-gateway/.env")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cases", default=str(BENCH / "eval_cases.jsonl"))
    common.add_argument("--runs-dir", default=str(BENCH / "runs"))
    common.add_argument("--cache", default=str(BENCH / "judge_cache.jsonl"))
    common.add_argument("--gateway", default="http://127.0.0.1:8000")
    common.add_argument("--judge-model", default="openai/gpt-oss-120b")
    common.add_argument("--delay", type=float, default=2.0)

    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", parents=[common])
    cal.add_argument("--labels", default=str(BENCH / "labels.jsonl"))
    cal.add_argument("--out", default=str(BENCH / "judge_calibration.json"))
    cal.set_defaults(func=do_calibrate)
    run = sub.add_parser("run", parents=[common])
    run.add_argument("name")
    run.set_defaults(func=do_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()