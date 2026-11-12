"""Compare two eval runs and decide whether a change made answers worse.

Usage (from ai-gateway/):
    python -m scripts.eval_compare benchmarks/runs/strong20b.json benchmarks/runs/candidate.json

The exit code is 1 on a REGRESSION (use it to block a change), 2 when there is not enough
data, otherwise 0.
"""

import argparse
import asyncio
import os
import sys

from app.evals.calibration import is_trusted, load_calibration
from app.evals.compare import (
    JUDGE_WEIGHT,
    compare_scores,
    comparison_lines,
    exit_code,
    final_scores,
)
from app.evals.cases import load_cases
from app.evals.judge import JudgeCache, JudgeConfig, judge_many
from app.evals.runner import EvalAbort, load_run
from app.evals.scoring import score_run
from app.settings import BASE_DIR

BENCH = BASE_DIR / "benchmarks"


def judge_verdicts(run, run_score, cases_by_id, config, cache, api_key) -> dict:
    answers = {r.id: r.answer or "" for r in run.results if r.status == "ok"}
    pairs = [(cases_by_id[c.id], answers[c.id]) for c in run_score.valid]
    results = asyncio.run(judge_many(pairs, config, api_key, cache))
    return {case.id: r.verdict for (case, _), r in zip(pairs, results) if r.verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--cases", default=str(BENCH / "eval_cases.jsonl"))
    parser.add_argument("--min-drop", type=float, default=0.02, help="smallest average drop (0 to 1) that counts, default 0.02")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--judge", action="store_true", help=f"add a trusted judge ({int(JUDGE_WEIGHT * 100)}% of the score)")
    parser.add_argument("--calibration", default=str(BENCH / "judge_calibration.json"))
    parser.add_argument("--cache", default=str(BENCH / "judge_cache.jsonl"))
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    cases_by_id = {c.id: c for c in cases}
    baseline_run, candidate_run = load_run(args.baseline), load_run(args.candidate)
    baseline_score, candidate_score = score_run(cases, baseline_run), score_run(cases, candidate_run)

    baseline_verdicts = candidate_verdicts = None
    if args.judge:
        config = JudgeConfig(model=args.judge_model, base_url=args.gateway)
        if not is_trusted(load_calibration(args.calibration), config.model, config.rubric_version):
            print("The judge is not trusted for this model and rubric (run: python -m scripts.eval_judge calibrate).")
            print("Using the deterministic checks only.\n")
        else:
            api_key = os.environ.get("EVAL_API_KEY") or sys.exit("EVAL_API_KEY is not set")
            cache = JudgeCache(args.cache)
            try:
                baseline_verdicts = judge_verdicts(baseline_run, baseline_score, cases_by_id, config, cache, api_key)
                candidate_verdicts = judge_verdicts(candidate_run, candidate_score, cases_by_id, config, cache, api_key)
            except EvalAbort as e:
                sys.exit(f"stopped: {e}")
            print(f"Final score: {int((1 - JUDGE_WEIGHT) * 100)}% checks plus {int(JUDGE_WEIGHT * 100)}% judge.\n")

    categories = {c.id: c.category for c in baseline_score.cases + candidate_score.cases}
    comparison = compare_scores(
        final_scores(baseline_score, baseline_verdicts),
        final_scores(candidate_score, candidate_verdicts),
        categories,
        min_drop=args.min_drop,
        iterations=args.bootstrap,
        seed=args.seed,
        min_cases=args.min_cases,
    )

    print(f"baseline  '{baseline_run.name}': {baseline_run.config.get('model')} (prompt {baseline_run.config.get('prompt')}:{baseline_run.config.get('version')})")
    print(f"candidate '{candidate_run.name}': {candidate_run.config.get('model')} (prompt {candidate_run.config.get('prompt')}:{candidate_run.config.get('version')})")
    print()
    for line in comparison_lines(comparison):
        print(line)

    def avg(value, unit):
        return "n/a" if value is None else f"{value:.0f} {unit}"

    print()
    print(f"  latency {avg(baseline_score.avg_latency_ms, 'ms')} -> {avg(candidate_score.avg_latency_ms, 'ms')}, "
          f"tokens {avg(baseline_score.avg_tokens, '')} -> {avg(candidate_score.avg_tokens, '')}")
    sys.exit(exit_code(comparison.verdict))


if __name__ == "__main__":
    main()