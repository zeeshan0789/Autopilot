"""Run the eval cases through the gateway and save the answers.

Put a gateway key in ai-gateway/.env:  EVAL_API_KEY=ap-...
(create one with: python -m scripts.create_key eval 600)

Examples (from ai-gateway/):
    python -m scripts.eval_run --name strong20b --model openai/gpt-oss-20b
    python -m scripts.eval_run --name weak05b --model qwen2.5:0.5b --delay 0.5
    python -m scripts.eval_run --name smoke --limit 5
    python -m scripts.eval_run --name support_v2 --model auto --prompt support --version 2
"""

import argparse
import asyncio
import os
import sys

from app.evals.cases import load_cases
from app.evals.runner import EvalAbort, RunConfig, run_cases, save_run
from app.evals.scoring import score_run, summary_lines
from app.settings import BASE_DIR  # importing settings also loads ai-gateway/.env


def parse_variables(items: list[str]) -> dict:
    variables = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"bad --var '{item}', use name=value")
        variables[key] = value
    return variables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="name of the run (also the file name)")
    parser.add_argument("--cases", default=str(BASE_DIR / "benchmarks" / "eval_cases.jsonl"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "benchmarks" / "runs"))
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", help="a managed prompt to evaluate")
    parser.add_argument("--version", type=int, help="force one version of the managed prompt")
    parser.add_argument("--var", action="append", default=[], help="prompt variable, name=value (repeat)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between cases (free tier limits)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--allow-fallback", action="store_true", help="score answers even when a fallback model gave them")
    parser.add_argument("--category", help="only this category")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    args = parser.parse_args()

    api_key = os.environ.get("EVAL_API_KEY")
    if not api_key:
        sys.exit("EVAL_API_KEY is not set. Add EVAL_API_KEY=ap-... to ai-gateway/.env")

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("no cases selected")

    config = RunConfig(
        name=args.name,
        base_url=args.gateway,
        model=args.model,
        prompt=args.prompt,
        version=args.version,
        variables=parse_variables(args.var),
        temperature=args.temperature,
        delay_seconds=args.delay,
        retries=args.retries,
        allow_fallback=args.allow_fallback,
    )

    def progress(index, total, case, result):
        extra = f" ({result.error})" if result.error else ""
        print(f"[{index:>3}/{total}] {case.id:<22} {result.status}{extra}", flush=True)

    print(f"Running {len(cases)} cases on {config.model} through {config.base_url}")
    try:
        run = asyncio.run(run_cases(cases, config, api_key, progress=progress))
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")

    path = os.path.join(args.out_dir, f"{args.name}.json")
    save_run(run, path)
    print(f"\nsaved to {path}\n")
    for line in summary_lines(score_run(cases, run)):
        print(line)


if __name__ == "__main__":
    main()