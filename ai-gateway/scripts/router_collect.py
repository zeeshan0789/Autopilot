"""Collect model answers for the learned router and report where a router could help.

From ai-gateway/ (gateway running, EVAL_API_KEY in .env):
    python -m scripts.router_collect generate
    python -m scripts.router_collect collect --model qwen2.5:0.5b --delay 0.2 --limit 10   # a small try first
    python -m scripts.router_collect collect --model qwen2.5:0.5b --delay 0.2
    python -m scripts.router_collect collect --model openai/gpt-oss-20b --delay 1.5
    python -m scripts.router_collect collect --model openai/gpt-oss-120b --delay 1.5
    python -m scripts.router_collect collect --model <model> --cases benchmarks/eval_cases.jsonl   # the 60 hand written cases
    python -m scripts.router_collect report
"""

import argparse
import asyncio
import os
import sys
from collections import Counter

from app.evals.cases import load_cases
from app.evals.runner import EvalAbort, RunConfig, check_gateway
from app.mlrouter.collect import collect_outcomes
from app.mlrouter.data import (
    family_matrix,
    load_outcomes,
    model_stats,
    oracle,
    parse_case_id,
    pending_cases,
    split_of,
)
from app.mlrouter.generate import generate_cases, write_cases
from app.settings import BASE_DIR

ROUTER_DIR = BASE_DIR / "benchmarks" / "router"
CASES_FILE = ROUTER_DIR / "cases.jsonl"
OUTCOMES_FILE = ROUTER_DIR / "outcomes.jsonl"
HANDWRITTEN_FILE = BASE_DIR / "benchmarks" / "eval_cases.jsonl"
MODELS = ["qwen2.5:0.5b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # cheapest first


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def money(value) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def short(model: str) -> str:
    return model.split("/")[-1][:14]


def do_generate(args) -> None:
    cases = generate_cases()
    write_cases(args.out, cases)
    print(f"{len(cases)} cases written to {args.out}")
    split = Counter(split_of(c.id) for c in cases)
    print(f"train {split['train']}, test {split['test']}")
    families = Counter(parse_case_id(c.id)[0] for c in cases)
    print("families:", ", ".join(f"{name} {count}" for name, count in families.items()))


def do_collect(args) -> None:
    api_key = os.environ.get("EVAL_API_KEY") or sys.exit("EVAL_API_KEY is not set (ai-gateway/.env)")
    try:
        check_gateway(args.gateway)
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")

    cases = load_cases(args.cases)
    todo = pending_cases(cases, load_outcomes(OUTCOMES_FILE), args.model)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.model}: {len(todo)} cases to collect ({len(cases) - len(todo)} already done or skipped)")
    if not todo:
        return

    config = RunConfig(
        name=f"collect_{short(args.model)}",
        base_url=args.gateway,
        model=args.model,
        delay_seconds=args.delay,
        retries=args.retries,
    )

    def progress(index, total, case, result):
        note = f" ({result.error})" if result.error else ""
        print(f"[{index:>3}/{total}] {case.id:<26} {result.status}{note}", flush=True)

    try:
        summary = asyncio.run(collect_outcomes(todo, config, api_key, OUTCOMES_FILE, progress=progress))
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")
    print(f"\nsaved {summary.saved}, failed {summary.failed}")
    if summary.stopped_early:
        print(f"STOPPED EARLY: {summary.reason}")


def print_set(title, cases, outcomes, models) -> None:
    print(f"\n{title} ({len(cases)} cases)")
    print(f"  {'model':<16}{'cases':>6}{'pass rate':>11}{'mean score':>12}{'avg ms':>9}{'avg tokens':>12}{'avg cost':>12}")
    for s in model_stats(cases, outcomes, models):
        latency = "n/a" if s.avg_latency_ms is None else f"{s.avg_latency_ms:.0f}"
        tokens = "n/a" if s.avg_tokens is None else f"{s.avg_tokens:.0f}"
        print(f"  {short(s.model):<16}{s.cases:>6}{pct(s.pass_rate):>11}{pct(s.mean_score):>12}{latency:>9}{tokens:>12}{money(s.avg_cost_usd):>12}")

    result = oracle(cases, outcomes, models)
    print(f"  Best any router could do (cheapest model that passes each case), on the {result.cases} cases every model answered:")
    if result.cases == 0:
        print("    no case has answers from every model yet")
        return
    print(f"    always {short(models[0])}:   pass {pct(result.cheapest_pass_rate)}, avg cost {money(result.cheapest_avg_cost_usd)}")
    print(f"    always {short(models[-1])}: pass {pct(result.strongest_pass_rate)}, avg cost {money(result.strongest_avg_cost_usd)}")
    print(f"    perfect router:          pass {pct(result.oracle_pass_rate)}, avg cost {money(result.oracle_avg_cost_usd)}")
    shares = ", ".join(f"{short(m)} {count}" for m, count in result.chosen.items())
    print(f"    the perfect router would pick: {shares}")


def do_report(args) -> None:
    outcomes = load_outcomes(OUTCOMES_FILE)
    models = args.models
    synthetic = load_cases(args.cases) if os.path.exists(args.cases) else []
    handwritten = load_cases(HANDWRITTEN_FILE) if os.path.exists(HANDWRITTEN_FILE) else []
    print(f"{len(outcomes)} usable answers in {OUTCOMES_FILE}")

    if synthetic:
        print_set("Generated cases, train", [c for c in synthetic if split_of(c.id) == "train"], outcomes, models)
        print_set("Generated cases, test", [c for c in synthetic if split_of(c.id) == "test"], outcomes, models)
        print("\nPass rate by family and level (all generated cases; cases / pass rate for each model)")
        print(f"  {'family':<18}" + "".join(f"{short(m):>16}" for m in models))
        for key, row in family_matrix(synthetic, outcomes, models).items():
            cells = "".join(f"{f'{n} / {pct(rate)}':>16}" for n, rate in row.values())
            print(f"  {key:<18}{cells}")
    if handwritten:
        print_set("Hand written cases", handwritten, outcomes, models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--out", default=str(CASES_FILE))
    gen.set_defaults(func=do_generate)

    col = sub.add_parser("collect")
    col.add_argument("--model", required=True)
    col.add_argument("--cases", default=str(CASES_FILE))
    col.add_argument("--gateway", default="http://127.0.0.1:8000")
    col.add_argument("--delay", type=float, default=1.0)
    col.add_argument("--retries", type=int, default=3)
    col.add_argument("--limit", type=int)
    col.set_defaults(func=do_collect)

    rep = sub.add_parser("report")
    rep.add_argument("--cases", default=str(CASES_FILE))
    rep.add_argument("--models", nargs="+", default=MODELS, help="cheapest first")
    rep.set_defaults(func=do_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()