"""Benchmark: the provider directly against the Autopilot gateway on a fixed workload.

From ai-gateway/ (needs GROQ_API_KEY in .env and Ollama running for the fallback; no gateway
server, Docker, Redis or Postgres are needed):
    python -m scripts.bench run                        # every mode that is not done yet
    python -m scripts.bench run --modes autopilot_strict
    python -m scripts.bench analyze                    # why did cache hits give wrong answers? (no API calls)
    python -m scripts.bench report --markdown
A finished mode is kept in benchmarks/bench/<mode>.json and is not run again without --force.
"""

import argparse
import asyncio
import dataclasses
import json
import sys
from collections import Counter

from app.bench.harness import MODES, BenchConfig, run_mode
from app.bench.results import (
    LABELS,
    ItemResult,
    classify_wrong_hits,
    markdown_section,
    paired_difference,
    summarize_mode,
    summary_lines,
)
from app.bench.workload import build_workload
from app.evals.cases import load_cases
from app.settings import BASE_DIR

BENCH_DIR = BASE_DIR / "benchmarks" / "bench"
CASES_FILE = BASE_DIR / "benchmarks" / "router" / "cases.jsonl"


def clip(text: str, size: int = 95) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= size else text[:size] + "..."


def load_results() -> tuple:
    results, configs = {}, {}
    for mode in MODES:
        path = BENCH_DIR / f"{mode}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            results[mode] = [ItemResult(**row) for row in data["items"]]
            configs[mode] = data["config"]
    return results, configs


def prompts_by_item() -> dict:
    """item id -> text of the last (user) message. The workload is rebuilt with the same seed."""
    workload = build_workload(load_cases(CASES_FILE))
    return {w.id: w.messages[-1]["content"] for w in workload}


def do_run(args) -> None:
    from app.cache.runtime import cache_service  # the shared embedding model

    workload = build_workload(load_cases(CASES_FILE))
    kinds = Counter(item.kind for item in workload)
    config = BenchConfig(model=args.model, delay_seconds=args.delay)
    start, end = config.outage_window(len(workload))
    print(f"{len(workload)} requests ({kinds['unique']} unique, {kinds['repeat']} repeats, {kinds['paraphrase']} reworded); "
          f"simulated outage during requests {start + 1} to {end}; model {config.model}")
    cache_service.embedder.encode("warm up")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        path = BENCH_DIR / f"{mode}.json"
        if path.exists() and not args.force:
            print(f"{mode}: already done, skipping (use --force to run it again)")
            continue
        print(f"\n=== {LABELS[mode]} ===")

        def progress(index, total, item, result):
            note = ""
            if result.status != "ok":
                note = f"  ERROR {result.error}"
            elif result.cache_status == "hit":
                note = "  cache hit" + ("" if result.passed else "  WRONG ANSWER")
            elif result.fallback_used:
                note = f"  fallback ({result.served_model})"
            flag = "outage" if result.in_outage else "      "
            print(f"[{index:>3}/{total}] {item.kind:<10} {flag} {result.latency_ms:7.0f} ms {'pass' if result.passed else 'FAIL'}{note}", flush=True)

        items = asyncio.run(run_mode(mode, workload, config, cache_service.embedder, progress=progress))
        payload = {
            "mode": mode,
            "config": {**dataclasses.asdict(config), "requests": len(workload), "outage_start": start, "outage_end": end,
                       "kinds": f"{kinds['unique']} unique prompts, {kinds['repeat']} exact repeats, {kinds['paraphrase']} lightly reworded repeats"},
            "items": [dataclasses.asdict(i) for i in items],
        }
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        for line in summary_lines(summarize_mode(mode, items)):
            print(line)


def do_analyze(args) -> None:
    results, _ = load_results()
    prompts = prompts_by_item()
    for mode, items in results.items():
        if mode == "direct":
            continue
        wrong = classify_wrong_hits(items, prompts)
        counts = Counter(w.kind for w in wrong)
        hits = sum(1 for i in items if i.cache_status == "hit")
        print(f"\n{LABELS[mode]}: {len(wrong)} wrong answers among {hits} cache hits "
              f"({counts['replay']} repeats of an earlier wrong answer, {counts['cross_prompt']} answers written for another prompt, {counts['unknown']} unknown)")
        for w in wrong:
            print(f"  {w.item_id} [{w.kind}]")
            print(f"      asked:    {clip(w.prompt)}")
            if w.source_prompt is not None:
                print(f"      got the answer of: {clip(w.source_prompt)}")
            print(f"      answer:   {clip(w.answer or '', 60)}")


def do_report(args) -> None:
    results, configs = load_results()
    if "direct" not in results:
        sys.exit("the 'direct' run is missing: python -m scripts.bench run --modes direct")
    prompts = prompts_by_item()

    summaries = {mode: summarize_mode(mode, items) for mode, items in results.items()}
    others = [mode for mode in results if mode != "direct"]
    diffs = {mode: paired_difference(results["direct"], results[mode]) for mode in others}
    part_diffs = {
        mode: {
            "normal": paired_difference(results["direct"], results[mode], outage=False),
            "outage": paired_difference(results["direct"], results[mode], outage=True),
        }
        for mode in others
    }
    wrong_kinds = {mode: Counter(w.kind for w in classify_wrong_hits(results[mode], prompts)) for mode in others}

    for mode in results:
        print()
        for line in summary_lines(summaries[mode]):
            print(line)
        if mode in wrong_kinds:
            kinds = wrong_kinds[mode]
            print(f"  wrong cache answers: {kinds['replay']} repeats of an earlier wrong answer, {kinds['cross_prompt']} answers written for another prompt, {kinds['unknown']} unknown")
    for mode in others:
        print(f"\n{LABELS[mode]} against direct (correct answers):")
        for label, diff in (("all requests", diffs[mode]), ("outside the outage", part_diffs[mode]["normal"]), ("during the outage", part_diffs[mode]["outage"])):
            print(f"  {label:<20}{diff['mean'] * 100:+6.1f} points (95% interval {diff['low'] * 100:+.1f} to {diff['high'] * 100:+.1f}, {diff['n']} requests)")
    if args.markdown:
        print("\n" + "=" * 30 + " paste this into README.md " + "=" * 30 + "\n")
        print(markdown_section(summaries, diffs, configs["direct"], part_diffs, wrong_kinds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    run.add_argument("--model", default="openai/gpt-oss-20b")
    run.add_argument("--delay", type=float, default=2.0, help="seconds after every real provider call")
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=do_run)

    sub.add_parser("analyze").set_defaults(func=do_analyze)

    report = sub.add_parser("report")
    report.add_argument("--markdown", action="store_true")
    report.set_defaults(func=do_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()