"""Seeded bad changes: measure the regression gate itself.

Steps (from ai-gateway/, gateway running, EVAL_API_KEY in .env):
    python -m scripts.eval_seeded setup     # creates the prompt "seeded_eval" and its versions
    python -m scripts.eval_seeded run       # evaluates every version (can take 20 to 30 minutes)
    python -m scripts.eval_seeded report --markdown
"""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time

from sqlalchemy import select

from app.db.models import Prompt, PromptVersion
from app.db.session import SessionLocal, engine
from app.evals.calibration import load_calibration
from app.evals.cases import load_cases
from app.evals.compare import comparison_lines, compare_scores, final_scores
from app.evals.runner import EvalAbort, RunConfig, load_run, run_cases, save_run
from app.evals.scoring import score_run
from app.evals.seeded import (
    PROMPT_NAME,
    VARIANTS,
    build_row,
    pick_cases,
    readme_section,
    run_name,
    summarize,
    unique_texts,
)
from app.settings import BASE_DIR

BENCH = BASE_DIR / "benchmarks"


async def fetch_mapping(create: bool) -> dict:
    """system prompt text -> version number of the seeded prompt."""
    try:
        async with SessionLocal() as session:
            prompt = (
                await session.execute(select(Prompt).where(Prompt.name == PROMPT_NAME))
            ).scalar_one_or_none()
            if prompt is None:
                if not create:
                    return {}
                prompt = Prompt(name=PROMPT_NAME, description="Seeded bad changes for testing the regression gate")
                session.add(prompt)
                await session.flush()

            versions = (
                await session.execute(select(PromptVersion).where(PromptVersion.prompt_id == prompt.id))
            ).scalars().all()
            by_text = {v.system_prompt: v.version for v in versions}
            if create:
                number = max(by_text.values(), default=0) + 1
                for text in unique_texts():
                    if text not in by_text:
                        session.add(
                            PromptVersion(
                                prompt_id=prompt.id,
                                version=number,
                                system_prompt=text,
                                weight=0,  # never served by chance, only forced by the eval
                                notes="seeded eval",
                            )
                        )
                        by_text[text] = number
                        number += 1
                await session.commit()
            return by_text
    finally:
        await engine.dispose()


def do_setup(args) -> None:
    mapping = asyncio.run(fetch_mapping(create=True))
    print(f"prompt '{PROMPT_NAME}' is ready:")
    for variant in VARIANTS:
        print(f"  {variant.name:<16} {variant.kind:<16} version {mapping[variant.system_prompt]}")
    print("\nThe gateway remembers prompts for 10 seconds: wait a little before 'run'.")


def do_run(args) -> None:
    api_key = os.environ.get("EVAL_API_KEY") or sys.exit("EVAL_API_KEY is not set (ai-gateway/.env)")
    mapping = asyncio.run(fetch_mapping(create=False))
    if not mapping:
        sys.exit("run 'python -m scripts.eval_seeded setup' first")

    cases = pick_cases(load_cases(args.cases), args.per_category)
    selected = [v for v in VARIANTS if not args.variants or v.name in args.variants]
    for v in selected:
        if v.system_prompt not in mapping:
            sys.exit(f"variant '{v.name}' has no version yet, run setup again")

    print(f"{len(cases)} cases per run, {len(selected)} runs, model {args.model}")
    if args.wait > 0:
        time.sleep(args.wait)  # the gateway caches prompt lookups for a few seconds

    for variant in selected:
        path = os.path.join(args.runs_dir, f"{run_name(variant)}.json")
        if os.path.exists(path) and not args.force:
            print(f"{variant.name}: already done, skipping (use --force to run again)")
            continue
        config = RunConfig(
            name=run_name(variant),
            base_url=args.gateway,
            model=args.model,
            prompt=PROMPT_NAME,
            version=mapping[variant.system_prompt],
            delay_seconds=args.delay,
        )
        print(f"\n=== {variant.name} ({variant.kind}): version {config.version} ===")

        def progress(index, total, case, result):
            note = f" ({result.error})" if result.error else ""
            print(f"[{index:>3}/{total}] {case.id:<22} {result.status}{note}", flush=True)

        try:
            run = asyncio.run(run_cases(cases, config, api_key, progress=progress))
        except EvalAbort as e:
            sys.exit(f"stopped: {e}")
        save_run(run, path)
        invalid = sum(1 for r in run.results if r.status != "ok")
        print(f"saved {path} ({invalid} invalid)")


def do_report(args) -> None:
    cases = load_cases(args.cases)
    categories = {c.id: c.category for c in cases}

    def load(variant):
        path = os.path.join(args.runs_dir, f"{run_name(variant)}.json")
        return load_run(path) if os.path.exists(path) else None

    baseline_run = load(VARIANTS[0])
    if baseline_run is None:
        sys.exit("the baseline run is missing. Run: python -m scripts.eval_seeded run")
    baseline = final_scores(score_run(cases, baseline_run))

    rows = []
    for variant in VARIANTS[1:]:
        run = load(variant)
        if run is None:
            print(f"{variant.name}: no run yet, skipped")
            continue
        comparison = compare_scores(
            baseline,
            final_scores(score_run(cases, run)),
            categories,
            min_drop=args.min_drop,
            iterations=args.bootstrap,
            seed=1,
            min_cases=args.min_cases,
        )
        rows.append(build_row(variant, comparison))
        if args.details:
            print(f"\n--- {variant.name} ({variant.kind}) ---")
            for line in comparison_lines(comparison):
                print(line)

    print(f"\n{'change':<18}{'kind':<17}{'verdict':<20}{'change (points)':>16}  interval")
    for r in rows:
        change = "n/a" if r.mean_diff is None else f"{r.mean_diff * 100:+.1f}"
        interval = "n/a" if r.ci_low is None else f"{r.ci_low * 100:+.1f} to {r.ci_high * 100:+.1f}"
        print(f"{r.variant:<18}{r.kind:<17}{r.verdict:<20}{change:>16}  {interval}")

    s = summarize(rows)
    print()
    print(f"Broken versions blocked: {s['bad_caught']} of {s['bad_total']} "
          f"(inconclusive drop: {s['bad_warned']}, missed: {s['bad_missed']})")
    print(f"Harmless versions wrongly blocked: {s['false_blocks']} of {s['control_total']} "
          f"(wrongly flagged as inconclusive: {s['false_warnings']})")

    out = BENCH / "seeded_results.json"
    out.write_text(
        json.dumps({"summary": s, "rows": [dataclasses.asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {out}")

    if args.markdown:
        print("\n" + "=" * 70 + " paste this into README.md " + "=" * 70 + "\n")
        print(readme_section(rows, load_calibration(BENCH / "judge_calibration.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cases", default=str(BENCH / "eval_cases.jsonl"))
    common.add_argument("--runs-dir", default=str(BENCH / "runs"))
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", parents=[common])
    setup.set_defaults(func=do_setup)

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--model", default="openai/gpt-oss-20b")
    run.add_argument("--gateway", default="http://127.0.0.1:8000")
    run.add_argument("--per-category", type=int, default=6, help="cases per category, 0 for all")
    run.add_argument("--delay", type=float, default=2.0)
    run.add_argument("--wait", type=float, default=12.0, help="seconds to wait first (prompt cache of the gateway)")
    run.add_argument("--variants", nargs="*", help="only these variants")
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=do_run)

    report = sub.add_parser("report", parents=[common])
    report.add_argument("--min-drop", type=float, default=0.02)
    report.add_argument("--bootstrap", type=int, default=5000)
    report.add_argument("--min-cases", type=int, default=20)
    report.add_argument("--details", action="store_true", help="the full comparison of every version")
    report.add_argument("--markdown", action="store_true", help="also print the README section")
    report.set_defaults(func=do_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()