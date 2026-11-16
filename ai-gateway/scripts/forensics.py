"""Why was an answer bad? A deterministic checklist over an eval run (and its traces).

From ai-gateway/ (runs are in benchmarks/runs/<name>.json):
    python -m scripts.forensics case --run candidate --baseline base --id fact_05
    python -m scripts.forensics failures --run candidate --baseline base
    python -m scripts.forensics failures --run smart60          # without a baseline

The trace rows come from Postgres through the request id of each answer. If the database is
not reachable the checks that need traces are reported as not checked.
"""

import argparse
import asyncio
import os
import sys

from app.evals.cases import load_cases
from app.evals.runner import load_run
from app.evals.scoring import score_case
from app.forensics.diagnose import Diagnosis, Observation, cause_counts, diagnose
from app.forensics.traces import fetch_traces
from app.settings import BASE_DIR

BENCH = BASE_DIR / "benchmarks"


def load_named(runs_dir, name):
    path = os.path.join(runs_dir, f"{name}.json")
    if not os.path.exists(path):
        sys.exit(f"no run file {path}")
    return load_run(path)


def get_traces(runs) -> dict:
    ids = [r.request_id for run in runs if run for r in run.results if r.request_id]
    if not ids:
        return {}
    try:
        return asyncio.run(fetch_traces(ids))
    except Exception as e:
        print(f"(trace rows not available: {type(e).__name__}. The checks that need them are not done.)\n")
        return {}


def observation(run, result, traces) -> Observation:
    return Observation(result, run.config, traces.get(result.request_id) if result.request_id else None)


def latency_p95(run) -> float | None:
    values = sorted(r.latency_ms for r in run.results if r.status == "ok" and r.latency_ms)
    return values[int(0.95 * (len(values) - 1))] if values else None


def print_diagnosis(d: Diagnosis) -> None:
    print(f"{d.case_id}: {'FAILED' if d.failed else 'passed'}")
    for f in d.findings:
        print(f"  [{f.role:<7}] {f.message}")
    print(f"  conclusion: {d.conclusion()}\n")


def do_case(args) -> None:
    cases = {c.id: c for c in load_cases(args.cases)}
    case = cases.get(args.id) or sys.exit(f"no case '{args.id}' in {args.cases}")
    run = load_named(args.runs_dir, args.run)
    base_run = load_named(args.runs_dir, args.baseline) if args.baseline else None
    result = next((r for r in run.results if r.id == args.id), None) or sys.exit(f"the run has no result for '{args.id}'")
    base_result = next((r for r in base_run.results if r.id == args.id), None) if base_run else None

    traces = get_traces([run, base_run])
    print_diagnosis(
        diagnose(
            case,
            observation(run, result, traces),
            observation(base_run, base_result, traces) if base_result else None,
            latency_p95(base_run) if base_run else None,
        )
    )
    if args.answers:
        print(f"candidate answer: {result.answer!r}")
        if base_result:
            print(f"baseline answer:  {base_result.answer!r}")


def do_failures(args) -> None:
    cases = {c.id: c for c in load_cases(args.cases)}
    run = load_named(args.runs_dir, args.run)
    base_run = load_named(args.runs_dir, args.baseline) if args.baseline else None
    base_results = {r.id: r for r in base_run.results} if base_run else {}
    traces = get_traces([run, base_run])
    p95 = latency_p95(base_run) if base_run else None

    diagnoses = []
    for result in run.results:
        case = cases.get(result.id)
        if case is None:
            continue
        base_result = base_results.get(result.id)
        diagnosis = diagnose(
            case,
            observation(run, result, traces),
            observation(base_run, base_result, traces) if base_result else None,
            p95,
        )
        diagnoses.append(diagnosis)
        if diagnosis.failed and not args.summary:
            print_diagnosis(diagnosis)

    failed = [d for d in diagnoses if d.failed]
    print(f"{len(failed)} failed cases out of {len(diagnoses)} in run '{run.name}'")
    for kind, count in cause_counts(diagnoses).most_common():
        print(f"  {kind:<18}{count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True, help="name of the run to explain")
    common.add_argument("--baseline", help="name of a run to compare with")
    common.add_argument("--cases", default=str(BENCH / "eval_cases.jsonl"))
    common.add_argument("--runs-dir", default=str(BENCH / "runs"))
    sub = parser.add_subparsers(dest="command", required=True)

    case = sub.add_parser("case", parents=[common])
    case.add_argument("--id", required=True)
    case.add_argument("--answers", action="store_true", help="also print the two answers")
    case.set_defaults(func=do_case)

    failures = sub.add_parser("failures", parents=[common])
    failures.add_argument("--summary", action="store_true", help="only the counts")
    failures.set_defaults(func=do_failures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()