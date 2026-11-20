"""Text to SQL demo on top of the gateway.

From ai-gateway/ (the gateway must run and EVAL_API_KEY must be in .env for ask and run):
    python -m scripts.sql_eval build
    python -m scripts.sql_eval check
    python -m scripts.sql_eval ask "How many customers are from Lisbon?"
    python -m scripts.sql_eval run --variant schema_only --name only_20b
    python -m scripts.sql_eval run --variant schema_values --name values_20b
    python -m scripts.sql_eval run --variant schema_values --repair --name values_repair_20b
    python -m scripts.sql_eval run --variant schema_values --model qwen2.5:0.5b --name values_qwen --delay 0.3
    python -m scripts.sql_eval report --markdown
    python -m scripts.sql_eval failures --run values_20b
"""

import argparse
import asyncio
import os
import sys
from collections import Counter

from app.evals.cases import EvalCase
from app.evals.runner import EvalAbort, RunConfig, check_gateway, run_case
from app.settings import BASE_DIR
from app.sqlbench.database import build_database, schema_text
from app.sqlbench.forensics import diagnose_sql
from app.sqlbench.pipeline import SqlConfig, build_messages, gold_results, load_run, run_sql_eval, save_run
from app.sqlbench.questions import generate_questions, load_questions, write_questions
from app.sqlbench.report import failure_counts, markdown_section, summarize, summary_lines
from app.sqlbench.safety import extract_sql, run_query

SQL_DIR = BASE_DIR / "benchmarks" / "sql"
DB_PATH = SQL_DIR / "shop.db"
QUESTIONS_PATH = SQL_DIR / "questions.jsonl"
RUNS_DIR = SQL_DIR / "runs"

SAFETY_DEMO = [
    "SELECT COUNT(*) FROM customers",
    "WITH t AS (SELECT 1 AS x) SELECT * FROM t",
    "DELETE FROM orders",
    "DROP TABLE customers",
    "UPDATE products SET price = 1",
    "SELECT 1; DROP TABLE orders",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE 'other.db' AS other",
    "SELECT * FROM sqlite_master",
    "SELECT randomblob(1000000000)",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r",
]


def need_key() -> str:
    return os.environ.get("EVAL_API_KEY") or sys.exit("EVAL_API_KEY is not set (ai-gateway/.env)")


def need_gateway(url: str) -> None:
    try:
        check_gateway(url)
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")


def do_build(args) -> None:
    build_database(DB_PATH)
    questions = generate_questions(DB_PATH)
    write_questions(QUESTIONS_PATH, questions)
    levels = Counter(q.level for q in questions if q.kind == "answerable")
    kinds = Counter(q.kind for q in questions)
    print(f"database written to {DB_PATH}")
    print(f"{len(questions)} questions: {kinds['answerable']} answerable (simple {levels[1]}, joins {levels[2]}, complex {levels[3]}), "
          f"{kinds['unsupported']} unsupported, {kinds['unsafe']} that ask to change data")


def do_check(args) -> None:
    questions = load_questions(QUESTIONS_PATH)
    gold = gold_results(questions, DB_PATH)
    print(f"all {len(gold)} gold queries run and return a result\n")
    for level in (1, 2, 3):
        example = next(q for q in questions if q.level == level)
        print(f"level {level}: {example.question}\n    {example.gold_sql}\n    -> {gold[example.id].rows[:3]}")
    print("\nThe safety layer against queries that a model could write:")
    for sql in SAFETY_DEMO:
        result = run_query(DB_PATH, sql, timeout_seconds=0.3)
        verdict = f"allowed, {len(result.rows)} row(s)" if result.ok else f"REFUSED ({result.kind}): {result.error}"
        print(f"  {sql[:70]:<72}{verdict}")


def do_ask(args) -> None:
    api_key = need_key()
    need_gateway(args.gateway)
    schema = schema_text(DB_PATH, rich=args.variant == "schema_values")
    config = RunConfig(name="ask", base_url=args.gateway, model=args.model, temperature=0.0)
    case = EvalCase("ask", "ask", build_messages(schema, args.question), [])

    async def go():
        import httpx

        async with httpx.AsyncClient(base_url=args.gateway, headers={"Authorization": f"Bearer {api_key}"}, timeout=180.0, trust_env=False) as client:
            return await run_case(client, case, config)

    result = asyncio.run(go())
    if result.status != "ok":
        sys.exit(f"no answer: {result.error}")
    extraction = extract_sql(result.answer or "")
    print(f"model reply ({result.served_model}): {result.answer}")
    if extraction.declined:
        print("the model declined (CANNOT_ANSWER)")
    elif extraction.sql is None:
        print("no SQL found in the reply")
    else:
        query = run_query(DB_PATH, extraction.sql)
        print(f"safety layer and database: {'ran' if query.ok else 'REFUSED (' + query.kind + '): ' + query.error}")
        if query.ok:
            print(f"result ({query.columns} column(s), {len(query.rows)} row(s)): {query.rows[:10]}")


def do_run(args) -> None:
    api_key = need_key()
    need_gateway(args.gateway)
    questions = load_questions(QUESTIONS_PATH)
    if args.limit:
        questions = questions[: args.limit]
    config = SqlConfig(
        name=args.name or f"{args.variant}_{args.model.split('/')[-1]}",
        model=args.model, variant=args.variant, repair=args.repair, base_url=args.gateway, delay_seconds=args.delay,
    )
    print(f"run '{config.name}': {len(questions)} questions, model {config.model}, prompt {config.variant}, repair {config.repair}")

    def progress(number, total, item):
        flag = "right" if item.correct else "WRONG"
        extra = f"  ({item.exec_kind})" if item.exec_kind not in (None, "ok") else ""
        print(f"[{number:>3}/{total}] {item.question_id} {item.kind:<12} {flag}{extra}", flush=True)

    try:
        run = asyncio.run(run_sql_eval(questions, DB_PATH, config, api_key, progress=progress))
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")
    path = RUNS_DIR / f"{config.name}.json"
    save_run(run, path)
    print(f"\nsaved to {path}\n")
    for line in summary_lines(config.name, summarize(run.items)):
        print(line)


def do_report(args) -> None:
    runs = {}
    names = args.runs or sorted(p.stem for p in RUNS_DIR.glob("*.json"))
    for name in names:
        path = RUNS_DIR / f"{name}.json"
        if path.exists():
            runs[name] = load_run(path)
    if not runs:
        sys.exit("no runs found: python -m scripts.sql_eval run")
    summaries = {name: summarize(run.items) for name, run in runs.items()}
    failures = {name: failure_counts(run.items) for name, run in runs.items()}
    for name in runs:
        print()
        for line in summary_lines(name, summaries[name]):
            print(line)
        print("  why the others failed: " + (", ".join(f"{c} {n}" for c, n in failures[name].most_common()) or "none"))
    if args.markdown:
        questions = load_questions(QUESTIONS_PATH)
        levels = Counter(q.level for q in questions if q.kind == "answerable")
        kinds = Counter(q.kind for q in questions)
        info = {"rows": "40 customers, 24 products, 200 orders", "questions": len(questions), "answerable": kinds["answerable"],
                "simple": levels[1], "joins": levels[2], "complex": levels[3], "unsupported": kinds["unsupported"], "unsafe": kinds["unsafe"]}
        print("\n" + "=" * 30 + " paste this into README.md " + "=" * 30 + "\n")
        print(markdown_section(summaries, failures, info))


def do_failures(args) -> None:
    run = load_run(RUNS_DIR / f"{args.run}.json")
    shown = 0
    for item in run.items:
        finding = diagnose_sql(item)
        if finding is None:
            continue
        shown += 1
        print(f"{item.question_id} [{item.kind}] {item.question}")
        print(f"    {finding.cause}: {finding.message}")
        for key, value in finding.evidence.items():
            if value not in (None, "", False):
                print(f"      {key}: {value}")
    print(f"\n{shown} questions without a right answer in run '{run.name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gateway", default="http://127.0.0.1:8000")
    common.add_argument("--model", default="openai/gpt-oss-20b")
    common.add_argument("--variant", choices=["schema_only", "schema_values"], default="schema_values")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build").set_defaults(func=do_build)
    sub.add_parser("check").set_defaults(func=do_check)

    ask = sub.add_parser("ask", parents=[common])
    ask.add_argument("question")
    ask.set_defaults(func=do_ask)

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--name")
    run.add_argument("--repair", action="store_true")
    run.add_argument("--delay", type=float, default=2.0)
    run.add_argument("--limit", type=int)
    run.set_defaults(func=do_run)

    report = sub.add_parser("report")
    report.add_argument("--runs", nargs="*")
    report.add_argument("--markdown", action="store_true")
    report.set_defaults(func=do_report)

    failures = sub.add_parser("failures")
    failures.add_argument("--run", required=True)
    failures.set_defaults(func=do_failures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()