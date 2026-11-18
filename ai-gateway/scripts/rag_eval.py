"""RAG demo on top of the gateway: documents, chunking, embeddings, retrieval, an answer through the gateway.

From ai-gateway/ (the gateway must run and EVAL_API_KEY must be in .env for run and ask):
    python -m scripts.rag_eval build
    python -m scripts.rag_eval retrieval
    python -m scripts.rag_eval ask "How much does the Zorvex Pro cost?"
    python -m scripts.rag_eval run --mode closed_book
    python -m scripts.rag_eval run --mode rag
    python -m scripts.rag_eval run --mode oracle
    python -m scripts.rag_eval report --markdown
    python -m scripts.rag_eval failures --run rag
"""

import argparse
import asyncio
import os
import sys
from collections import Counter
from statistics import median

from app.cache.embedder import Embedder
from app.evals.cases import EvalCase
from app.evals.runner import EvalAbort, RunConfig, check_gateway, run_case
from app.rag.chunking import chunk_documents
from app.rag.corpus import generate, load_docs, load_questions, write_corpus
from app.rag.forensics import diagnose_rag
from app.rag.index import ChunkIndex
from app.rag.metrics import KS, abstention_sweep, failure_counts, recommend_min_score, retrieval_summary, summarize_items
from app.rag.pipeline import RETRIEVAL_DEPTH, RagConfig, build_messages, load_rag_run, run_rag, save_rag_run
from app.rag.report import markdown_section, pct, summary_lines
from app.settings import BASE_DIR

RAG_DIR = BASE_DIR / "benchmarks" / "rag"
RUNS_DIR = RAG_DIR / "runs"


def load_index(args):
    docs = load_docs(RAG_DIR / "docs.jsonl")
    questions = load_questions(RAG_DIR / "questions.jsonl")
    chunks = chunk_documents(docs, args.chunk_chars, args.overlap)
    return docs, questions, chunks, ChunkIndex(chunks, Embedder())


def do_build(args) -> None:
    docs, questions = generate()
    write_corpus(RAG_DIR, docs, questions)
    kinds = Counter(q.kind for q in questions)
    print(f"{len(docs)} documents and {len(questions)} questions written to {RAG_DIR}")
    print(f"questions: {kinds['answerable']} answerable, {kinds['unanswerable']} unanswerable, {kinds['injection']} with an injected instruction")
    print("examples:")
    for q in [q for q in questions if q.kind == "answerable"][:3]:
        print(f"  {q.question}   (answer: {q.reference})")


def do_retrieval(args) -> None:
    docs, questions, chunks, index = load_index(args)
    print(f"{len(docs)} documents, {len(chunks)} chunks (up to {args.chunk_chars} characters, {args.overlap} sentence of overlap)")

    answerable = [q for q in questions if q.doc_id]
    ranks, kinds_of_doc, misses = [], {}, []
    for q in answerable:
        hits = index.search(q.question, RETRIEVAL_DEPTH)
        rank = next((h.rank for h in hits if h.chunk.doc_id == q.doc_id), None)
        ranks.append(rank)
        group = kinds_of_doc.setdefault(q.doc_id.split("_")[0], [])
        group.append(rank)
        if rank is None or rank > args.k:
            misses.append((q, rank, hits[: args.k]))

    summary = retrieval_summary(ranks)
    print(f"\nRetrieval on {summary['n']} answerable questions: "
          + ", ".join(f"hit@{k} {pct(summary[f'hit@{k}'])}" for k in KS) + f", MRR {summary['mrr']:.3f}")
    print("  by kind of document:")
    for name, group in sorted(kinds_of_doc.items()):
        s = retrieval_summary(group)
        print(f"    {name:<9}{s['n']:>3} questions  hit@1 {pct(s['hit@1'])}  hit@3 {pct(s['hit@3'])}  hit@5 {pct(s['hit@5'])}")

    if misses:
        print(f"\n  questions whose document is not in the first {args.k} chunks ({len(misses)}):")
        for q, rank, hits in misses[:12]:
            top = ", ".join(f"{h.chunk.doc_id} {h.score:.2f}" for h in hits)
            print(f"    {q.id} [{q.doc_id}] rank {rank}: {q.question}  ->  got {top}")

    top_answerable = [index.search(q.question, 1)[0].score for q in answerable]
    top_unanswerable = [index.search(q.question, 1)[0].score for q in questions if q.kind == "unanswerable"]
    print(f"\nBest retrieval score: answerable questions median {median(top_answerable):.3f} (lowest {min(top_answerable):.3f}), "
          f"unanswerable questions median {median(top_unanswerable):.3f} (highest {max(top_unanswerable):.3f})")
    sweep = abstention_sweep(top_answerable, top_unanswerable)
    print("  If the best score is below a minimum, answer 'not found' without asking a model:")
    print(f"    {'minimum':>8}{'answerable wrongly refused':>28}{'unanswerable rightly refused':>31}")
    for row in sweep:
        print(f"    {row['threshold']:>8.2f}{pct(row['false_abstain']):>28}{pct(row['correct_abstain']):>31}")
    recommended = recommend_min_score(sweep)
    if recommended is None:
        print("  no minimum keeps the wrongly refused answerable questions under 2%")
    else:
        print(f"  highest minimum with at most 2% wrongly refused: {recommended:.2f} (chosen on these same questions, so optimistic)")


def do_ask(args) -> None:
    api_key = os.environ.get("EVAL_API_KEY") or sys.exit("EVAL_API_KEY is not set (ai-gateway/.env)")
    try:
        check_gateway(args.gateway)
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")
    _, _, _, index = load_index(args)
    hits = index.search(args.question, args.k)
    print("Retrieved:")
    for hit in hits:
        print(f"  {hit.score:.3f}  {hit.chunk.id}  {hit.chunk.title}: {hit.chunk.text[:90]}...")

    config = RunConfig(name="ask", base_url=args.gateway, model=args.model, temperature=0.0)
    case = EvalCase("ask", "ask", build_messages("rag", args.question, hits), [{"type": "no_refusal"}])

    async def go():
        import httpx

        async with httpx.AsyncClient(base_url=args.gateway, headers={"Authorization": f"Bearer {api_key}"}, timeout=180.0, trust_env=False) as client:
            return await run_case(client, case, config)

    result = asyncio.run(go())
    print(f"\nAnswer ({result.served_model}): {result.answer}" if result.status == "ok" else f"\nNo answer: {result.error}")


def do_run(args) -> None:
    api_key = os.environ.get("EVAL_API_KEY") or sys.exit("EVAL_API_KEY is not set (ai-gateway/.env)")
    try:
        check_gateway(args.gateway)
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")
    _, questions, chunks, index = load_index(args)
    if args.limit:
        questions = questions[: args.limit]
    config = RagConfig(
        name=args.name or args.mode,
        mode=args.mode,
        model=args.model,
        base_url=args.gateway,
        k=args.k,
        min_score=args.min_score,
        delay_seconds=args.delay,
    )
    print(f"mode {config.mode}, {len(questions)} questions, model {config.model}, k={config.k}, minimum score {config.min_score}")

    def progress(number, total, item):
        note = "" if item.status == "ok" else f"  ({item.error})"
        flag = "pass" if item.passed else "FAIL"
        extra = "  abstained" if item.abstained else ""
        print(f"[{number:>3}/{total}] {item.question_id} {item.kind:<12} {flag}{extra}{note}", flush=True)

    try:
        run = asyncio.run(run_rag(questions, index, config, api_key, progress=progress))
    except EvalAbort as e:
        sys.exit(f"stopped: {e}")
    path = RUNS_DIR / f"{config.name}.json"
    save_rag_run(run, path)
    print(f"\nsaved to {path}\n")
    for line in summary_lines(config.name, summarize_items(run.items)):
        print(line)


def do_report(args) -> None:
    runs = {}
    for name in args.runs:
        path = RUNS_DIR / f"{name}.json"
        if path.exists():
            runs[name] = load_rag_run(path)
    if not runs:
        sys.exit("no runs found: python -m scripts.rag_eval run --mode rag")

    summaries = {name: summarize_items(run.items) for name, run in runs.items()}
    for name in runs:
        print()
        for line in summary_lines(name, summaries[name]):
            print(line)

    retrieval = failures = None
    if "rag" in runs:
        items = runs["rag"].items
        ranks = [i.gold_rank for i in items if i.kind != "unanswerable"]
        retrieval = retrieval_summary(ranks)
        failures = failure_counts(items)
        print("\nRAG retrieval: " + ", ".join(f"hit@{k} {pct(retrieval[f'hit@{k}'])}" for k in KS) + f", MRR {retrieval['mrr']:.3f}")
        print("Why the RAG answers failed: " + (", ".join(f"{n} {c}" for n, c in failures.most_common()) or "none failed"))

    if args.markdown:
        rag_config = runs["rag"].config if "rag" in runs else next(iter(runs.values())).config
        docs = load_docs(RAG_DIR / "docs.jsonl")
        questions = load_questions(RAG_DIR / "questions.jsonl")
        kinds = Counter(q.kind for q in questions)
        info = {
            "docs": len(docs),
            "chunks": len(chunk_documents(docs, args.chunk_chars, args.overlap)),
            "questions": len(questions),
            "answerable": kinds["answerable"],
            "unanswerable": kinds["unanswerable"],
            "injection": kinds["injection"],
            "model": rag_config["model"],
            "k": rag_config["k"],
        }
        print("\n" + "=" * 30 + " paste this into README.md " + "=" * 30 + "\n")
        print(markdown_section(summaries, retrieval, failures, info))


def do_failures(args) -> None:
    run = load_rag_run(RUNS_DIR / f"{args.run}.json")
    shown = 0
    for item in run.items:
        finding = diagnose_rag(item)
        if finding is None:
            continue
        shown += 1
        print(f"{item.question_id} [{item.kind}] {item.question}")
        print(f"    {finding.cause}: {finding.message}")
        for key, value in finding.evidence.items():
            print(f"      {key}: {value}")
    print(f"\n{shown} questions without a right answer in run '{run.name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--k", type=int, default=3, help="chunks given to the model")
    common.add_argument("--chunk-chars", type=int, default=450)
    common.add_argument("--overlap", type=int, default=1, help="sentences shared by neighbouring chunks")
    common.add_argument("--gateway", default="http://127.0.0.1:8000")
    common.add_argument("--model", default="openai/gpt-oss-20b")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build").set_defaults(func=do_build)
    sub.add_parser("retrieval", parents=[common]).set_defaults(func=do_retrieval)

    ask = sub.add_parser("ask", parents=[common])
    ask.add_argument("question")
    ask.set_defaults(func=do_ask)

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--mode", choices=["rag", "closed_book", "oracle"], required=True)
    run.add_argument("--name")
    run.add_argument("--min-score", type=float, default=0.0)
    run.add_argument("--delay", type=float, default=2.0)
    run.add_argument("--limit", type=int)
    run.set_defaults(func=do_run)

    report = sub.add_parser("report", parents=[common])
    report.add_argument("--runs", nargs="+", default=["closed_book", "rag", "oracle"])
    report.add_argument("--markdown", action="store_true")
    report.set_defaults(func=do_report)

    failures = sub.add_parser("failures")
    failures.add_argument("--run", default="rag")
    failures.set_defaults(func=do_failures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()