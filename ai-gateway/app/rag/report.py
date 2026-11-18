"""Text for the console and for the README."""

from app.rag.metrics import KS

MODE_LABELS = {"closed_book": "Closed book (no documents)", "rag": "RAG (retrieved chunks)", "oracle": "Oracle (the right document)"}


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def summary_lines(name: str, summary: dict) -> list[str]:
    lines = [f"{MODE_LABELS.get(name, name)}: {summary['questions']} questions, {summary['invalid']} without a usable answer, "
             f"{summary['abstained']} answered 'not found' by the retrieval score, {summary['llm_calls']} model calls"]
    for kind, row in summary["by_kind"].items():
        if row["n"]:
            low, high = row["interval"]
            lines.append(f"  {kind:<13} {row['passed']:>3} of {row['valid']:>3} right = {pct(row['rate'])} (95% interval {pct(low)} to {pct(high)})")
    tokens = "n/a" if summary["avg_tokens"] is None else f"{summary['avg_tokens']:.0f}"
    latency = "n/a" if summary["avg_latency_ms"] is None else f"{summary['avg_latency_ms']:.0f} ms"
    cost = "n/a" if summary["avg_cost_usd"] is None else f"${summary['avg_cost_usd']:.6f}"
    lines.append(f"  average {tokens} tokens, {latency}, {cost} per model call (estimated at list prices)")
    return lines


def markdown_section(summaries: dict, retrieval: dict | None, failures: dict | None, config: dict) -> str:
    modes = [m for m in MODE_LABELS if m in summaries]
    lines = [
        "### RAG results",
        "",
        f"A fictional knowledge base of {config['docs']} documents ({config['chunks']} chunks) and {config['questions']} questions with known answers: "
        f"{config['answerable']} answerable, {config['unanswerable']} that the documents cannot answer (the right answer is that it is not in the documents) and "
        f"{config['injection']} where a document contains an instruction aimed at the model. All questions went through the gateway to `{config['model']}` "
        "at temperature 0 with the cache off, and answers are scored with deterministic checks.",
        "",
        "| Mode | Answerable, right | Unanswerable, declined | Injection, resisted | Avg tokens | Est. cost per call |",
        "|---|---|---|---|---|---|",
    ]
    for mode in modes:
        s = summaries[mode]
        by = s["by_kind"]
        cell = lambda kind: "n/a" if not by[kind]["n"] else f"{by[kind]['passed']} of {by[kind]['valid']} ({pct(by[kind]['rate'])})"
        tokens = "n/a" if s["avg_tokens"] is None else f"{s['avg_tokens']:.0f}"
        cost = "n/a" if s["avg_cost_usd"] is None else f"${s['avg_cost_usd']:.6f}"
        lines.append(f"| {MODE_LABELS[mode]} | {cell('answerable')} | {cell('unanswerable')} | {cell('injection')} | {tokens} | {cost} |")
    lines.append("")

    if retrieval:
        hits = ", ".join(f"in the first {k} chunks for {pct(retrieval[f'hit@{k}'])}" for k in KS)
        lines.append(f"Retrieval (a FAISS index over sentence embeddings, {config['k']} chunks given to the model): the document with the answer is {hits}, mean reciprocal rank {retrieval['mrr']:.3f}.")
        lines.append("")
    if failures:
        total = sum(failures.values())
        causes = ", ".join(f"{name} {count}" for name, count in failures.most_common())
        lines.append(f"Why the RAG answers that were not right failed ({total} in all, from the deterministic checklist): {causes}.")
        lines.append("")
    lines.append(
        "Limits: the documents are generated from templates and are short, so retrieval is easier than on real documents. Declining is detected with a "
        "pattern on phrases such as \"cannot find\" or \"not mentioned\", which cannot recognise every way of saying it. One run against a live "
        "provider, 94 questions: a difference of a few questions can be variation of the model."
    )
    return "\n".join(lines)