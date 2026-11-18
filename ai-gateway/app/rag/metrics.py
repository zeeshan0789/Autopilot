"""Retrieval metrics, the threshold sweep and the summary of a RAG run."""

from collections import Counter

from app.core.cost import estimate_cost
from app.metrics.cache_eval import wilson_interval

KS = (1, 3, 5)
KINDS = ("answerable", "unanswerable", "injection")


def retrieval_summary(ranks: list, ks=KS) -> dict:
    """ranks: for each question the rank of the first chunk of its document (None when not found)."""
    n = len(ranks)
    summary = {"n": n, "mrr": sum(1 / r for r in ranks if r) / n if n else None}
    for k in ks:
        summary[f"hit@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else None
    return summary


def abstention_sweep(answerable_scores: list, unanswerable_scores: list, thresholds=None) -> list:
    """For every threshold: the share of answerable questions that would wrongly get "not found"
    and the share of unanswerable questions that would rightly get it (decided by the best retrieval score only)."""
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(2, 17)]
    rows = []
    for t in thresholds:
        false_abstain = sum(1 for s in answerable_scores if s < t) / len(answerable_scores) if answerable_scores else None
        correct_abstain = sum(1 for s in unanswerable_scores if s < t) / len(unanswerable_scores) if unanswerable_scores else None
        rows.append({"threshold": t, "false_abstain": false_abstain, "correct_abstain": correct_abstain})
    return rows


def recommend_min_score(sweep: list, max_false_abstain: float = 0.02):
    """The highest threshold that wrongly refuses at most max_false_abstain of the answerable questions."""
    fine = [r["threshold"] for r in sweep if r["false_abstain"] is not None and r["false_abstain"] <= max_false_abstain]
    return max(fine) if fine else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_items(items: list) -> dict:
    """Numbers of one run. Items without a usable answer (status invalid) are counted apart, not as wrong."""
    by_kind = {}
    for kind in KINDS:
        group = [i for i in items if i.kind == kind]
        valid = [i for i in group if i.status == "ok"]
        passed = sum(1 for i in valid if i.passed)
        by_kind[kind] = {
            "n": len(group),
            "valid": len(valid),
            "passed": passed,
            "rate": passed / len(valid) if valid else None,
            "interval": wilson_interval(passed, len(valid)),
        }
    called = [i for i in items if i.latency_ms is not None]
    costs = [
        float(estimate_cost(i.served_model, i.prompt_tokens, i.completion_tokens) or 0)
        for i in called
        if i.served_model and i.prompt_tokens is not None and i.completion_tokens is not None
    ]
    return {
        "by_kind": by_kind,
        "questions": len(items),
        "invalid": sum(1 for i in items if i.status != "ok"),
        "abstained": sum(1 for i in items if i.abstained),
        "llm_calls": len(called),
        "avg_tokens": _mean([i.prompt_tokens + i.completion_tokens for i in called if i.prompt_tokens is not None and i.completion_tokens is not None]),
        "avg_latency_ms": _mean([i.latency_ms for i in called]),
        "avg_cost_usd": _mean(costs),
    }


def failure_counts(items: list) -> Counter:
    from app.rag.forensics import diagnose_rag

    counts: Counter = Counter()
    for item in items:
        finding = diagnose_rag(item)
        if finding is not None:
            counts[finding.cause] += 1
    return counts