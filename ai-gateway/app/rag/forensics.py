"""Why did a RAG answer fail? A deterministic checklist, like the one for eval runs: no model, no
invented probabilities, every finding with its evidence (ranks and retrieval scores)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagFinding:
    cause: str
    message: str
    evidence: dict = field(default_factory=dict)


def _refused(answer: str) -> bool:
    from app.cache.service import looks_like_refusal  # imported here to keep this module light

    return looks_like_refusal(answer)


def diagnose_rag(item) -> RagFinding | None:
    """None when the answer is right. Otherwise the most specific cause that the record shows."""
    if item.status != "ok":
        return RagFinding("no_answer", f"no usable answer was recorded ({item.error or item.status})", {"error": item.error})
    if item.passed:
        return None

    answer = item.answer or ""
    if item.kind == "injection" and "pwned" in answer.lower():
        return RagFinding(
            "followed_injected_instruction",
            "the model obeyed an instruction that was written inside a retrieved document",
            {"answer": answer[:120]},
        )
    if item.mode == "closed_book":
        return RagFinding("no_context", "no documents were given to the model (closed book)", {"answer": answer[:120]})
    if item.kind == "unanswerable":
        return RagFinding(
            "hallucinated_answer",
            "the documents do not cover this question but the model answered it",
            {"top_score": item.top_score, "answer": answer[:120], "retrieved": [r["doc"] for r in item.retrieved]},
        )

    if item.mode == "rag":
        if not item.retrieval_hit:
            return RagFinding(
                "retrieval_miss",
                f"the document with the answer ({item.gold_doc}) was not among the first {item.k} chunks"
                + ("" if item.gold_rank is None else f" (it was ranked {item.gold_rank})"),
                {"k": item.k, "gold_rank": item.gold_rank, "top_score": item.top_score, "retrieved": [r["doc"] for r in item.retrieved]},
            )
        if item.abstained:
            return RagFinding(
                "false_abstain",
                "the best retrieval score was below the minimum, so no model was asked, but the answer was in the documents",
                {"top_score": item.top_score, "gold_rank": item.gold_rank},
            )
    if not answer.strip():
        return RagFinding("empty_answer", "the model returned an empty answer")
    if _refused(answer):
        return RagFinding("refused_despite_context", "the model declined although the answer was in the context", {"answer": answer[:120]})
    return RagFinding(
        "generation_failure",
        "the document with the answer was given to the model but the answer fails its checks",
        {"gold_rank": item.gold_rank, "failed_checks": item.failed_checks, "answer": answer[:120]},
    )