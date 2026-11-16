"""Failure forensics: why was this answer bad?

A deterministic checklist: no model, no invented probabilities. Every finding names its
evidence (numbers and fields from the run and the trace). When nothing explains the failure,
the conclusion says so and lists what was and was not checked.
"""

from collections import Counter
from dataclasses import dataclass, field

from app.evals.cases import EvalCase
from app.evals.runner import CaseResult
from app.evals.scoring import score_case

LATENCY_FACTOR = 2.0
LATENCY_MIN_EXTRA_MS = 1000
TOKEN_FACTOR = 3.0
TOKEN_MIN_EXTRA = 100


@dataclass(frozen=True)
class Finding:
    kind: str
    role: str  # cause (likely contributed), context (unusual, not conclusive) or info
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Observation:
    """What is known about one answer: the run result, the run configuration, the trace row."""

    result: CaseResult
    config: dict
    trace: dict | None = None


@dataclass
class Diagnosis:
    case_id: str
    failed: bool
    findings: list = field(default_factory=list)
    checked: list = field(default_factory=list)
    unchecked: list = field(default_factory=list)

    @property
    def causes(self) -> list:
        return [f for f in self.findings if f.role == "cause"]

    def conclusion(self) -> str:
        if not self.failed:
            return "the answer passed every check"
        if self.causes:
            return "likely cause: " + "; ".join(f.message for f in self.causes)
        text = "no deterministic cause found"
        if self.checked:
            text += f" among the checked factors ({', '.join(self.checked)})"
        if self.unchecked:
            text += f". Not checked: {', '.join(self.unchecked)}"
        return text + ". Random variation of the model or a change at the provider cannot be ruled out."


def _refused(answer: str) -> bool:
    from app.cache.service import looks_like_refusal  # imported here to keep this module light

    return looks_like_refusal(answer)


def _prompt_label(config: dict) -> str:
    name = config.get("prompt")
    return f"{name}:{config.get('version')}" if name else "none"


def diagnose(
    case: EvalCase,
    candidate: Observation,
    baseline: Observation | None = None,
    baseline_latency_p95: float | None = None,
) -> Diagnosis:
    score = score_case(case, candidate.result)
    if score.valid and score.passed:
        return Diagnosis(case.id, failed=False)

    result, trace = candidate.result, candidate.trace
    base = baseline.result if baseline else None
    base_passed = None
    if baseline is not None:
        base_score = score_case(case, base)
        base_passed = base_score.valid and base_score.passed

    findings: list = []
    checked: list = []
    unchecked: list = []

    def add(kind, role, message, **evidence):
        findings.append(Finding(kind, role, message, evidence))

    # The answer itself
    if not score.valid:
        add("no_answer", "cause", f"no usable answer was recorded ({result.error or result.status})",
            status=result.status, error=result.error)
    else:
        failed = [c for c in score.checks if not c.passed]
        add(
            "failed_checks",
            "info",
            "failed checks: " + "; ".join(f"{c.type} ({c.detail})" if c.detail else c.type for c in failed),
            checks=[{"type": c.type, "detail": c.detail} for c in failed],
        )
        checked += ["empty answer", "refusal"]
        answer = result.answer or ""
        if not answer.strip():
            add("empty_answer", "cause", "the model returned an empty answer")
        elif _refused(answer):
            add("refusal", "cause", "the model refused or declined to answer", answer=answer[:120])

    # Who answered
    checked.append("fallback")
    served = result.served_model or (trace or {}).get("served_model")
    fallback_used = bool(result.fallback_used or (trace and trace.get("fallback_used")))
    if fallback_used:
        add("fallback", "cause",
            f"the answer came from a fallback model ({served}), not the requested {candidate.config.get('model')}",
            served_model=served, requested_model=candidate.config.get("model"))

    checked.append("retries")
    if result.attempts and result.attempts > 1:
        add("retries", "context", f"the request needed {result.attempts} attempts (rate limits or provider errors)",
            attempts=result.attempts)

    # The gateway's own record
    routing_cause = False
    if trace is None:
        unchecked += ["cache", "routing", "gateway errors (no trace row found)"]
    else:
        checked += ["cache", "routing", "gateway errors"]
        if trace.get("cache_status") == "hit":
            add("cache", "cause",
                f"the answer was served from the semantic cache (similarity {trace.get('cache_similarity')})",
                similarity=trace.get("cache_similarity"), entry=trace.get("cache_entry_id"))
        if trace.get("status") == "error":
            add("gateway_error", "cause",
                f"the gateway recorded an error (http {trace.get('http_status')}): {trace.get('error')}",
                http_status=trace.get("http_status"), error=trace.get("error"))
        rule = trace.get("policy_rule")
        if rule and rule != "explicit":
            routed = trace.get("routed_model")
            base_served = base.served_model if base else None
            differs = bool(base and routed and base_served and routed != base_served)
            routing_cause = bool(base_passed and differs)
            message = f"routed to {routed} by '{rule}'"
            if differs:
                message += f" (the baseline was served by {base_served})"
            add("routing", "cause" if routing_cause else "context", message,
                rule=rule, routed_model=routed, ml_status=trace.get("ml_status"),
                ml_pick=trace.get("ml_pick"), ml_prob=trace.get("ml_prob"),
                ml_similarity=trace.get("ml_similarity"))

    # Compared with a baseline
    if baseline is None:
        unchecked.append("model, prompt version, latency and length compared with a baseline")
    else:
        checked += ["model", "prompt version", "latency", "length"]
        role = "cause" if base_passed else "context"
        cand_model, base_model = result.served_model, base.served_model
        model_differs = bool(
            candidate.config.get("model") != baseline.config.get("model")
            or (cand_model and base_model and cand_model != base_model)
        )
        if model_differs:
            add("model", role,
                f"served by {cand_model or candidate.config.get('model')}, "
                f"the baseline was served by {base_model or baseline.config.get('model')}",
                candidate=cand_model, baseline=base_model)

        prompt_differs = (candidate.config.get("prompt"), candidate.config.get("version")) != (
            baseline.config.get("prompt"), baseline.config.get("version")
        )
        if prompt_differs:
            add("prompt_version", role,
                f"prompt {_prompt_label(candidate.config)} instead of {_prompt_label(baseline.config)}",
                candidate=_prompt_label(candidate.config), baseline=_prompt_label(baseline.config))

        slow, base_slow = result.latency_ms, base.latency_ms
        if slow and base_slow:
            vs_baseline = slow > base_slow * LATENCY_FACTOR and slow - base_slow >= LATENCY_MIN_EXTRA_MS
            vs_p95 = (
                baseline_latency_p95 is not None
                and slow > baseline_latency_p95 * 1.5
                and slow - baseline_latency_p95 >= LATENCY_MIN_EXTRA_MS
            )
            if vs_baseline or vs_p95:
                add("latency", "context", f"unusually slow: {slow} ms (baseline {base_slow} ms)",
                    latency_ms=slow, baseline_ms=base_slow, baseline_p95_ms=baseline_latency_p95)

        tokens, base_tokens = result.completion_tokens, base.completion_tokens
        if tokens is not None and base_tokens is not None:
            if tokens >= base_tokens * TOKEN_FACTOR and tokens - base_tokens >= TOKEN_MIN_EXTRA:
                add("length", "context", f"the answer is much longer: {tokens} tokens (baseline {base_tokens})",
                    tokens=tokens, baseline_tokens=base_tokens)
            elif base_tokens >= tokens * TOKEN_FACTOR and base_tokens - tokens >= TOKEN_MIN_EXTRA:
                add("length", "context", f"the answer is much shorter: {tokens} tokens (baseline {base_tokens})",
                    tokens=tokens, baseline_tokens=base_tokens)

        if base_passed and not (model_differs or prompt_differs or routing_cause or fallback_used):
            add("same_setup", "info",
                "same model, prompt version and routing as the baseline, which passed: "
                "no configuration difference explains the failure")

    return Diagnosis(case.id, True, findings, checked, unchecked)


def cause_counts(diagnoses: list) -> Counter:
    """How often each kind of cause was found. A failed case without any is counted as 'no_cause_found'."""
    counts: Counter = Counter()
    for d in diagnoses:
        if not d.failed:
            continue
        kinds = {f.kind for f in d.causes}
        if not kinds:
            counts["no_cause_found"] += 1
        for kind in kinds:
            counts[kind] += 1
    return counts