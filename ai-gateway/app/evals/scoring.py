"""Turns an eval run plus its cases into scores."""

from dataclasses import dataclass, field

from app.evals.cases import EvalCase
from app.evals.checks import CheckResult, run_checks
from app.evals.runner import CaseResult, EvalRun


@dataclass
class CaseScore:
    id: str
    category: str
    valid: bool  # False when there is no usable answer (excluded from the averages)
    score: float | None  # share of passed checks, 0 to 1
    checks: list = field(default_factory=list)
    reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.valid and self.score == 1.0


def score_case(case: EvalCase, result: CaseResult) -> CaseScore:
    if result.status != "ok":
        reason = result.error or result.status
        return CaseScore(case.id, case.category, False, None, [], reason)
    checks: list[CheckResult] = run_checks(case.checks, result.answer or "")
    score = sum(1 for c in checks if c.passed) / len(checks)
    return CaseScore(case.id, case.category, True, score, checks)


@dataclass
class RunScore:
    name: str
    cases: list
    results: dict  # case id -> CaseResult

    @property
    def valid(self) -> list:
        return [c for c in self.cases if c.valid]

    @property
    def invalid_count(self) -> int:
        return len(self.cases) - len(self.valid)

    @property
    def mean_score(self) -> float | None:
        valid = self.valid
        return sum(c.score for c in valid) / len(valid) if valid else None

    @property
    def pass_rate(self) -> float | None:
        """Share of cases where every check passed."""
        valid = self.valid
        return sum(1 for c in valid if c.passed) / len(valid) if valid else None

    def by_category(self) -> dict:
        groups: dict = {}
        for c in self.valid:
            groups.setdefault(c.category, []).append(c)
        return {
            name: (len(items), sum(c.score for c in items) / len(items), sum(1 for c in items if c.passed) / len(items))
            for name, items in sorted(groups.items())
        }

    def _mean_of(self, attribute: str) -> float | None:
        values = [
            getattr(self.results[c.id], attribute)
            for c in self.valid
            if getattr(self.results[c.id], attribute) is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def avg_latency_ms(self) -> float | None:
        return self._mean_of("latency_ms")

    @property
    def avg_tokens(self) -> float | None:
        return self._mean_of("total_tokens")


def score_run(cases: list[EvalCase], run: EvalRun) -> RunScore:
    by_id = {c.id: c for c in cases}
    scored = []
    results = {}
    for result in run.results:
        case = by_id.get(result.id)
        if case is None:
            raise ValueError(f"the run has a result for '{result.id}' which is not in the cases file")
        scored.append(score_case(case, result))
        results[result.id] = result
    return RunScore(run.name, scored, results)


def summary_lines(score: RunScore) -> list[str]:
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    lines = [
        f"run '{score.name}': {len(score.cases)} cases, {len(score.valid)} scored, {score.invalid_count} invalid",
        f"  mean score {pct(score.mean_score)}, cases with every check passed {pct(score.pass_rate)}",
    ]
    if score.avg_latency_ms is not None:
        tokens = "n/a" if score.avg_tokens is None else f"{score.avg_tokens:.0f}"
        lines.append(f"  average latency {score.avg_latency_ms:.0f} ms, average tokens {tokens}")
    lines.append(f"  {'category':<14}{'cases':>6}{'mean':>9}{'all passed':>12}")
    for name, (count, mean, passed) in score.by_category().items():
        lines.append(f"  {name:<14}{count:>6}{pct(mean):>9}{pct(passed):>12}")
    for c in score.cases:
        if not c.valid:
            lines.append(f"  invalid {c.id}: {c.reason}")
    return lines