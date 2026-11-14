"""Outcomes of models on eval cases: storage, splits and the numbers that show where a router can help."""

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.cost import estimate_cost
from app.evals.cases import EvalCase
from app.evals.runner import CaseResult
from app.evals.scoring import score_case

_ID = re.compile(r"^gen_(?P<family>[a-z_]+)_d(?P<level>\d)_(?P<index>\d+)$")
_FIELDS = {f.name for f in dataclasses.fields(CaseResult)}
TEST_SHARE = 25  # percent of the generated cases that are held out for testing


def parse_case_id(case_id: str) -> tuple:
    """(family, difficulty level). Hand written cases have no level."""
    match = _ID.match(case_id)
    if not match:
        return "handwritten", None
    return match["family"], int(match["level"])


def split_of(case_id: str) -> str:
    """train or test for generated cases (fixed by a hash of the id), handwritten otherwise."""
    if not case_id.startswith("gen_"):
        return "handwritten"
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if bucket < TEST_SHARE else "train"


# ---------- Storage ----------


def append_outcome(path, model: str, result: CaseResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": model, **dataclasses.asdict(result)}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_outcomes(path) -> dict:
    """(case id, model) -> CaseResult. Only usable answers are kept, a later line replaces an earlier one."""
    path = Path(path)
    outcomes: dict = {}
    if not path.exists():
        return outcomes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        result = CaseResult(**{k: v for k, v in row.items() if k in _FIELDS})
        outcomes[(result.id, row["model"])] = result
    return outcomes


def pending_cases(cases: list[EvalCase], outcomes: dict, model: str) -> list[EvalCase]:
    return [c for c in cases if (c.id, model) not in outcomes]


# ---------- Numbers ----------


def outcome_cost(model: str, result: CaseResult) -> float | None:
    """Estimated cost at list prices, None when the tokens or the price are unknown."""
    if result.prompt_tokens is None or result.completion_tokens is None:
        return None
    cost = estimate_cost(model, result.prompt_tokens, result.completion_tokens)
    return None if cost is None else float(cost)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


@dataclass
class ModelStats:
    model: str
    cases: int
    pass_rate: float | None  # share of cases where every check passed
    mean_score: float | None
    avg_latency_ms: float | None
    avg_tokens: float | None
    avg_cost_usd: float | None


def model_stats(cases: list[EvalCase], outcomes: dict, models: list[str]) -> list[ModelStats]:
    stats = []
    for model in models:
        scored, results = [], []
        for case in cases:
            result = outcomes.get((case.id, model))
            if result is not None:
                scored.append(score_case(case, result))
                results.append(result)
        stats.append(
            ModelStats(
                model=model,
                cases=len(scored),
                pass_rate=_mean([1.0 if s.passed else 0.0 for s in scored]),
                mean_score=_mean([s.score for s in scored]),
                avg_latency_ms=_mean([r.latency_ms for r in results]),
                avg_tokens=_mean([r.total_tokens for r in results]),
                avg_cost_usd=_mean([outcome_cost(model, r) for r in results]),
            )
        )
    return stats


def family_matrix(cases: list[EvalCase], outcomes: dict, models: list[str]) -> dict:
    """'family_dN' -> model -> (cases, pass rate)."""
    groups: dict = {}
    for case in cases:
        family, level = parse_case_id(case.id)
        groups.setdefault(f"{family}_d{level}" if level else family, []).append(case)
    matrix = {}
    for key, items in groups.items():
        row = {}
        for model in models:
            passed = [
                score_case(c, outcomes[(c.id, model)]).passed for c in items if (c.id, model) in outcomes
            ]
            row[model] = (len(passed), sum(passed) / len(passed)) if passed else (0, None)
        matrix[key] = row
    return matrix


@dataclass
class OracleResult:
    cases: int  # cases that every model answered
    oracle_pass_rate: float | None
    oracle_avg_cost_usd: float | None
    chosen: dict = field(default_factory=dict)  # model -> how often the oracle picks it
    strongest_pass_rate: float | None = None
    strongest_avg_cost_usd: float | None = None
    cheapest_pass_rate: float | None = None
    cheapest_avg_cost_usd: float | None = None


def oracle(cases: list[EvalCase], outcomes: dict, models: list[str]) -> OracleResult:
    """The best any router could do on these cases: the cheapest model that passes each one.

    models must be ordered from the cheapest to the strongest. A case that nobody passes counts
    as a failure answered by the strongest model.
    """
    complete = [c for c in cases if all((c.id, m) in outcomes for m in models)]
    if not complete:
        return OracleResult(0, None, None)

    chosen = {m: 0 for m in models}
    totals = {"oracle": [0, 0.0], "strongest": [0, 0.0], "cheapest": [0, 0.0]}

    def add(name, passed, cost):
        totals[name][0] += int(passed)
        totals[name][1] += cost

    for case in complete:
        passed = {m: score_case(case, outcomes[(case.id, m)]).passed for m in models}
        cost = {m: outcome_cost(m, outcomes[(case.id, m)]) or 0.0 for m in models}
        pick = next((m for m in models if passed[m]), models[-1])
        chosen[pick] += 1
        add("oracle", passed[pick], cost[pick])
        add("strongest", passed[models[-1]], cost[models[-1]])
        add("cheapest", passed[models[0]], cost[models[0]])

    n = len(complete)
    return OracleResult(
        cases=n,
        oracle_pass_rate=totals["oracle"][0] / n,
        oracle_avg_cost_usd=totals["oracle"][1] / n,
        chosen=chosen,
        strongest_pass_rate=totals["strongest"][0] / n,
        strongest_avg_cost_usd=totals["strongest"][1] / n,
        cheapest_pass_rate=totals["cheapest"][0] / n,
        cheapest_avg_cost_usd=totals["cheapest"][1] / n,
    )