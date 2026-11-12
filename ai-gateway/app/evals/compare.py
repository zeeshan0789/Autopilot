"""Did a change make answers worse? A statistical comparison of two eval runs.

The cases are paired (same case in both runs). The difference of each case is resampled
(bootstrap) to get a 95 percent interval for the average change. A drop only blocks a
change when the whole interval is below zero, so a lucky or unlucky sample on a small
eval set does not decide by itself. The size of the smallest change this sample can
detect is reported: "no regression found" never means "no regression".
"""

import math
import random
from dataclasses import dataclass, field
from statistics import stdev

from app.evals.scoring import RunScore

JUDGE_WEIGHT = 0.3  # share of the final score that comes from a trusted judge

REGRESSION = "REGRESSION"
IMPROVEMENT = "IMPROVEMENT"
INCONCLUSIVE_DROP = "INCONCLUSIVE_DROP"
OK = "OK"
NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"


def final_scores(run_score: RunScore, verdicts: dict | None = None, judge_weight: float = JUDGE_WEIGHT) -> dict:
    """case id -> final score. Checks only, unless a trusted judge verdict is given for the case."""
    scores = {}
    for case in run_score.valid:
        verdict = (verdicts or {}).get(case.id)
        if verdict is None:
            scores[case.id] = case.score
        else:
            judged = 1.0 if verdict == "good" else 0.0
            scores[case.id] = (1 - judge_weight) * case.score + judge_weight * judged
    return scores


def paired_bootstrap(diffs: list[float], iterations: int = 5000, seed: int = 1) -> tuple[float, float]:
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(iterations))
    return means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]


@dataclass
class Comparison:
    verdict: str
    message: str
    n_paired: int
    excluded: int
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    mean_diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    detectable: float | None = None
    better: int = 0
    worse: int = 0
    same: int = 0
    categories: dict = field(default_factory=dict)  # name -> (cases, mean difference)
    worst: list = field(default_factory=list)  # (case id, baseline score, candidate score)


def exit_code(verdict: str) -> int:
    return {REGRESSION: 1, NOT_ENOUGH_DATA: 2}.get(verdict, 0)


def compare_scores(
    baseline: dict,
    candidate: dict,
    categories: dict,
    *,
    min_drop: float = 0.02,
    iterations: int = 5000,
    seed: int = 1,
    min_cases: int = 20,
) -> Comparison:
    ids = sorted(set(baseline) & set(candidate))
    excluded = len(set(baseline) | set(candidate)) - len(ids)
    n = len(ids)
    if n < min_cases:
        return Comparison(
            NOT_ENOUGH_DATA,
            f"only {n} cases are scored in both runs, at least {min_cases} are needed for a comparison",
            n,
            excluded,
        )

    diffs = [candidate[i] - baseline[i] for i in ids]
    mean_diff = sum(diffs) / n
    low, high = paired_bootstrap(diffs, iterations, seed)
    detectable = 1.96 * stdev(diffs) / math.sqrt(n) if n > 1 else 0.0

    if high < 0 and mean_diff <= -min_drop:
        verdict, message = REGRESSION, "the answers got worse, and the drop is larger than chance explains"
    elif low > 0 and mean_diff >= min_drop:
        verdict, message = IMPROVEMENT, "the answers got better, and the gain is larger than chance explains"
    elif mean_diff <= -min_drop:
        verdict, message = INCONCLUSIVE_DROP, "the average dropped but the interval includes zero: possibly worse, this sample cannot tell"
    else:
        verdict, message = OK, "no meaningful change found"

    by_category: dict = {}
    for i, d in zip(ids, diffs):
        by_category.setdefault(categories.get(i, "?"), []).append(d)

    return Comparison(
        verdict=verdict,
        message=message,
        n_paired=n,
        excluded=excluded,
        baseline_mean=sum(baseline[i] for i in ids) / n,
        candidate_mean=sum(candidate[i] for i in ids) / n,
        mean_diff=mean_diff,
        ci_low=low,
        ci_high=high,
        detectable=detectable,
        better=sum(1 for d in diffs if d > 1e-9),
        worse=sum(1 for d in diffs if d < -1e-9),
        same=sum(1 for d in diffs if abs(d) <= 1e-9),
        categories={name: (len(v), sum(v) / len(v)) for name, v in sorted(by_category.items())},
        worst=sorted(((i, baseline[i], candidate[i]) for i, d in zip(ids, diffs) if d < -1e-9), key=lambda t: t[2] - t[1])[:5],
    )


def comparison_lines(c: Comparison) -> list[str]:
    lines = [f"{c.verdict}: {c.message}"]
    if c.mean_diff is None:
        return lines
    lines += [
        f"  {c.n_paired} cases compared ({c.excluded} left out because one run has no usable answer)",
        f"  average score {c.baseline_mean * 100:.1f}% -> {c.candidate_mean * 100:.1f}% "
        f"(change {c.mean_diff * 100:+.1f} points, 95% interval {c.ci_low * 100:+.1f} to {c.ci_high * 100:+.1f})",
        f"  cases better {c.better}, worse {c.worse}, same {c.same}",
        f"  this sample can only detect average drops of about {c.detectable * 100:.1f} points or more",
        "  by category (change in points):",
    ]
    for name, (count, diff) in c.categories.items():
        lines.append(f"    {name:<20}{count:>4} cases {diff * 100:+7.1f}")
    if c.worst:
        lines.append("  biggest drops:")
        for case_id, before, after in c.worst:
            lines.append(f"    {case_id:<22}{before * 100:5.0f}% -> {after * 100:5.0f}%")
    return lines