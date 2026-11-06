"""Offline evaluation of the semantic cache on a labeled sample of prompt pairs.

A pair is (new prompt, cached prompt, reusable). reusable=True means the answer cached
for the cached prompt is also a correct answer for the new prompt. A cache hit on a
non reusable pair is a false hit: the caller would get a wrong answer.

The numbers are only as good as the labels and the sample size. With few pairs the
confidence intervals are wide, and that is reported rather than hidden.
"""

import json
import math
from dataclasses import dataclass
from statistics import median

import numpy as np

from app.cache.guard import check_guard


@dataclass(frozen=True)
class LabeledPair:
    new_prompt: str
    cached_prompt: str
    reusable: bool
    category: str


@dataclass(frozen=True)
class ScoredPair:
    pair: LabeledPair
    similarity: float
    guard_ok: bool
    guard_reason: str


def load_pairs(path) -> list[LabeledPair]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                pair = LabeledPair(
                    new_prompt=row["new_prompt"],
                    cached_prompt=row["cached_prompt"],
                    reusable=row["reusable"],
                    category=row["category"],
                )
            except KeyError as e:
                raise ValueError(f"line {line_number}: missing field {e}") from e
            if not isinstance(pair.reusable, bool):
                raise ValueError(f"line {line_number}: reusable must be true or false")
            if not pair.new_prompt.strip() or not pair.cached_prompt.strip():
                raise ValueError(f"line {line_number}: prompts must not be empty")
            pairs.append(pair)
    if not pairs:
        raise ValueError("no pairs found")
    return pairs


def score_pairs(pairs: list[LabeledPair], embedder) -> list[ScoredPair]:
    """Embedding similarity and the guard verdict for every pair.

    The embedder must return normalized vectors, so the dot product is the cosine
    similarity. Every distinct text is embedded once.
    """
    texts = sorted({t.strip() for p in pairs for t in (p.new_prompt, p.cached_prompt)})
    vectors = {text: embedder.encode(text) for text in texts}
    scored = []
    for p in pairs:
        similarity = float(
            np.dot(vectors[p.new_prompt.strip()], vectors[p.cached_prompt.strip()])
        )
        guard = check_guard(p.new_prompt.strip(), p.cached_prompt.strip())
        scored.append(ScoredPair(p, similarity, guard.ok, guard.reason))
    return scored


@dataclass(frozen=True)
class Outcome:
    threshold: float
    use_guard: bool
    hits: int
    false_hits: int
    reusable_total: int
    reusable_served: int

    @property
    def false_hit_rate(self) -> float | None:
        """Share of the answers served from the cache that were wrong."""
        return self.false_hits / self.hits if self.hits else None

    @property
    def recall(self) -> float | None:
        """Share of the reusable pairs that the cache actually served."""
        return self.reusable_served / self.reusable_total if self.reusable_total else None


def is_hit(scored: ScoredPair, threshold: float, use_guard: bool = True) -> bool:
    return scored.similarity >= threshold and (scored.guard_ok or not use_guard)


def evaluate(scored: list[ScoredPair], threshold: float, use_guard: bool = True) -> Outcome:
    hits = false_hits = reusable_total = reusable_served = 0
    for s in scored:
        hit = is_hit(s, threshold, use_guard)
        if s.pair.reusable:
            reusable_total += 1
            if hit:
                reusable_served += 1
        if hit:
            hits += 1
            if not s.pair.reusable:
                false_hits += 1
    return Outcome(threshold, use_guard, hits, false_hits, reusable_total, reusable_served)


def default_thresholds() -> list[float]:
    return [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 up to 0.99


def sweep(scored: list[ScoredPair], thresholds, use_guard: bool = True) -> list[Outcome]:
    return [evaluate(scored, t, use_guard) for t in thresholds]


def lowest_safe_threshold(outcomes: list[Outcome], max_false_hits: int = 0) -> Outcome | None:
    """The lowest threshold in the sweep with at most max_false_hits false hits.

    Raising the threshold can only remove hits, so false hits never increase with it.
    """
    for outcome in sorted(outcomes, key=lambda o: o.threshold):
        if outcome.false_hits <= max_false_hits:
            return outcome
    return None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 percent confidence interval for a proportion (Wilson score)."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def false_hit_pairs(scored: list[ScoredPair], threshold: float, use_guard: bool = True):
    return [
        s for s in scored
        if is_hit(s, threshold, use_guard) and not s.pair.reusable
    ]


def missed_pairs(scored: list[ScoredPair], threshold: float, use_guard: bool = True):
    return [
        s for s in scored
        if s.pair.reusable and not is_hit(s, threshold, use_guard)
    ]


@dataclass(frozen=True)
class CategoryStats:
    category: str
    pairs: int
    reusable: int  # how many of the pairs are labeled reusable
    min_similarity: float
    median_similarity: float
    max_similarity: float
    guard_blocked: int


def category_stats(scored: list[ScoredPair]) -> list[CategoryStats]:
    groups: dict[str, list[ScoredPair]] = {}
    for s in scored:
        groups.setdefault(s.pair.category, []).append(s)
    stats = []
    for category, items in groups.items():
        sims = [s.similarity for s in items]
        stats.append(
            CategoryStats(
                category=category,
                pairs=len(items),
                reusable=sum(1 for s in items if s.pair.reusable),
                min_similarity=min(sims),
                median_similarity=median(sims),
                max_similarity=max(sims),
                guard_blocked=sum(1 for s in items if not s.guard_ok),
            )
        )
    return sorted(stats, key=lambda c: (-c.reusable / c.pairs, c.category))