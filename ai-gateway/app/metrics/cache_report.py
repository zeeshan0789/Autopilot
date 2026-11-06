"""Cache numbers from real traffic (the traces table).

The false hit rate cannot be measured from traffic: traces do not store prompts and
nobody labels every answer. It is measured on the labeled sample (scripts/tune_cache.py).
Costs are estimated at list prices, not real spend.
"""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from statistics import median


@dataclass
class CacheSummary:
    total: int = 0
    hits: int = 0
    misses: int = 0
    blocked: int = 0
    bypass: int = 0
    bypass_reasons: dict = field(default_factory=dict)
    tokens_saved: int = 0
    cost_saved_usd: Decimal = Decimal(0)
    cost_spent_usd: Decimal = Decimal(0)
    unpriced_hits: int = 0
    median_ms_hit: float | None = None
    median_ms_not_hit: float | None = None

    @property
    def consulted(self) -> int:
        """Requests where the cache was actually looked at."""
        return self.hits + self.misses + self.blocked

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.consulted if self.consulted else None

    @property
    def saving_pct(self) -> float | None:
        """Estimated cost avoided compared with the same traffic without a cache."""
        without_cache = self.cost_saved_usd + self.cost_spent_usd
        return float(self.cost_saved_usd / without_cache) if without_cache > 0 else None


def summarize(rows) -> CacheSummary:
    """rows: mappings with cache_status, cache_reason, saved_tokens, saved_cost_usd,
    estimated_cost_usd, latency_ms and status. Rows without a cache_status are ignored."""
    summary = CacheSummary()
    reasons: Counter = Counter()
    hit_ms: list = []
    not_hit_ms: list = []

    for row in rows:
        status = row["cache_status"]
        if status is None:
            continue
        summary.total += 1
        summary.cost_spent_usd += row["estimated_cost_usd"] or Decimal(0)
        succeeded = row["status"] == "success" and row["latency_ms"] is not None

        if status == "hit":
            summary.hits += 1
            summary.tokens_saved += row["saved_tokens"] or 0
            if row["saved_cost_usd"] is None:
                summary.unpriced_hits += 1
            else:
                summary.cost_saved_usd += row["saved_cost_usd"]
            if succeeded:
                hit_ms.append(row["latency_ms"])
        elif status in ("miss", "blocked"):
            if status == "miss":
                summary.misses += 1
            else:
                summary.blocked += 1
            if succeeded:
                not_hit_ms.append(row["latency_ms"])
        elif status == "bypass":
            summary.bypass += 1
            reasons[row["cache_reason"] or "unknown"] += 1

    summary.bypass_reasons = dict(reasons)
    summary.median_ms_hit = median(hit_ms) if hit_ms else None
    summary.median_ms_not_hit = median(not_hit_ms) if not_hit_ms else None
    return summary