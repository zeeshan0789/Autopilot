"""Measure the semantic cache on the labeled pairs and suggest a similarity threshold.

Usage (from ai-gateway/):  python -m scripts.tune_cache
"""

import argparse

from app.cache.embedder import Embedder
from app.cache.service import CacheSettings
from app.metrics.cache_eval import (
    category_stats,
    default_thresholds,
    evaluate,
    false_hit_pairs,
    load_pairs,
    lowest_safe_threshold,
    missed_pairs,
    score_pairs,
    sweep,
    wilson_interval,
)
from app.settings import BASE_DIR, load_cache_config


def pct(value) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def describe(label: str, outcome) -> None:
    low, high = wilson_interval(outcome.false_hits, outcome.hits)
    r_low, r_high = wilson_interval(outcome.reusable_served, outcome.reusable_total)
    guard = "on" if outcome.use_guard else "off"
    print(f"{label}: threshold {outcome.threshold:.2f}, guard {guard}")
    print(
        f"    hits {outcome.hits}, false hits {outcome.false_hits}, "
        f"false hit rate {pct(outcome.false_hit_rate)} "
        f"(95% interval {pct(low)} to {pct(high)})"
    )
    print(
        f"    reusable pairs served {outcome.reusable_served} of {outcome.reusable_total}, "
        f"recall {pct(outcome.recall)} (95% interval {pct(r_low)} to {pct(r_high)})"
    )


def print_pairs(title: str, items, limit: int = 15) -> None:
    print(title)
    if not items:
        print("    none")
    for s in items[:limit]:
        print(
            f"    sim {s.similarity:.3f} [{s.pair.category}] "
            f"{s.pair.new_prompt!r} <- {s.pair.cached_prompt!r}"
        )
    if len(items) > limit:
        print(f"    ... and {len(items) - limit} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(BASE_DIR / "benchmarks" / "cache_pairs.jsonl"))
    parser.add_argument("--margin", type=float, default=0.02, help="safety margin added to the lowest safe threshold")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    reusable = sum(1 for p in pairs if p.reusable)
    print(f"{len(pairs)} labeled pairs: {reusable} reusable, {len(pairs) - reusable} not reusable")
    print("Embedding the prompts (the first run loads the model)...")
    scored = score_pairs(pairs, Embedder())
    current = CacheSettings(**load_cache_config()).similarity_threshold

    print()
    print("1. Similarity by category (the guard column counts pairs the guard blocks)")
    print(f"   {'category':<16}{'pairs':>6}{'reusable':>9}{'min':>7}{'median':>8}{'max':>7}{'guard blocks':>14}")
    for c in category_stats(scored):
        print(
            f"   {c.category:<16}{c.pairs:>6}{c.reusable:>9}{c.min_similarity:>7.3f}"
            f"{c.median_similarity:>8.3f}{c.max_similarity:>7.3f}{c.guard_blocked:>14}"
        )

    print()
    print("2. Threshold sweep (FHR = false hit rate = wrong answers among the answers served)")
    print(f"   {'thr':>5} | {'guard OFF':^21} | {'guard ON':^29}")
    print(f"   {'':>5} | {'hits':>5}{'false':>6}{'FHR':>8} | {'hits':>5}{'false':>6}{'FHR':>8}{'recall':>8}")
    for t in default_thresholds():
        if t < 0.70 or round(t * 100) % 2 != 0:
            continue
        a = evaluate(scored, t, use_guard=False)
        b = evaluate(scored, t, use_guard=True)
        print(
            f"   {t:>5.2f} | {a.hits:>5}{a.false_hits:>6}{pct(a.false_hit_rate):>8} | "
            f"{b.hits:>5}{b.false_hits:>6}{pct(b.false_hit_rate):>8}{pct(b.recall):>8}"
        )

    print()
    print("3. The threshold in config/cache.yaml right now")
    describe("   guard off", evaluate(scored, current, use_guard=False))
    describe("   guard on ", evaluate(scored, current, use_guard=True))

    print()
    print("4. Suggested threshold (lowest one with zero false hits on this sample, plus a margin)")
    safe = lowest_safe_threshold(sweep(scored, default_thresholds(), use_guard=True), 0)
    if safe is None:
        print("   No threshold up to 0.99 removes every false hit on this sample.")
        print("   Look at the false hits listed below: they need a better guard rule.")
        recommended = None
    else:
        recommended = min(1.0, round(safe.threshold + args.margin, 2))
        print(f"   lowest safe threshold {safe.threshold:.2f}, with margin {args.margin:.2f}: {recommended:.2f}")
        describe("   at the suggestion", evaluate(scored, recommended, use_guard=True))

    print()
    print_pairs(
        f"5. False hits at the current threshold {current:.2f} (guard on)",
        false_hit_pairs(scored, current, use_guard=True),
    )
    if recommended is not None:
        print()
        print_pairs(
            f"6. Reusable pairs the cache would still miss at {recommended:.2f} (guard on)",
            missed_pairs(scored, recommended, use_guard=True),
        )


if __name__ == "__main__":
    main()