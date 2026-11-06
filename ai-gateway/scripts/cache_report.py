"""Print cache numbers from the traces table.

Usage (from ai-gateway/):  python -m scripts.cache_report --hours 48
"""

import argparse
import asyncio

from sqlalchemy import text

from app.db.session import engine
from app.metrics.cache_report import summarize

QUERY = """
select cache_status, cache_reason, saved_tokens, saved_cost_usd,
       estimated_cost_usd, latency_ms, status
from traces
where cache_status is not null
  and created_at >= now() - make_interval(hours => :hours)
"""


async def fetch(hours: int) -> list[dict]:
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(QUERY), {"hours": hours})
            return [dict(row._mapping) for row in result]
    finally:
        await engine.dispose()


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def ms(value) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24, help="how far back to look")
    args = parser.parse_args()

    s = summarize(asyncio.run(fetch(args.hours)))
    print(f"Last {args.hours} hours, {s.total} requests that reached the cache step")
    print(f"  hits {s.hits}, misses {s.misses}, blocked by the guard {s.blocked}, bypassed {s.bypass}")
    print(f"  hit rate (hits among the {s.consulted} requests the cache looked at): {pct(s.hit_rate)}")
    if s.bypass_reasons:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s.bypass_reasons.items()))
        print(f"  bypass reasons: {reasons}")
    print(f"  tokens saved: {s.tokens_saved}")
    print(f"  estimated cost saved at list prices: ${s.cost_saved_usd:.8f}")
    print(f"  estimated cost spent at list prices: ${s.cost_spent_usd:.8f}")
    print(f"  estimated saving versus no cache: {pct(s.saving_pct)}")
    print(f"  median latency: cache hit {ms(s.median_ms_hit)}, provider call {ms(s.median_ms_not_hit)}")
    if s.unpriced_hits:
        print(f"  note: {s.unpriced_hits} hits had no price, so their savings are not counted")
    print("  false hit rate: not measurable from traffic, see scripts.tune_cache")


if __name__ == "__main__":
    main()