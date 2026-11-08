"""Compare prompt versions from real traffic (the A/B numbers).

Usage (from ai-gateway/):  python -m scripts.prompt_report --hours 24

Quality is not measured here (that is phase 4). Small samples can differ by chance.
"""

import argparse
import asyncio

from sqlalchemy import text

from app.db.session import engine

QUERY = """
select prompt_name, prompt_version,
       count(*) as requests,
       count(*) filter (where status = 'error') as errors,
       count(*) filter (where cache_status = 'hit') as hits,
       count(*) filter (where cache_status in ('hit', 'miss', 'blocked')) as consulted,
       round(avg(latency_ms) filter (where status = 'success' and cache_status is distinct from 'hit')) as avg_ms,
       round(avg(total_tokens) filter (where status = 'success' and cache_status is distinct from 'hit')) as avg_tokens,
       coalesce(sum(estimated_cost_usd), 0) as cost
from traces
where prompt_name is not null
  and created_at >= now() - make_interval(hours => :hours)
group by prompt_name, prompt_version
order by prompt_name, prompt_version
"""


async def fetch(hours: int):
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(QUERY), {"hours": hours})
            return [dict(row._mapping) for row in result]
    finally:
        await engine.dispose()


def num(value) -> str:
    return "n/a" if value is None else str(int(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    rows = asyncio.run(fetch(args.hours))
    if not rows:
        print(f"no prompt traffic in the last {args.hours} hours")
        return

    print(f"Last {args.hours} hours (avg ms and avg tokens count provider calls only, not cache hits)")
    print(f"{'prompt':<20}{'ver':>4}{'requests':>9}{'errors':>7}{'cache hit':>10}{'avg ms':>8}{'avg tok':>8}{'est cost':>14}")
    for r in rows:
        hit_rate = f"{r['hits'] / r['consulted'] * 100:.0f}%" if r["consulted"] else "n/a"
        print(
            f"{r['prompt_name']:<20}{r['prompt_version']:>4}{r['requests']:>9}{r['errors']:>7}"
            f"{hit_rate:>10}{num(r['avg_ms']):>8}{num(r['avg_tokens']):>8}{float(r['cost']):>14.8f}"
        )


if __name__ == "__main__":
    main()