"""Collects the answers of one model on many cases, saving every usable answer at once.

A run can be stopped at any time (or by a free tier limit) and continued later: finished
cases are in the outcomes file and are not asked again.
"""

import asyncio
from dataclasses import dataclass

import httpx

from app.evals.runner import RunConfig, run_case
from app.mlrouter.data import append_outcome


@dataclass
class CollectSummary:
    saved: int = 0
    failed: int = 0
    stopped_early: bool = False
    reason: str | None = None


async def collect_outcomes(
    cases,
    config: RunConfig,
    api_key: str,
    outcomes_path,
    *,
    client=None,
    sleep=asyncio.sleep,
    progress=None,
    max_consecutive_failures: int = 8,
) -> CollectSummary:
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        )

    summary = CollectSummary()
    streak = 0
    try:
        for index, case in enumerate(cases, 1):
            result = await run_case(client, case, config, sleep)
            if result.status == "ok":
                append_outcome(outcomes_path, config.model, result)
                summary.saved += 1
                streak = 0
            else:
                summary.failed += 1
                streak += 1
            if progress:
                progress(index, len(cases), case, result)

            if streak >= max_consecutive_failures:
                summary.stopped_early = True
                summary.reason = (
                    f"{streak} failures in a row (last: {result.error}). Probably a rate limit or a "
                    "daily quota: run the same command later, finished cases are kept."
                )
                break
            if index < len(cases) and config.delay_seconds > 0:
                await sleep(config.delay_seconds)
    finally:
        if owns_client:
            await client.aclose()
    return summary