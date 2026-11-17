"""Per request results of a benchmark run, their summaries and the report."""

from collections import Counter
from dataclasses import dataclass

from app.evals.compare import paired_bootstrap
from app.metrics.cache_eval import wilson_interval

LABELS = {
    "direct": "Direct to the provider",
    "autopilot": "Autopilot",
    "autopilot_no_guard": "Autopilot, similarity only (guard off)",
    "autopilot_strict": "Autopilot, strict word guard",
}


@dataclass
class ItemResult:
    item_id: str
    kind: str  # unique, repeat or paraphrase
    base_id: str
    in_outage: bool
    status: str  # ok or error
    passed: bool  # answered and every check passed
    latency_ms: float
    cost_usd: float = 0.0
    cache_status: str | None = None
    fallback_used: bool = False
    served_model: str | None = None
    error: str | None = None
    answer: str | None = None


def percentile(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(round(q * (len(ordered) - 1)))]


@dataclass
class ModeSummary:
    name: str
    requests: int
    answered: int
    errors: int
    errors_in_outage: int
    errors_outside_outage: int
    correct: int
    correct_rate: float | None
    correct_interval: tuple
    correct_outside_outage: float | None
    correct_in_outage: float | None
    cache_hits: int
    wrong_hits: int  # answers served from the cache that fail their checks
    fallbacks: int
    fallbacks_in_outage: int
    latency_p50: float | None
    latency_p95: float | None
    total_cost: float
    avg_cost: float | None


def _rate(items, predicate):
    return sum(1 for i in items if predicate(i)) / len(items) if items else None


def summarize_mode(name: str, items: list[ItemResult]) -> ModeSummary:
    outage = [i for i in items if i.in_outage]
    outside = [i for i in items if not i.in_outage]
    answered = [i for i in items if i.status == "ok"]
    latencies = [i.latency_ms for i in answered]
    correct = sum(1 for i in items if i.passed)
    total_cost = sum(i.cost_usd for i in items)
    return ModeSummary(
        name=name,
        requests=len(items),
        answered=len(answered),
        errors=len(items) - len(answered),
        errors_in_outage=sum(1 for i in outage if i.status != "ok"),
        errors_outside_outage=sum(1 for i in outside if i.status != "ok"),
        correct=correct,
        correct_rate=correct / len(items) if items else None,
        correct_interval=wilson_interval(correct, len(items)),
        correct_outside_outage=_rate(outside, lambda i: i.passed),
        correct_in_outage=_rate(outage, lambda i: i.passed),
        cache_hits=sum(1 for i in items if i.cache_status == "hit"),
        wrong_hits=sum(1 for i in items if i.cache_status == "hit" and not i.passed),
        fallbacks=sum(1 for i in items if i.fallback_used),
        fallbacks_in_outage=sum(1 for i in outage if i.fallback_used),
        latency_p50=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        total_cost=total_cost,
        avg_cost=total_cost / len(items) if items else None,
    )


def paired_difference(baseline: list[ItemResult], other: list[ItemResult], iterations: int = 2000, outage: bool | None = None) -> dict:
    """Correct answers of `other` minus `baseline`, request by request (the same request in both).

    outage=True looks only at the requests during the outage, False only at those outside it.
    """
    if [i.item_id for i in baseline] != [i.item_id for i in other]:
        raise ValueError("the two runs are not about the same requests")
    pairs = [(b, o) for b, o in zip(baseline, other) if outage is None or b.in_outage == outage]
    if not pairs:
        raise ValueError("no requests in this part of the workload")
    diffs = [float(int(o.passed) - int(b.passed)) for b, o in pairs]
    low, high = paired_bootstrap(diffs, iterations)
    return {"mean": sum(diffs) / len(diffs), "low": low, "high": high, "n": len(diffs)}


# ---------- Wrong answers from the cache ----------


@dataclass(frozen=True)
class WrongHit:
    item_id: str
    kind: str  # replay, cross_prompt or unknown
    prompt: str
    source_id: str | None
    source_prompt: str | None
    answer: str | None


def classify_wrong_hits(items: list[ItemResult], prompts: dict) -> list[WrongHit]:
    """Why did a cache hit fail its checks?

    replay: the cache repeated an earlier answer to the same prompt (that answer was wrong already).
    cross_prompt: the cache gave the answer that was originally written for another prompt (a false hit).
    unknown: no earlier answer with the same text was found.
    prompts maps an item id to the text of its last user message.
    """
    earlier: dict = {}  # answer text -> the earlier answers (not from the cache) with this text
    wrong = []
    for item in items:
        if item.status != "ok":
            continue
        if item.cache_status == "hit":
            if item.passed:
                continue
            sources = earlier.get(item.answer or "", [])
            same = next((s for s in sources if s.base_id == item.base_id), None)
            source = same or (sources[0] if sources else None)
            kind = "replay" if same else ("cross_prompt" if source else "unknown")
            wrong.append(
                WrongHit(
                    item.item_id,
                    kind,
                    prompts.get(item.item_id, ""),
                    source.item_id if source else None,
                    prompts.get(source.item_id) if source else None,
                    item.answer,
                )
            )
        elif item.answer is not None:
            earlier.setdefault(item.answer, []).append(item)
    return wrong


# ---------- Report ----------


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(value) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


def _usd(value) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def summary_lines(summary: ModeSummary) -> list[str]:
    low, high = summary.correct_interval
    return [
        f"{LABELS.get(summary.name, summary.name)}",
        f"  answered {summary.answered} of {summary.requests} (errors: {summary.errors_in_outage} during the outage, {summary.errors_outside_outage} outside it)",
        f"  correct (every check passed) {summary.correct} of {summary.requests} = {_pct(summary.correct_rate)} (95% interval {_pct(low)} to {_pct(high)}); "
        f"outside the outage {_pct(summary.correct_outside_outage)}, during it {_pct(summary.correct_in_outage)}",
        f"  cache hits {summary.cache_hits}, wrong answers served from the cache {summary.wrong_hits}, fallback answers {summary.fallbacks} ({summary.fallbacks_in_outage} during the outage)",
        f"  latency p50 {_ms(summary.latency_p50)}, p95 {_ms(summary.latency_p95)}; estimated cost {_usd(summary.total_cost)} ({_usd(summary.avg_cost)} per request)",
    ]


def _diff_text(diff: dict) -> str:
    return f"{diff['mean'] * 100:+.1f} points (95% interval {diff['low'] * 100:+.1f} to {diff['high'] * 100:+.1f}, paired by request)"


def markdown_section(summaries: dict, diffs: dict, config: dict, part_diffs: dict | None = None, wrong_kinds: dict | None = None) -> str:
    names = [n for n in LABELS if n in summaries]
    direct = summaries.get("direct")
    lines = [
        "### Benchmark results",
        "",
        f"A fixed workload of {config['requests']} requests ({config['kinds']}) built from the generated cases (12 task families), all asking "
        f"`{config['model']}` at temperature 0. A simulated outage of the primary provider covers requests {config['outage_start'] + 1} to {config['outage_end']}. "
        "Quality is measured with the deterministic checks of every case, and an error counts as a failed request. Cost is estimated at list prices. "
        "The gateway ran in the same process as the benchmark (the provider calls are real).",
        "",
        "| | " + " | ".join(LABELS[n] for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def row(label, getter):
        lines.append(f"| {label} | " + " | ".join(getter(summaries[n]) for n in names) + " |")

    row("Requests answered", lambda s: f"{s.answered} of {s.requests}")
    row("Correct answers", lambda s: f"{s.correct} ({_pct(s.correct_rate)})")
    row("Correct outside the outage", lambda s: _pct(s.correct_outside_outage))
    row("Correct during the outage", lambda s: _pct(s.correct_in_outage))
    row("Cache hits", lambda s: str(s.cache_hits))
    row("Wrong answers served from the cache", lambda s: str(s.wrong_hits))
    if wrong_kinds:
        row("of which repeats of an earlier wrong answer", lambda s: str(wrong_kinds.get(s.name, Counter()).get("replay", 0)))
        row("of which the answer written for another prompt", lambda s: str(wrong_kinds.get(s.name, Counter()).get("cross_prompt", 0)))
    row("Answers from a fallback model", lambda s: str(s.fallbacks))
    row("Latency p50 / p95", lambda s: f"{_ms(s.latency_p50)} / {_ms(s.latency_p95)}")
    row("Estimated cost", lambda s: _usd(s.total_cost))
    if direct and direct.total_cost > 0:
        row("Cost against direct", lambda s: f"{(s.total_cost / direct.total_cost - 1) * 100:+.0f}%")
    lines.append("")

    for name, diff in diffs.items():
        lines.append(f"- {LABELS[name]} against direct, correct answers: {_diff_text(diff)}.")
        for key, label in (("normal", "outside the outage"), ("outage", "during the outage")):
            part = (part_diffs or {}).get(name, {}).get(key)
            if part:
                lines.append(f"  - {label}: {_diff_text(part)}, {part['n']} requests.")
    lines += [
        "",
        "Limits: the workload is synthetic, the rewording is mechanical (a prefix or a suffix), so the share of repeated and reworded "
        "prompts is a choice of this benchmark and says nothing about real traffic. The outage is simulated. This is one run per mode "
        "against a live provider, so a difference of a few requests can be model variation. The local fallback model is much weaker "
        "than the primary one, and its price is a placeholder.",
    ]
    return "\n".join(lines)