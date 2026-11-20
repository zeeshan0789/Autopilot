"""Text for the console and for the README."""

from collections import Counter

from app.core.cost import estimate_cost
from app.metrics.cache_eval import wilson_interval
from app.sqlbench.forensics import diagnose_sql

LEVELS = {1: "simple", 2: "joins", 3: "complex"}


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _group(items, passed_of=lambda i: i.correct) -> dict:
    passed = sum(1 for i in items if passed_of(i))
    return {"n": len(items), "passed": passed, "rate": passed / len(items) if items else None,
            "interval": wilson_interval(passed, len(items))}


def summarize(items) -> dict:
    ok = [i for i in items if i.status == "ok"]
    answerable = [i for i in ok if i.kind == "answerable"]
    costs = [
        float(estimate_cost(i.served_model, i.prompt_tokens, i.completion_tokens) or 0)
        for i in ok
        if i.served_model and i.prompt_tokens is not None and i.completion_tokens is not None
    ]
    return {
        "questions": len(items),
        "invalid": len(items) - len(ok),
        "levels": {level: _group([i for i in answerable if i.level == level]) for level in LEVELS},
        "answerable": _group(answerable),
        "unsupported": _group([i for i in ok if i.kind == "unsupported"]),
        "unsafe": _group([i for i in ok if i.kind == "unsafe"]),
        "executed": sum(1 for i in answerable if i.exec_kind == "ok") / len(answerable) if answerable else None,
        "write_attempts": sum(1 for i in ok if i.kind == "unsafe" and i.sql is not None),
        "blocked": sum(1 for i in ok if i.exec_kind == "blocked"),
        "avg_calls": _mean([i.calls for i in ok]),
        "avg_tokens": _mean([i.prompt_tokens + i.completion_tokens for i in ok if i.prompt_tokens is not None and i.completion_tokens is not None]),
        "avg_cost_usd": _mean(costs),
    }


def failure_counts(items) -> Counter:
    counts: Counter = Counter()
    for item in items:
        finding = diagnose_sql(item)
        if finding is not None:
            counts[finding.cause] += 1
    return counts


def _cell(group) -> str:
    return "n/a" if not group["n"] else f"{group['passed']} of {group['n']} ({pct(group['rate'])})"


def summary_lines(name: str, s: dict) -> list[str]:
    lines = [f"{name}: {s['questions']} questions, {s['invalid']} without a usable reply"]
    for level, label in LEVELS.items():
        group = s["levels"][level]
        low, high = group["interval"]
        lines.append(f"  {label:<8} {_cell(group):<18} (95% interval {pct(low)} to {pct(high)})")
    lines.append(f"  all answerable {_cell(s['answerable'])}; queries that ran {pct(s['executed'])}")
    lines.append(f"  not answerable from the schema, declined: {_cell(s['unsupported'])}")
    lines.append(f"  requests to change data, declined: {_cell(s['unsafe'])} (the model wrote a statement {s['write_attempts']} times, the safety layer blocked {s['blocked']} queries)")
    tokens = "n/a" if s["avg_tokens"] is None else f"{s['avg_tokens']:.0f}"
    cost = "n/a" if s["avg_cost_usd"] is None else f"${s['avg_cost_usd']:.6f}"
    lines.append(f"  average {s['avg_calls']:.2f} model calls, {tokens} tokens and {cost} per question (estimated at list prices)")
    return lines


def markdown_section(summaries: dict, failures: dict, info: dict) -> str:
    lines = [
        "### Text to SQL results",
        "",
        f"A generated online shop database ({info['rows']}) and {info['questions']} questions: {info['answerable']} answerable "
        f"({info['simple']} on one table, {info['joins']} with joins, {info['complex']} complex), {info['unsupported']} that the schema cannot answer and "
        f"{info['unsafe']} that ask to change data (the right reply to those is CANNOT_ANSWER). A query is right when it returns the same result as the "
        "gold query (columns, rows and values, the order only when the question asks for one). Every request went through the gateway at temperature 0 with the cache off. "
        "Model queries run only through a safety layer: a SELECT or WITH query, one statement, a read only connection, an authorizer that allows reading the four tables, "
        "a 2 second limit and a 200 row limit.",
        "",
        "| Run | Simple | Joins | Complex | All answerable | Unsupported declined | Change requests declined | Model calls per question | Est. cost per question |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        cost = "n/a" if s["avg_cost_usd"] is None else f"${s['avg_cost_usd']:.6f}"
        lines.append(
            f"| {name} | {_cell(s['levels'][1])} | {_cell(s['levels'][2])} | {_cell(s['levels'][3])} | {_cell(s['answerable'])} | "
            f"{_cell(s['unsupported'])} | {_cell(s['unsafe'])} | {s['avg_calls']:.2f} | {cost} |"
        )
    lines.append("")
    for name, counts in failures.items():
        total = sum(counts.values())
        causes = ", ".join(f"{cause} {count}" for cause, count in counts.most_common()) or "none"
        lines.append(f"- `{name}`: {total} questions without a right answer ({causes}).")
    lines += [
        "",
        "Limits: the database and the questions are generated from templates, 67 questions is a small sample (a difference of a few questions can be model "
        "variation), and the comparison is strict (a query that returns an extra column is wrong). The safety layer is what protects the database, not the "
        "model's good behaviour: the numbers above for change requests count what the model wrote, not what ran.",
    ]
    return "\n".join(lines)