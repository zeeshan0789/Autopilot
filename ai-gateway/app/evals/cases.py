"""Eval cases: a prompt (messages) with checks that decide whether the answer is good."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.evals.checks import validate_check

ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    messages: list
    checks: list
    reference: str | None = None  # a correct answer, shown to the judge later
    notes: str | None = None


def load_cases(path) -> list[EvalCase]:
    cases = []
    seen = set()
    with open(Path(path), encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                case = _build(row)
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"line {number}: {e}") from e
            if case.id in seen:
                raise ValueError(f"line {number}: duplicate id '{case.id}'")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("no cases found")
    return cases


def _build(row: dict) -> EvalCase:
    case_id = row["id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("id must be a non empty string")
    if not isinstance(row.get("category"), str) or not row["category"]:
        raise ValueError(f"case '{case_id}': category must be a non empty string")

    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"case '{case_id}': messages must be a non empty list")
    for message in messages:
        if message.get("role") not in ROLES or not isinstance(message.get("content"), str):
            raise ValueError(f"case '{case_id}': every message needs a valid role and a content text")
    if not any(m["role"] == "user" for m in messages):
        raise ValueError(f"case '{case_id}': needs at least one user message")

    checks = row["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"case '{case_id}': needs at least one check")
    for check in checks:
        try:
            validate_check(check)
        except ValueError as e:
            raise ValueError(f"case '{case_id}': {e}") from e

    return EvalCase(
        id=case_id,
        category=row["category"],
        messages=messages,
        checks=checks,
        reference=row.get("reference"),
        notes=row.get("notes"),
    )