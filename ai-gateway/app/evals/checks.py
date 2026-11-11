"""Deterministic checks on a model answer.

Each eval case lists checks (contains, JSON, number of bullets, a numeric answer...).
A case score is the share of its checks that pass. Checks cannot judge style or nuance,
that is what the LLM judge adds (and the judge is only trusted after calibration).
"""

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    type: str
    passed: bool
    detail: str = ""


REQUIRED_FIELDS = {
    "contains": ("value",),
    "contains_any": ("values",),
    "contains_all": ("values",),
    "not_contains": ("value",),
    "regex": ("pattern",),
    "not_regex": ("pattern",),
    "exact": ("value",),
    "starts_with": ("value",),
    "ends_with": ("value",),
    "max_words": ("value",),
    "min_words": ("value",),
    "max_chars": ("value",),
    "min_chars": ("value",),
    "is_json": (),
    "json_keys": ("keys",),
    "json_equals": ("value",),
    "bullet_count": ("value",),
    "sentence_count": ("value",),
    "numeric_answer": ("value",),
    "no_refusal": (),
    "refuses": (),
}

_INT_CHECKS = {
    "max_words", "min_words", "max_chars", "min_chars", "bullet_count", "sentence_count",
}
_LIST_CHECKS = {"contains_any", "contains_all", "json_keys"}
_LIST_FIELD = {"contains_any": "values", "contains_all": "values", "json_keys": "keys"}

# Characters that decorate an answer without being part of it
_EDGE_CHARS = " \t\r\n*_\"'`.!?:;,"


def _regex_flags(text: str) -> int:
    flags = 0
    for letter in text:
        flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}[letter]
    return flags


def validate_check(check) -> None:
    """Raise ValueError when a check from a dataset file is malformed."""
    if not isinstance(check, dict) or "type" not in check:
        raise ValueError("a check needs a type")
    kind = check["type"]
    if kind not in REQUIRED_FIELDS:
        raise ValueError(f"unknown check type '{kind}'")
    for field in REQUIRED_FIELDS[kind]:
        if field not in check:
            raise ValueError(f"check '{kind}' needs '{field}'")

    if kind in ("regex", "not_regex"):
        try:
            re.compile(check["pattern"], _regex_flags(check.get("flags", "")))
        except (re.error, KeyError) as e:
            raise ValueError(f"check '{kind}' has a bad pattern or flags: {e}") from e
    if kind in _LIST_CHECKS:
        items = check[_LIST_FIELD[kind]]
        if not isinstance(items, list) or not items or not all(isinstance(i, str) and i for i in items):
            raise ValueError(f"check '{kind}' needs a non empty list of strings")
    if kind in _INT_CHECKS:
        value = check["value"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"check '{kind}' needs a non negative integer value")
    if kind == "numeric_answer":
        value = check["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("check 'numeric_answer' needs a number")
        if check.get("position", "last") not in ("last", "any"):
            raise ValueError("numeric_answer position must be 'last' or 'any'")


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER.finditer(text):
        try:
            numbers.append(float(match.group(0).replace(",", "")))
        except ValueError:
            pass
    return numbers


_FENCE = re.compile(r"^\s*```[A-Za-z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def parse_json_answer(answer: str):
    """(True, value) when the whole answer is JSON, optionally inside a code fence."""
    text = answer.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return True, json.loads(text)
    except ValueError:
        return False, None


_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_END.split(text.strip()) if s.strip()])


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(_EDGE_CHARS)).lower()


def _count_check(measure, label):
    def check(c, answer):
        actual = measure(answer)
        ok = actual == c["value"]
        return ok, "" if ok else f"{label} is {actual}, expected {c['value']}"

    return check


def _limit_check(measure, label, upper):
    def check(c, answer):
        actual = measure(answer)
        ok = actual <= c["value"] if upper else actual >= c["value"]
        word = "at most" if upper else "at least"
        return ok, "" if ok else f"{label} is {actual}, expected {word} {c['value']}"

    return check


def _contains(c, answer):
    ok = c["value"].lower() in answer.lower()
    return ok, "" if ok else f"missing {c['value']!r}"


def _contains_any(c, answer):
    lowered = answer.lower()
    ok = any(v.lower() in lowered for v in c["values"])
    return ok, "" if ok else f"none of {c['values']} found"


def _contains_all(c, answer):
    lowered = answer.lower()
    missing = [v for v in c["values"] if v.lower() not in lowered]
    return not missing, "" if not missing else f"missing {missing}"


def _not_contains(c, answer):
    ok = c["value"].lower() not in answer.lower()
    return ok, "" if ok else f"contains the forbidden text {c['value']!r}"


def _regex(c, answer):
    ok = re.search(c["pattern"], answer, _regex_flags(c.get("flags", ""))) is not None
    return ok, "" if ok else f"pattern {c['pattern']!r} not found"


def _not_regex(c, answer):
    ok = re.search(c["pattern"], answer, _regex_flags(c.get("flags", ""))) is None
    return ok, "" if ok else f"forbidden pattern {c['pattern']!r} found"


def _exact(c, answer):
    ok = _norm(answer) == _norm(c["value"])
    return ok, "" if ok else f"expected {c['value']!r}"


def _starts_with(c, answer):
    ok = answer.lstrip(_EDGE_CHARS).lower().startswith(c["value"].lower())
    return ok, "" if ok else f"does not start with {c['value']!r}"


def _ends_with(c, answer):
    ok = answer.rstrip(_EDGE_CHARS).lower().endswith(c["value"].lower())
    return ok, "" if ok else f"does not end with {c['value']!r}"


def _is_json(c, answer):
    ok, _ = parse_json_answer(answer)
    return ok, "" if ok else "the answer is not valid JSON"


def _json_keys(c, answer):
    ok, value = parse_json_answer(answer)
    if not ok:
        return False, "the answer is not valid JSON"
    if not isinstance(value, dict):
        return False, "the JSON is not an object"
    missing = [k for k in c["keys"] if k not in value]
    return not missing, "" if not missing else f"missing keys {missing}"


def _json_equals(c, answer):
    ok, value = parse_json_answer(answer)
    if not ok:
        return False, "the answer is not valid JSON"
    return value == c["value"], "" if value == c["value"] else f"expected {c['value']!r}"


def _numeric(c, answer):
    numbers = numbers_in(answer)
    if not numbers:
        return False, "no number in the answer"
    target = float(c["value"])
    tolerance = float(c.get("tolerance", 0.001))
    candidates = numbers if c.get("position", "last") == "any" else [numbers[-1]]
    ok = any(abs(n - target) <= tolerance for n in candidates)
    return ok, "" if ok else f"expected {target:g}, found {numbers}"


def _refusal(answer: str) -> bool:
    from app.cache.service import looks_like_refusal  # imported here to keep this module light

    return looks_like_refusal(answer)


def _no_refusal(c, answer):
    refused = _refusal(answer)
    return not refused, "the model refused" if refused else ""


def _refuses(c, answer):
    refused = _refusal(answer)
    return refused, "" if refused else "the model did not refuse"


_CHECKS = {
    "contains": _contains,
    "contains_any": _contains_any,
    "contains_all": _contains_all,
    "not_contains": _not_contains,
    "regex": _regex,
    "not_regex": _not_regex,
    "exact": _exact,
    "starts_with": _starts_with,
    "ends_with": _ends_with,
    "max_words": _limit_check(lambda a: len(a.split()), "word count", upper=True),
    "min_words": _limit_check(lambda a: len(a.split()), "word count", upper=False),
    "max_chars": _limit_check(lambda a: len(a.strip()), "length", upper=True),
    "min_chars": _limit_check(lambda a: len(a.strip()), "length", upper=False),
    "is_json": _is_json,
    "json_keys": _json_keys,
    "json_equals": _json_equals,
    "bullet_count": _count_check(lambda a: len(_BULLET.findall(a)), "bullet count"),
    "sentence_count": _count_check(_count_sentences, "sentence count"),
    "numeric_answer": _numeric,
    "no_refusal": _no_refusal,
    "refuses": _refuses,
}


def run_check(check: dict, answer: str) -> CheckResult:
    passed, detail = _CHECKS[check["type"]](check, answer)
    return CheckResult(check["type"], passed, detail)


def run_checks(checks: list[dict], answer: str) -> list[CheckResult]:
    return [run_check(check, answer) for check in checks]