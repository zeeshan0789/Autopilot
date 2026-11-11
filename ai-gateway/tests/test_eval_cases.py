import json

import pytest

from app.evals.cases import load_cases
from app.settings import BASE_DIR

CASES_FILE = BASE_DIR / "benchmarks" / "eval_cases.jsonl"


def write(tmp_path, *rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def good_row(case_id="a"):
    return {
        "id": case_id,
        "category": "factual",
        "messages": [{"role": "user", "content": "hi"}],
        "checks": [{"type": "contains", "value": "x"}],
    }


def test_shipped_dataset_is_valid():
    cases = load_cases(CASES_FILE)
    assert len(cases) >= 50
    categories = {c.category for c in cases}
    assert {"factual", "math", "reasoning", "instruction_format", "extraction"} <= categories


def test_a_valid_file_loads(tmp_path):
    cases = load_cases(write(tmp_path, good_row("a"), good_row("b")))
    assert [c.id for c in cases] == ["a", "b"]


@pytest.mark.parametrize(
    "change",
    [
        {"id": ""},
        {"category": ""},
        {"messages": []},
        {"messages": [{"role": "robot", "content": "hi"}]},
        {"messages": [{"role": "system", "content": "only a system message"}]},
        {"messages": [{"role": "user"}]},
        {"checks": []},
        {"checks": [{"type": "nonsense"}]},
    ],
)
def test_invalid_rows_are_rejected_with_the_line_number(tmp_path, change):
    with pytest.raises(ValueError, match="line 1"):
        load_cases(write(tmp_path, {**good_row(), **change}))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(write(tmp_path, good_row("a"), good_row("a")))


def test_bad_json_and_empty_files_are_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_cases(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        load_cases(path)