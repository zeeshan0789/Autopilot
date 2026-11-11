import json

import pytest

from app.evals.cases import EvalCase
from app.evals.labels import answer_sha, append_label, collect_items, load_labels, pick_items
from app.evals.runner import CaseResult, EvalRun


def make_case(case_id):
    return EvalCase(
        id=case_id,
        category="x",
        messages=[{"role": "user", "content": case_id}],
        checks=[{"type": "contains", "value": "a"}],
    )


CASES = [make_case(f"c{i}") for i in range(10)]


def make_run(name, statuses=None):
    results = []
    for i, case in enumerate(CASES):
        status = (statuses or {}).get(case.id, "ok")
        results.append(CaseResult(id=case.id, status=status, answer=f"{name} answer {i}" if status == "ok" else None))
    return EvalRun(name=name, created_at="now", config={}, results=results)


def test_only_usable_answers_can_be_labeled():
    items = collect_items(CASES, [make_run("a", {"c1": "invalid", "c2": "error"})])
    assert len(items) == 8
    assert {i.case.id for i in items}.isdisjoint({"c1", "c2"})


def test_items_from_several_runs_are_collected():
    assert len(collect_items(CASES, [make_run("a"), make_run("b")])) == 20


def test_same_seed_same_order_and_count_is_respected():
    items = collect_items(CASES, [make_run("a")])
    first = pick_items(items, [], 4, seed=1)
    again = pick_items(items, [], 4, seed=1)
    assert [i.case.id for i in first] == [i.case.id for i in again]
    assert len(first) == 4
    assert [i.case.id for i in pick_items(items, [], 4, seed=2)] != [i.case.id for i in first]


def test_labeled_answers_are_not_shown_again(tmp_path):
    path = tmp_path / "labels.jsonl"
    items = collect_items(CASES, [make_run("a")])
    chosen = pick_items(items, [], 3, seed=1)
    for item in chosen:
        append_label(path, item, "good")

    remaining = pick_items(items, load_labels(path), 10, seed=1)
    assert len(remaining) == 7
    assert {i.case.id for i in remaining}.isdisjoint({i.case.id for i in chosen})


def test_a_changed_answer_needs_a_new_label(tmp_path):
    path = tmp_path / "labels.jsonl"
    old = collect_items(CASES, [make_run("a")])[0]
    append_label(path, old, "bad")

    rerun = make_run("a")
    rerun.results[0].answer = "a different answer"
    items = collect_items(CASES, [rerun])
    assert len(pick_items(items, load_labels(path), 10)) == 10  # the label is for the old answer


def test_labels_are_stored_with_the_answer_fingerprint(tmp_path):
    path = tmp_path / "labels.jsonl"
    item = collect_items(CASES, [make_run("a")])[0]
    append_label(path, item, "good")
    row = json.loads(path.read_text().strip())
    assert row == {"run": "a", "id": "c0", "label": "good", "answer_sha": answer_sha(item.answer)}


def test_only_good_and_bad_are_accepted(tmp_path):
    item = collect_items(CASES, [make_run("a")])[0]
    with pytest.raises(ValueError):
        append_label(tmp_path / "labels.jsonl", item, "maybe")


def test_missing_label_file_means_no_labels(tmp_path):
    assert load_labels(tmp_path / "nothing.jsonl") == []