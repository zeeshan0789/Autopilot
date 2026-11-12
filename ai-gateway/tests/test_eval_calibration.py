import pytest

from app.evals.calibration import (
    MIN_LABELS,
    Pair,
    bootstrap_kappa_interval,
    calibrate,
    cohens_kappa,
    collect_labeled,
    confusion,
    is_trusted,
    load_calibration,
    save_calibration,
)
from app.evals.cases import EvalCase
from app.evals.labels import answer_sha
from app.evals.runner import CaseResult, EvalRun


def pairs(tp, fp, fn, tn):
    return (
        [Pair(True, True)] * tp
        + [Pair(False, True)] * fp
        + [Pair(True, False)] * fn
        + [Pair(False, False)] * tn
    )


def test_confusion_counts():
    assert confusion(pairs(3, 2, 1, 4)) == (3, 2, 1, 4)


def test_kappa_values():
    assert cohens_kappa(20, 5, 5, 20) == pytest.approx(0.6)
    assert cohens_kappa(25, 0, 0, 25) == pytest.approx(1.0)
    assert cohens_kappa(10, 10, 10, 10) == pytest.approx(0.0)
    assert cohens_kappa(5, 0, 0, 0) is None  # one class only
    assert cohens_kappa(0, 0, 0, 0) is None


def test_a_good_judge_is_trusted():
    result = calibrate(pairs(23, 3, 2, 22), "judge", "v1")
    assert result.n == 50
    assert result.accuracy == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.8)
    assert result.specificity == pytest.approx(22 / 25)
    assert result.trusted and result.reasons == []


def test_a_judge_that_agrees_only_by_chance_is_not_trusted():
    result = calibrate(pairs(15, 10, 10, 15), "judge", "v1")
    assert not result.trusted
    assert any("kappa" in r for r in result.reasons)


def test_a_lenient_judge_that_accepts_everything_is_not_trusted():
    result = calibrate(pairs(25, 25, 0, 0), "judge", "v1")
    assert result.specificity == 0.0
    assert not result.trusted
    assert any("bad answers" in r for r in result.reasons)


def test_too_few_labels_or_one_sided_labels_are_not_enough():
    few = calibrate(pairs(10, 0, 0, 10), "judge", "v1")
    assert not few.trusted and any(str(MIN_LABELS) in r for r in few.reasons)

    one_sided = calibrate(pairs(40, 0, 0, 2), "judge", "v1")
    assert not one_sided.trusted and any("one class" in r for r in one_sided.reasons)


def test_the_kappa_interval_is_deterministic_and_contains_the_estimate():
    data = pairs(23, 3, 2, 22)
    first = bootstrap_kappa_interval(data, seed=3)
    assert first == bootstrap_kappa_interval(data, seed=3)
    low, high = first
    assert low < 0.8 < high
    assert low > 0.4  # 50 labels with this agreement cannot be consistent with chance


def test_the_interval_is_wider_with_fewer_labels():
    wide = bootstrap_kappa_interval(pairs(9, 1, 1, 9))
    narrow = bootstrap_kappa_interval(pairs(90, 10, 10, 90))
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_a_calibration_can_be_saved_and_loaded(tmp_path):
    result = calibrate(pairs(23, 3, 2, 22), "judge", "v1", unparsable=2)
    path = tmp_path / "calibration.json"
    save_calibration(path, result)
    loaded = load_calibration(path)
    assert loaded == result
    assert loaded.unparsable == 2
    assert load_calibration(tmp_path / "missing.json") is None


def test_a_calibration_only_counts_for_the_same_judge_and_rubric():
    trusted = calibrate(pairs(23, 3, 2, 22), "judge", "v1")
    assert is_trusted(trusted, "judge", "v1")
    assert not is_trusted(trusted, "another-judge", "v1")
    assert not is_trusted(trusted, "judge", "v2")
    assert not is_trusted(None, "judge", "v1")
    assert not is_trusted(calibrate(pairs(15, 10, 10, 15), "judge", "v1"), "judge", "v1")


# ---------- Matching labels to answers ----------


def case(case_id):
    return EvalCase(id=case_id, category="x", messages=[{"role": "user", "content": "q"}], checks=[{"type": "contains", "value": "a"}])


def run_with(name, answers, status="ok"):
    results = [CaseResult(id=i, status=status, answer=a) for i, a in answers.items()]
    return EvalRun(name=name, created_at="now", config={}, results=results)


def label(run, case_id, text, value="good"):
    return {"run": run, "id": case_id, "label": value, "answer_sha": answer_sha(text)}


def test_labels_are_matched_to_the_answers_they_were_given_for():
    run = run_with("a", {"c1": "one", "c2": "two", "c3": "three"})
    labels = [
        label("a", "c1", "one"),  # fine
        label("a", "c2", "an older answer"),  # the answer changed since: stale
        label("gone", "c3", "three"),  # run missing
        label("a", "ghost", "x"),  # case missing
    ]
    items, stale, missing = collect_labeled(labels, {"a": run}, [case("c1"), case("c2"), case("c3")])
    assert [(item[0]["id"], item[2]) for item in items] == [("c1", "one")]
    assert (stale, missing) == (1, 2)


def test_an_unusable_answer_cannot_be_matched():
    run = run_with("a", {"c1": None}, status="invalid")
    items, stale, missing = collect_labeled([label("a", "c1", "")], {"a": run}, [case("c1")])
    assert items == [] and missing == 1


def test_a_later_label_for_the_same_answer_wins():
    run = run_with("a", {"c1": "one"})
    labels = [label("a", "c1", "one", "bad"), label("a", "c1", "one", "good")]
    items, _, _ = collect_labeled(labels, {"a": run}, [case("c1")])
    assert len(items) == 1 and items[0][0]["label"] == "good"