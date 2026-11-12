import pytest

from app.evals.compare import (
    IMPROVEMENT,
    INCONCLUSIVE_DROP,
    NOT_ENOUGH_DATA,
    OK,
    REGRESSION,
    compare_scores,
    comparison_lines,
    exit_code,
    final_scores,
    paired_bootstrap,
)
from app.evals.scoring import CaseScore, RunScore


def scores(values):
    return {f"c{i:02d}": v for i, v in enumerate(values)}


def categories(n):
    return {f"c{i:02d}": ("math" if i % 2 else "factual") for i in range(n)}


def compare(baseline, candidate, **kwargs):
    return compare_scores(scores(baseline), scores(candidate), categories(len(baseline)), **kwargs)


def test_a_constant_change_has_a_point_interval():
    assert paired_bootstrap([-0.25] * 30) == (-0.25, -0.25)
    assert paired_bootstrap([0.0] * 30) == (0.0, 0.0)


def test_the_bootstrap_is_deterministic_for_a_seed():
    diffs = [-1.0, 0.0, 0.5, 0.0, -0.5, 1.0, 0.0, 0.0]
    assert paired_bootstrap(diffs, 1000, seed=2) == paired_bootstrap(diffs, 1000, seed=2)


def test_a_large_clear_drop_is_a_regression():
    result = compare([1.0] * 60, [1.0] * 30 + [0.5] * 30)
    assert result.verdict == REGRESSION
    assert result.mean_diff == pytest.approx(-0.25)
    assert result.ci_high < 0
    assert (result.better, result.worse, result.same) == (0, 30, 30)
    assert exit_code(result.verdict) == 1


def test_no_change_is_ok():
    result = compare([0.8] * 60, [0.8] * 60)
    assert result.verdict == OK
    assert exit_code(result.verdict) == 0


def test_a_clear_gain_is_an_improvement_and_does_not_block():
    result = compare([0.5] * 60, [1.0] * 30 + [0.5] * 30)
    assert result.verdict == IMPROVEMENT
    assert exit_code(result.verdict) == 0


def test_a_noisy_small_drop_is_inconclusive_not_a_regression():
    # 10 cases got much worse, 10 got better, 40 are unchanged: average -0.05 but within noise
    baseline = [1.0] * 10 + [0.3] * 10 + [0.5] * 40
    candidate = [0.0] * 10 + [1.0] * 10 + [0.5] * 40
    result = compare(baseline, candidate)
    assert result.verdict == INCONCLUSIVE_DROP
    assert result.mean_diff == pytest.approx(-0.05)
    assert result.ci_low < 0 < result.ci_high
    assert exit_code(result.verdict) == 0


def test_a_tiny_drop_below_the_minimum_is_ignored():
    result = compare([1.0] * 60, [1.0] * 59 + [0.0], min_drop=0.05)  # about 1.7 points
    assert result.verdict == OK


def test_too_few_cases_is_not_enough_data():
    result = compare([1.0] * 10, [0.0] * 10)
    assert result.verdict == NOT_ENOUGH_DATA
    assert exit_code(result.verdict) == 2
    assert result.mean_diff is None


def test_only_cases_scored_in_both_runs_are_compared():
    baseline = scores([1.0] * 40)
    candidate = {k: v for k, v in scores([1.0] * 40).items() if k not in ("c00", "c01")}
    result = compare_scores(baseline, candidate, categories(40))
    assert (result.n_paired, result.excluded) == (38, 2)


def test_the_detectable_drop_is_reported_and_depends_on_noise():
    quiet = compare([0.5] * 60, [0.5] * 60)
    noisy = compare([1.0] * 30 + [0.0] * 30, [0.0] * 30 + [1.0] * 30)
    assert quiet.detectable == 0.0
    assert noisy.detectable > 0.2


def test_categories_and_worst_cases_are_listed():
    candidate = [1.0] * 60
    candidate[0] = 0.0  # c00 is a "factual" case that got much worse
    result = compare([1.0] * 60, candidate)
    assert result.categories["factual"][0] == 30
    assert result.categories["factual"][1] < 0 and result.categories["math"][1] == 0
    assert result.worst[0][0] == "c00"


def test_the_report_names_the_verdict_and_the_limit_of_the_test():
    text = "\n".join(comparison_lines(compare([1.0] * 60, [1.0] * 30 + [0.5] * 30)))
    assert text.startswith("REGRESSION")
    assert "95% interval" in text
    assert "can only detect" in text
    assert "NOT_ENOUGH_DATA" in "\n".join(comparison_lines(compare([1.0] * 3, [1.0] * 3)))


def case_score(case_id, score):
    return CaseScore(id=case_id, category="x", valid=True, score=score)


def run_score(*cases):
    return RunScore(name="r", cases=list(cases), results={})


def test_checks_alone_make_the_final_score():
    run = run_score(case_score("a", 0.5), case_score("b", 1.0))
    assert final_scores(run) == {"a": 0.5, "b": 1.0}


def test_a_judge_verdict_adds_its_share():
    run = run_score(case_score("a", 1.0), case_score("b", 1.0), case_score("c", 0.5))
    result = final_scores(run, {"a": "good", "b": "bad"}, judge_weight=0.3)
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(0.7)
    assert result["c"] == pytest.approx(0.5)  # no verdict, checks only


def test_invalid_cases_get_no_score():
    run = run_score(case_score("a", 1.0), CaseScore(id="x", category="x", valid=False, score=None))
    assert set(final_scores(run)) == {"a"}