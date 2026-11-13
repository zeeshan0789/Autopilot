import re

import pytest

from app.evals.calibration import Pair, calibrate
from app.evals.cases import EvalCase
from app.evals.compare import (
    IMPROVEMENT,
    INCONCLUSIVE_DROP,
    NOT_ENOUGH_DATA,
    OK,
    REGRESSION,
    Comparison,
)
from app.evals.seeded import (
    VARIANTS,
    SeededRow,
    build_row,
    classify,
    pick_cases,
    readme_section,
    run_name,
    summarize,
    unique_texts,
)


def test_the_variants_are_well_formed():
    names = [v.name for v in VARIANTS]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-z0-9_]+", n) for n in names)  # safe in file names
    assert all(v.system_prompt.strip() and "{{" not in v.system_prompt for v in VARIANTS)
    assert {v.kind for v in VARIANTS} == {"baseline", "baseline_repeat", "bad", "control"}
    assert VARIANTS[0].kind == "baseline"


def test_enough_bad_and_harmless_variants():
    assert sum(1 for v in VARIANTS if v.kind == "bad") >= 5
    assert sum(1 for v in VARIANTS if v.kind == "control") >= 3


def test_the_repeat_uses_the_same_text_as_the_baseline():
    by_name = {v.name: v for v in VARIANTS}
    assert by_name["baseline"].system_prompt == by_name["baseline_repeat"].system_prompt


def test_every_other_prompt_text_is_different():
    texts = unique_texts()
    assert len(texts) == len(VARIANTS) - 1  # only the repeat shares a text
    assert len(texts) == len(set(texts))


def test_run_names():
    assert run_name(VARIANTS[0]) == "seeded_baseline"


@pytest.mark.parametrize(
    "kind,verdict,expected",
    [
        ("bad", REGRESSION, "caught"),
        ("bad", INCONCLUSIVE_DROP, "warned"),
        ("bad", OK, "missed"),
        ("bad", IMPROVEMENT, "missed"),
        ("control", REGRESSION, "false_block"),
        ("control", INCONCLUSIVE_DROP, "false_warning"),
        ("control", OK, "fine"),
        ("control", IMPROVEMENT, "fine"),
        ("baseline_repeat", REGRESSION, "false_block"),
        ("baseline_repeat", OK, "fine"),
        ("bad", NOT_ENOUGH_DATA, "not_evaluated"),
        ("control", NOT_ENOUGH_DATA, "not_evaluated"),
    ],
)
def test_classify(kind, verdict, expected):
    assert classify(kind, verdict) == expected


def row(kind, verdict, name="x", diff=-0.3):
    return SeededRow(name, kind, verdict, classify(kind, verdict), 40, diff, diff - 0.1, diff + 0.1)


def test_a_row_is_built_from_a_comparison():
    comparison = Comparison(REGRESSION, "worse", 40, 2, mean_diff=-0.5, ci_low=-0.6, ci_high=-0.4)
    built = build_row(VARIANTS[2], comparison)
    assert (built.variant, built.kind, built.verdict, built.outcome) == ("one_word", "bad", REGRESSION, "caught")
    assert (built.n_paired, built.mean_diff) == (40, -0.5)


def test_summary_counts():
    rows = [
        row("bad", REGRESSION, "a"),
        row("bad", REGRESSION, "b"),
        row("bad", INCONCLUSIVE_DROP, "c"),
        row("bad", OK, "d"),
        row("bad", NOT_ENOUGH_DATA, "e"),
        row("control", OK, "f"),
        row("control", REGRESSION, "g"),
        row("baseline_repeat", INCONCLUSIVE_DROP, "h"),
    ]
    s = summarize(rows)
    assert (s["bad_total"], s["bad_caught"], s["bad_warned"], s["bad_missed"]) == (4, 2, 1, 1)
    assert (s["control_total"], s["false_blocks"], s["false_warnings"]) == (3, 1, 1)
    assert s["not_evaluated"] == 1


def test_the_readme_section_has_the_numbers_and_the_caveats():
    rows = [row("bad", REGRESSION, "one_word"), row("bad", OK, "no_digits", -0.01), row("control", OK, "polite", 0.0)]
    text = readme_section(rows, None)
    assert "**1 of 2**" in text
    assert "wrongly blocked: **0 of 1**" in text
    assert "| one_word | bad | REGRESSION |" in text
    assert "designed by me" in text
    assert "has not been calibrated" in text


def test_the_readme_section_reports_a_calibration():
    good = calibrate([Pair(True, True)] * 23 + [Pair(False, True)] * 3 + [Pair(True, False)] * 2 + [Pair(False, False)] * 22, "judge-x", "v1")
    text = readme_section([row("bad", REGRESSION)], good)
    assert "judge-x" in text and "50 answers" in text and "Cohen's kappa 0.80" in text
    assert "It is trusted" in text

    weak = calibrate([Pair(True, True)] * 25 + [Pair(False, True)] * 25, "judge-x", "v1")
    assert "NOT trusted" in readme_section([row("bad", REGRESSION)], weak)


def make_case(case_id, category):
    return EvalCase(id=case_id, category=category, messages=[{"role": "user", "content": "q"}], checks=[{"type": "contains", "value": "a"}])


def test_pick_cases_takes_the_first_of_every_category_in_file_order():
    cases = [make_case("a1", "a"), make_case("b1", "b"), make_case("a2", "a"), make_case("a3", "a"), make_case("b2", "b")]
    assert [c.id for c in pick_cases(cases, 2)] == ["a1", "b1", "a2", "b2"]
    assert [c.id for c in pick_cases(cases, 0)] == [c.id for c in cases]
    assert len(pick_cases(cases, 10)) == 5