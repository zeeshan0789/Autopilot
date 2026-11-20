from collections import Counter

import pytest

from app.sqlbench.compare import compare_results, normalize_rows
from app.sqlbench.database import build_database
from app.sqlbench.questions import generate_questions, load_questions, write_questions
from app.sqlbench.safety import run_query


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("shop") / "shop.db")


@pytest.fixture(scope="module")
def questions(db):
    return generate_questions(db)


def test_the_questions_have_the_expected_mix(questions):
    assert len(questions) == 67
    assert Counter(q.kind for q in questions) == {"answerable": 57, "unsupported": 5, "unsafe": 5}
    assert Counter(q.level for q in questions if q.kind == "answerable") == {1: 20, 2: 20, 3: 17}


def test_ids_and_questions_are_unique(questions):
    assert len({q.id for q in questions}) == len(questions)
    assert len({q.question for q in questions}) == len(questions)


def test_every_gold_query_runs_and_returns_something(db, questions):
    for q in questions:
        if q.kind != "answerable":
            assert q.gold_sql is None
            continue
        result = run_query(db, q.gold_sql, max_rows=1000)
        assert result.ok and result.rows, (q.id, q.question, result.error)
        if len(result.rows) == 1 and len(result.rows[0]) == 1:
            assert result.rows[0][0] not in (None, 0), q.id


def test_an_ordered_question_has_an_order_in_its_gold_query(questions):
    for q in questions:
        if q.ordered:
            assert "ORDER BY" in q.gold_sql, q.id
    assert any(q.ordered for q in questions)


def test_a_top_three_question_returns_three_rows(db, questions):
    top = [q for q in questions if "top 3 products" in q.question]
    assert len(top) == 3
    for q in top:
        assert len(run_query(db, q.gold_sql).rows) == 3


def test_questions_that_must_not_get_a_query_have_level_zero(questions):
    for q in questions:
        if q.kind != "answerable":
            assert q.level == 0 and q.gold_sql is None


def test_generation_is_deterministic(db, questions):
    assert generate_questions(db) == questions


def test_questions_survive_writing_and_loading(questions, tmp_path):
    write_questions(tmp_path / "q.jsonl", questions)
    assert load_questions(tmp_path / "q.jsonl") == questions


# ---------- Comparing results ----------


def test_the_same_rows_in_another_order_match_unless_the_order_matters():
    gold = [("a", 1), ("b", 2)]
    shuffled = [("b", 2), ("a", 1)]
    assert compare_results(gold, shuffled).match
    result = compare_results(gold, shuffled, ordered=True)
    assert not result.match and result.reason == "values"


def test_numbers_are_compared_to_two_decimals_and_ints_equal_floats():
    assert compare_results([(10.3333,)], [(10.33,)]).match
    assert compare_results([(3,)], [(3.0,)]).match
    assert not compare_results([(10.3333,)], [(10.4,)]).match


def test_the_kind_of_mismatch_is_named():
    assert compare_results([(1, 2)], [(1,)]).reason == "column_count"
    assert compare_results([(1,), (2,)], [(1,)]).reason == "row_count"
    assert compare_results([(1,)], [(2,)]).reason == "values"
    assert compare_results([(1,)], []).reason == "row_count"


def test_repeated_rows_count_as_repeated():
    assert not compare_results([("a",), ("a",)], [("a",), ("b",)]).match
    assert compare_results([("a",), ("a",)], [("a",), ("a",)]).match


def test_strings_ignore_surrounding_spaces_but_not_case():
    assert normalize_rows([(" Lisbon ",)]) == [("Lisbon",)]
    assert not compare_results([("Lisbon",)], [("lisbon",)]).match

    