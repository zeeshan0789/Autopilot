import pytest

from app.evals.checks import numbers_in, run_check, validate_check


def check(kind, answer, **fields):
    return run_check({"type": kind, **fields}, answer).passed


def test_contains_family_ignores_case():
    assert check("contains", "The capital is CANBERRA.", value="Canberra")
    assert not check("contains", "Sydney", value="Canberra")
    assert check("contains_any", "It is six", values=["6", "six"])
    assert not check("contains_any", "seven", values=["6", "six"])
    assert check("contains_all", "red, blue and green", values=["red", "blue", "green"])
    assert not check("contains_all", "red and blue", values=["red", "blue", "green"])
    assert check("not_contains", "no secret here", value="PINEAPPLE")
    assert not check("not_contains", "the pineapple code", value="PINEAPPLE")


def test_regex_flags():
    assert check("regex", "Na", pattern=r"\bNa\b")
    assert not check("regex", "na", pattern=r"\bNa\b")
    assert check("regex", "na", pattern=r"\bNa\b", flags="i")
    assert check("not_regex", "all fine", pattern=r"[0-9]")
    assert not check("not_regex", "room 7", pattern=r"[0-9]")


def test_exact_and_edges_ignore_decoration():
    assert check("exact", "Ali.", value="ali")
    assert check("exact", "**Ali**", value="Ali")
    assert not check("exact", "Ali Khan", value="Ali")
    assert check("starts_with", "**Yes**, because it follows", value="yes")
    assert not check("starts_with", "No, it does not", value="yes")
    assert check("ends_with", "Fine day, matey ARRR!", value="ARRR")
    assert not check("ends_with", "ARRR and more", value="ARRR")


def test_length_checks():
    assert check("max_words", "one two three", value=3)
    assert not check("max_words", "one two three four", value=3)
    assert check("min_words", "one two three", value=3)
    assert not check("min_words", "one two", value=3)
    assert check("max_chars", "  abc  ", value=3)
    assert check("min_chars", "abcd", value=4)


def test_json_checks_accept_a_code_fence():
    assert check("is_json", '{"a": 1}')
    assert check("is_json", '```json\n{"a": 1}\n```')
    assert not check("is_json", 'Here it is: {"a": 1}')
    assert check("json_keys", '{"name": "x", "age": 3}', keys=["name", "age"])
    assert not check("json_keys", '{"name": "x"}', keys=["name", "age"])
    assert not check("json_keys", "[1, 2]", keys=["name"])
    assert check("json_equals", "[1, 2, 3]", value=[1, 2, 3])
    assert not check("json_equals", "[1, 2]", value=[1, 2, 3])


def test_bullet_count():
    assert check("bullet_count", "- a\n- b\n- c", value=3)
    assert check("bullet_count", "* a\n* b", value=2)
    assert check("bullet_count", "1. a\n2) b\n3. c", value=3)
    assert check("bullet_count", "\u2022 only one", value=1)
    assert not check("bullet_count", "- a\n- b", value=3)
    assert not check("bullet_count", "-- not a bullet\ntext", value=1)


def test_sentence_count():
    assert check("sentence_count", "Cats purr. Cats sleep.", value=2)
    assert check("sentence_count", "One sentence only.", value=1)
    assert check("sentence_count", "Hello", value=1)
    assert not check("sentence_count", "One. Two. Three.", value=2)


def test_numbers_are_read_with_commas_decimals_and_signs():
    assert numbers_in("The total is 1,250.50.") == [1250.5]
    assert numbers_in("5 - 3 = 2") == [5.0, 3.0, 2.0]
    assert numbers_in("x = -4") == [-4.0]
    assert numbers_in("no digits") == []


def test_numeric_answer_uses_the_last_number_by_default():
    assert check("numeric_answer", "391", value=391)
    assert check("numeric_answer", "17 times 23 is 391", value=391)
    assert not check("numeric_answer", "391 is the result of 17 times 23", value=391)
    assert check("numeric_answer", "391 is the result of 17 times 23", value=391, position="any")
    assert check("numeric_answer", "1,250.50", value=1250.5)
    assert check("numeric_answer", "0.3334", value=0.3333, tolerance=0.001)
    assert not check("numeric_answer", "no number", value=1)


def test_refusal_checks():
    refusal = "I\u2019m sorry, but I can\u2019t help with that."
    assert check("refuses", refusal)
    assert not check("no_refusal", refusal)
    assert check("no_refusal", "Paris is the capital of France.")
    assert not check("refuses", "Paris is the capital of France.")


def test_a_failed_check_explains_itself():
    result = run_check({"type": "contains", "value": "Canberra"}, "Sydney")
    assert not result.passed and "Canberra" in result.detail


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"type": "nonsense"},
        {"type": "contains"},
        {"type": "regex", "pattern": "("},
        {"type": "regex", "pattern": "a", "flags": "z"},
        {"type": "max_words", "value": -1},
        {"type": "max_words", "value": True},
        {"type": "contains_any", "values": []},
        {"type": "json_keys", "keys": "name"},
        {"type": "numeric_answer", "value": "5"},
        {"type": "numeric_answer", "value": 5, "position": "first"},
    ],
)
def test_invalid_checks_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_check(bad)


def test_valid_checks_pass_validation():
    for good in (
        {"type": "contains", "value": "x"},
        {"type": "regex", "pattern": "a+", "flags": "im"},
        {"type": "is_json"},
        {"type": "numeric_answer", "value": 2.5, "tolerance": 0.1, "position": "any"},
        {"type": "no_refusal"},
    ):
        validate_check(good)