import pytest

from app.cache.guard import (
    check_guard,
    decide_hit,
    extract_entities,
    extract_negations,
    extract_numbers,
    is_time_sensitive,
)


def blocked(new_text: str, cached_text: str, reason: str) -> bool:
    result = check_guard(new_text, cached_text)
    return (not result.ok) and result.reason == reason


def allowed(new_text: str, cached_text: str) -> bool:
    return check_guard(new_text, cached_text).ok


# ---------- Numbers ----------


def test_different_year_is_blocked():
    assert blocked(
        "What was Apple revenue in 2025?",
        "What was Apple revenue in 2024?",
        "number_mismatch",
    )


def test_different_amount_is_blocked():
    assert blocked("Convert 200 USD to PKR", "Convert 100 USD to PKR", "number_mismatch")


def test_digit_inside_a_word_counts():
    assert blocked("Revenue in Q4", "Revenue in Q3", "number_mismatch")


def test_number_words_are_compared():
    assert blocked("Top ten movies", "Top five movies", "number_mismatch")


def test_number_word_equals_digit():
    assert allowed("List 5 fruits", "List five fruits")


def test_thousands_comma_is_ignored():
    assert allowed("Convert 1,000 USD to PKR", "Convert 1000 USD to PKR")


def test_versions_are_not_treated_as_decimals():
    assert blocked("Install Python 3.1", "Install Python 3.10", "number_mismatch")
    assert blocked("Install Python 3.12", "Install Python 3.11", "number_mismatch")


def test_number_order_matters():
    assert blocked("Is 10 greater than 5?", "Is 5 greater than 10?", "number_mismatch")


def test_same_numbers_pass():
    assert allowed(
        "How many days are in 2 weeks?", "How many days are there in 2 weeks?"
    )


def test_extract_numbers_in_order():
    assert extract_numbers("From 1,000 to 2.5 and three more in 2024.") == [
        "1000",
        "2.5",
        "3",
        "2024",
    ]


def test_number_words_inside_other_words_are_ignored():
    # "someone", "stone", "money" contain number words but are not numbers
    assert extract_numbers("someone has money and a stone") == []


# ---------- Named entities ----------


def test_different_country_is_blocked():
    assert blocked(
        "What is the capital of Germany?",
        "What is the capital of France?",
        "entity_mismatch",
    )


def test_paraphrase_with_same_entity_passes():
    assert allowed(
        "Which city is the capital of France?", "What is the capital of France?"
    )


def test_entity_at_sentence_start_is_detected():
    assert blocked("London weather forecast", "Paris weather forecast", "entity_mismatch")


def test_possessive_matches_plain_name():
    assert allowed("What is France's capital?", "What is the capital of France?")


def test_acronyms_are_entities():
    assert blocked("Convert 100 USD to EUR", "Convert 100 USD to PKR", "entity_mismatch")


def test_capitals_inside_a_word_are_entities():
    assert blocked("Is the Pixel good?", "Is the iPhone good?", "entity_mismatch")


def test_pronoun_i_is_not_an_entity():
    assert allowed("How do I reset my password?", "How can I reset my password?")


def test_question_words_are_not_entities():
    assert extract_entities("What is the capital of France?") == {"france"}


def test_known_limitation_lowercase_names_are_not_detected():
    """Documents a blind spot: the guard cannot see names typed in lowercase.

    The similarity threshold and the measured false hit rate are the defense here.
    """
    assert allowed("google revenue", "apple revenue")


# ---------- Negations ----------


def test_negation_is_blocked():
    assert blocked(
        "Is it not safe to eat raw eggs?",
        "Is it safe to eat raw eggs?",
        "negation_mismatch",
    )


def test_contraction_negation_is_blocked():
    assert blocked("Why doesn't it work?", "Why does it work?", "negation_mismatch")


def test_curly_apostrophe_negation_is_blocked():
    assert blocked("Why doesn\u2019t it work?", "Why does it work?", "negation_mismatch")


def test_same_negation_passes():
    assert allowed("Why isn't it working?", "Why isn't it working?")


def test_extract_negations_normalizes_contractions():
    assert extract_negations("Don\u2019t and can't, never") == {"dont", "cant", "never"}


# ---------- Time sensitive prompts ----------


@pytest.mark.parametrize(
    "text",
    [
        "What is the weather today?",
        "Who is the current CEO of Netflix?",
        "What is the latest iPhone?",
        "How are sales this year?",
        "What happened last week?",
        "Tell me what is happening right now",
        "Latest news about AI",
    ],
)
def test_time_sensitive_prompts_are_detected(text):
    assert is_time_sensitive(text)


@pytest.mark.parametrize(
    "text",
    [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Knowing is half the battle",  # contains "now" inside a word
        "An unknown snowy mountain",
        "Explain the history of the Roman Empire",
    ],
)
def test_timeless_prompts_are_not_flagged(text):
    assert not is_time_sensitive(text)


# ---------- The final decision ----------


def test_below_threshold_is_a_miss():
    decision = decide_hit(0.50, "same text", "same text", threshold=0.90)
    assert decision.hit is False
    assert decision.reason == "below_threshold"


def test_threshold_is_inclusive():
    decision = decide_hit(0.90, "same text", "same text", threshold=0.90)
    assert decision.hit is True
    assert decision.reason == "hit"


def test_high_similarity_but_guard_blocks():
    """The trap pair: embeddings are close, answers are not."""
    decision = decide_hit(
        0.97,
        "What was Apple revenue in 2025?",
        "What was Apple revenue in 2024?",
        threshold=0.90,
    )
    assert decision.hit is False
    assert decision.reason == "number_mismatch"
    assert decision.similarity == 0.97


def test_high_similarity_and_guard_passes_is_a_hit():
    decision = decide_hit(
        0.93,
        "Which city is the capital of France?",
        "What is the capital of France?",
        threshold=0.90,
    )
    assert decision.hit is True