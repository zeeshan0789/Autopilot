import pytest

from app.mlrouter.features import FEATURE_NAMES, extract_features


def value(messages, name):
    return extract_features(messages)[FEATURE_NAMES.index(name)]


def user(text):
    return [{"role": "user", "content": text}]


def test_one_number_per_name_and_names_are_unique():
    assert len(extract_features(user("hi"))) == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_basic_counts():
    messages = user("What is 12 + 30? Reply with only the final number.")
    assert value(messages, "digits") == 4
    assert value(messages, "numbers") == 2
    assert value(messages, "max_number_digits") == 2
    assert value(messages, "operators") == 1
    assert value(messages, "question_marks") == 1
    assert value(messages, "has_system") == 0
    assert value(messages, "messages") == 1
    assert value(messages, "cue_only") == 1
    assert value(messages, "cue_reply_with") == 1
    assert value(messages, "cue_json") == 0


def test_the_system_prompt_is_seen_by_the_cues_but_is_not_the_text():
    system = "Reply in all capital letters."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "Say hello."}]
    assert value(messages, "has_system") == 1
    assert value(messages, "system_chars") == len(system)
    assert value(messages, "chars") == len("Say hello.")
    assert value(messages, "total_chars") == len(system) + len("Say hello.")
    assert value(messages, "cue_all_caps") == 1


def test_the_last_user_message_is_the_text():
    messages = [
        {"role": "user", "content": "first message with many words"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert value(messages, "chars") == len("second")
    assert value(messages, "messages") == 3


def test_ratios_and_word_lengths():
    assert value(user("ABC def"), "upper_ratio") == pytest.approx(0.5)
    assert value(user("ab abcd"), "avg_word_len") == pytest.approx(3.0)
    assert value(user("ab abcd"), "longest_word") == 4


def test_cues_ignore_case_and_whole_words():
    assert value(user("Return JSON please"), "cue_json") == 1
    assert value(user("Sort these"), "cue_sort") == 1
    assert value(user("a sorted list"), "cue_sort") == 0  # "sorted" is not "sort"
    assert value(user("How MANY apples"), "cue_how_many") == 1


def test_quotes_commas_parentheses_and_lines():
    messages = user("a, b, c 'x' \"y\" (z)\nsecond line")
    assert value(messages, "commas") == 2
    assert value(messages, "quotes") == 4
    assert value(messages, "parentheses") == 1
    assert value(messages, "lines") == 2


def test_empty_text_is_safe():
    assert all(x == 0.0 for x in extract_features(user(""))[:1])
    features = extract_features([{"role": "system", "content": "only a system message"}])
    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("chars")] == 0.0