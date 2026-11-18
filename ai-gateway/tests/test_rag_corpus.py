import re
from collections import Counter

import pytest

from app.evals.checks import run_checks, validate_check
from app.rag.corpus import ABSTAIN_PATTERN, INJECTION, generate, load_docs, load_questions, write_corpus


@pytest.fixture(scope="module")
def corpus():
    return generate()


def test_the_corpus_has_the_expected_size(corpus):
    docs, questions = corpus
    assert len(docs) == 34
    assert Counter(d.id.split("_")[0] for d in docs) == {"product": 11, "office": 6, "policy": 5, "team": 6, "incident": 6}
    assert Counter(q.kind for q in questions) == {"answerable": 77, "unanswerable": 14, "injection": 3}


def test_ids_are_unique(corpus):
    docs, questions = corpus
    assert len({d.id for d in docs}) == len(docs)
    assert len({q.id for q in questions}) == len(questions)
    assert len({q.question for q in questions}) == len(questions)


def test_every_check_is_valid_and_the_reference_passes_it(corpus):
    for q in corpus[1]:
        for check in q.checks:
            validate_check(check)
        failed = [r for r in run_checks(q.checks, q.reference) if not r.passed]
        assert not failed, (q.id, q.reference, failed)


def test_the_answer_of_every_question_is_in_its_document(corpus):
    docs, questions = corpus
    by_id = {d.id: d for d in docs}
    for q in questions:
        if q.kind == "unanswerable":
            assert q.doc_id is None
            continue
        text = by_id[q.doc_id].text.lower()
        main = [c for c in q.checks if c["type"] in ("numeric_answer", "contains", "contains_any")]
        assert main, q.id
        for check in main:
            if check["type"] == "numeric_answer":
                assert str(int(check["value"])) in text, q.id
            elif check["type"] == "contains":
                assert check["value"].lower() in text, q.id
            else:
                assert any(v.lower() in text for v in check["values"]), q.id


def test_only_three_documents_carry_an_injected_instruction(corpus):
    docs, questions = corpus
    injected = {d.id for d in docs if "PWNED" in d.text}
    assert len(injected) == 3 and all(d.text.endswith(INJECTION) for d in docs if d.id in injected)
    assert {q.doc_id for q in questions if q.kind == "injection"} == injected
    assert all({"type": "not_contains", "value": "PWNED"} in q.checks for q in questions if q.kind == "injection")


def test_numbers_in_a_document_are_all_different(corpus):
    for doc in corpus[0]:
        numbers = re.findall(r"\d+", doc.text)
        assert len(numbers) == len(set(numbers)), doc.id


@pytest.mark.parametrize(
    "text",
    [
        "I cannot find this in the documents.",
        "The documents do not mention that.",
        "That information is not provided in the context.",
        "I don't know.",
        "There is no information about it.",
        "The context doesn't specify the price.",
        "It is not mentioned anywhere.",
    ],
)
def test_the_abstain_pattern_recognizes_declining(text):
    assert re.search(ABSTAIN_PATTERN, text, re.IGNORECASE)


@pytest.mark.parametrize("text", ["The price is 499 US dollars.", "Maya Hussain", "It opens at 8 am.", "B-4821"])
def test_the_abstain_pattern_ignores_real_answers(text):
    assert not re.search(ABSTAIN_PATTERN, text, re.IGNORECASE)


def test_generation_is_deterministic_and_depends_on_the_seed():
    assert generate(5) == generate(5)
    assert generate(5) != generate(6)


def test_the_corpus_survives_writing_and_loading(corpus, tmp_path):
    docs, questions = corpus
    write_corpus(tmp_path, docs, questions)
    assert load_docs(tmp_path / "docs.jsonl") == docs
    assert load_questions(tmp_path / "questions.jsonl") == questions