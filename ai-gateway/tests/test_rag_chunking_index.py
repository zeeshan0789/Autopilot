import re
import zlib

import numpy as np
import pytest

from app.rag.chunking import chunk_document, chunk_documents, split_sentences
from app.rag.corpus import Doc
from app.rag.index import ChunkIndex


class FakeEmbedder:
    """A bag of words embedding: texts that share words are close."""

    def encode(self, text):
        vector = np.zeros(64, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % 64] += 1.0
        return vector / np.linalg.norm(vector)


DOC = Doc("d1", "Title", "First sentence here. Second sentence follows. Third one ends it. Fourth is last.")


def test_sentences_are_split_after_their_punctuation():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("  ") == []


def test_a_short_document_is_one_chunk():
    chunks = chunk_document(DOC, max_chars=1000)
    assert [c.id for c in chunks] == ["d1#0"]
    assert chunks[0].text == "First sentence here. Second sentence follows. Third one ends it. Fourth is last."
    assert chunks[0].doc_id == "d1" and chunks[0].title == "Title"


def test_a_long_document_is_split_into_whole_sentences_with_overlap():
    chunks = chunk_document(DOC, max_chars=45, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(re.search(r"[.!?]$", c.text) for c in chunks)  # no sentence is cut
    for first, second in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(first.text)[-1]
        assert split_sentences(second.text)[0] == last_sentence  # the neighbours share a sentence


def test_without_overlap_no_sentence_repeats():
    chunks = chunk_document(DOC, max_chars=45, overlap_sentences=0)
    sentences = [s for c in chunks for s in split_sentences(c.text)]
    assert sentences == split_sentences(DOC.text)


def test_every_sentence_is_in_some_chunk():
    chunks = chunk_document(DOC, max_chars=45)
    covered = {s for c in chunks for s in split_sentences(c.text)}
    assert covered == set(split_sentences(DOC.text))


def test_a_sentence_longer_than_the_limit_is_kept_whole():
    long_doc = Doc("d2", "T", "A" * 200 + ". Short one.")
    chunks = chunk_document(long_doc, max_chars=50)
    assert any(len(c.text) >= 200 for c in chunks)


def test_the_title_is_part_of_the_embedded_text():
    assert chunk_document(DOC)[0].embedding_text.startswith("Title. First sentence")


def test_documents_are_chunked_together():
    docs = [DOC, Doc("d3", "Other", "Only one sentence.")]
    assert [c.id for c in chunk_documents(docs, max_chars=1000)] == ["d1#0", "d3#0"]


DOCS = [
    Doc("paris", "Paris facts", "The capital of France is Paris. It lies on the river Seine."),
    Doc("berlin", "Berlin facts", "The capital of Germany is Berlin. It lies on the river Spree."),
    Doc("tokyo", "Tokyo facts", "Tokyo is a very large city in Japan with many trains."),
]


@pytest.fixture(scope="module")
def index():
    return ChunkIndex(chunk_documents(DOCS), FakeEmbedder())


def test_the_best_matching_chunk_comes_first(index):
    hits = index.search("capital of France", 3)
    assert [h.chunk.doc_id for h in hits][0] == "paris"
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score >= hits[2].score
    assert index.search("trains in Japan", 1)[0].chunk.doc_id == "tokyo"


def test_k_is_limited_by_the_number_of_chunks(index):
    assert len(index.search("anything", 50)) == 3


def test_the_chunks_of_a_document(index):
    assert [c.doc_id for c in index.chunks_of("berlin")] == ["berlin"]
    assert index.chunks_of("nothing") == []


def test_an_empty_index_is_refused():
    with pytest.raises(ValueError):
        ChunkIndex([], FakeEmbedder())