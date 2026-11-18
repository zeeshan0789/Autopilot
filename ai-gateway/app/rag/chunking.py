"""Splitting documents into chunks of whole sentences, with a little overlap between neighbours."""

import re
from dataclasses import dataclass

from app.rag.corpus import Doc

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str

    @property
    def embedding_text(self) -> str:
        """The title is part of what is embedded, because questions usually name the thing they ask about."""
        return f"{self.title}. {self.text}"


def split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text.strip()) if part]


def chunk_document(doc: Doc, max_chars: int = 450, overlap_sentences: int = 1) -> list[Chunk]:
    """Chunks of whole sentences, up to max_chars (a longer single sentence is its own chunk).
    The last sentences of a chunk open the next one."""
    groups: list = []
    current: list = []
    for sentence in split_sentences(doc.text):
        size = sum(len(s) + 1 for s in current)
        if current and size + len(sentence) > max_chars:
            groups.append(current)
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
        current.append(sentence)
    if current:
        groups.append(current)
    return [Chunk(f"{doc.id}#{i}", doc.id, doc.title, " ".join(group)) for i, group in enumerate(groups)]


def chunk_documents(docs: list[Doc], max_chars: int = 450, overlap_sentences: int = 1) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, max_chars, overlap_sentences)]