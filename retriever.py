"""Glue between the corpus and the ranker.

A :class:`Retriever` holds a list of :class:`~rag.corpus.Chunk` objects and a
:class:`~rag.bm25.BM25` index built over their text. :meth:`retrieve` returns the
top-``k`` chunks for a query, each paired with its BM25 score, ready to be
formatted into a grounded prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .corpus import Chunk


@dataclass(frozen=True)
class Retrieved:
    """A chunk that matched a query, with its relevance score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source


class Retriever:
    """A BM25 index over a fixed set of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("Retriever needs at least one chunk to index.")
        self.chunks = list(chunks)
        self.bm25 = BM25(k1=k1, b=b).index([c.text for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        """Top-``k`` chunks for ``query``, best first. May be shorter than ``k``
        (or empty) when fewer chunks share any term with the query."""
        hits = self.bm25.search(query, k=k)
        return [Retrieved(chunk=self.chunks[i], score=score) for i, score in hits]

    def __len__(self) -> int:
        return len(self.chunks)
