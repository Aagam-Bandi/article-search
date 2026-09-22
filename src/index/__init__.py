"""Index backends.

`SearchIndex` is the seam that lets this repository run with no external
service. `BM25Index` is the default and is what CI exercises; `ElasticIndex`
speaks to a real cluster and is selected only when one is configured. Both
satisfy the same protocol, so ranking, evaluation and the API are written
once against the interface rather than twice against two engines.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..documents import Document


@runtime_checkable
class SearchIndex(Protocol):
    def add(self, documents: list[Document]) -> int:
        """Index documents. Returns the number newly added (duplicates skipped)."""

    def search(self, query: str, *, limit: int = 10) -> list[tuple[Document, float]]:
        """Return (document, relevance_score) ordered by score descending."""

    def __len__(self) -> int: ...


def build_index(backend: str = "bm25", **kwargs) -> SearchIndex:
    if backend == "bm25":
        from .bm25 import BM25Index

        return BM25Index(**kwargs)
    if backend == "elasticsearch":
        from .elastic import ElasticIndex

        return ElasticIndex(**kwargs)
    raise ValueError(f"unknown backend {backend!r} (expected 'bm25' or 'elasticsearch')")
