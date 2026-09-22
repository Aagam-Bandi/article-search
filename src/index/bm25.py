"""In-process BM25F.

Okapi BM25 computed per field, then combined with weights. Field-weighted
scoring matters more than it looks on blog data: a term in the title is
strong evidence about what the post is *about*, whereas the same term once
in a 2,000-word body is often incidental.

Why not just concatenate title and body into one field? Because BM25's length
normalisation would then treat the title as a handful of tokens in a long
document and effectively erase it.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..documents import Document, tokenize

K1 = 1.5   # term-frequency saturation
B = 0.75   # length-normalisation strength


class _Field:
    """Postings and statistics for one field."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.lengths: list[int] = []
        self.total_length = 0

    @property
    def avg_length(self) -> float:
        return self.total_length / len(self.lengths) if self.lengths else 0.0

    def add(self, ordinal: int, tokens: tuple[str, ...]) -> None:
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for token, tf in counts.items():
            self.postings[token][ordinal] = tf
        self.lengths.append(len(tokens))
        self.total_length += len(tokens)

    def score(self, term: str, n_docs: int) -> dict[int, float]:
        """BM25 contribution of one term across every document containing it."""
        postings = self.postings.get(term)
        if not postings:
            return {}
        df = len(postings)
        # Robertson/Sparck-Jones IDF with the +1 guard, which keeps the value
        # positive for terms appearing in more than half the corpus. Without
        # it, common terms score negative and can push a document below zero.
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        avg = self.avg_length or 1.0
        out: dict[int, float] = {}
        for ordinal, tf in postings.items():
            norm = 1.0 - B + B * (self.lengths[ordinal] / avg)
            out[ordinal] = idf * (tf * (K1 + 1.0)) / (tf + K1 * norm)
        return out


class BM25Index:
    def __init__(self, *, title_weight: float = 2.5, body_weight: float = 1.0) -> None:
        self.title_weight = title_weight
        self.body_weight = body_weight
        self._docs: list[Document] = []
        self._seen: set[str] = set()
        self._title = _Field()
        self._body = _Field()

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def documents(self) -> list[Document]:
        return list(self._docs)

    def add(self, documents: list[Document]) -> int:
        added = 0
        for doc in documents:
            if doc.doc_id in self._seen:
                continue
            self._seen.add(doc.doc_id)
            ordinal = len(self._docs)
            self._docs.append(doc)
            self._title.add(ordinal, doc.tokens_title)
            self._body.add(ordinal, doc.tokens_body)
            added += 1
        return added

    def search(self, query: str, *, limit: int = 10) -> list[tuple[Document, float]]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        n = len(self._docs)
        totals: dict[int, float] = defaultdict(float)
        for term in terms:
            for ordinal, score in self._title.score(term, n).items():
                totals[ordinal] += self.title_weight * score
            for ordinal, score in self._body.score(term, n).items():
                totals[ordinal] += self.body_weight * score
        ranked = sorted(totals.items(), key=lambda kv: (-kv[1], self._docs[kv[0]].doc_id))
        return [(self._docs[o], s) for o, s in ranked[:limit]]
