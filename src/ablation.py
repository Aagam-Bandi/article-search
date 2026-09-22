"""Ablations.

A single nDCG number is close to meaningless on its own: it tells you nothing
about whether the score came from the ranker or from the corpus being easy.
These configurations exist to answer that question by removing one component
at a time and reporting what it cost.

The shuffled baseline is the important row. It retrieves the same candidate
documents and destroys only the ordering, which means any gap between it and
a real configuration is attributable to ranking rather than to matching. If
the full system were only marginally above it, nDCG would be measuring the
retrieval filter and not the ranking at all.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .corpus import load_corpus
from .evaluate import Query, evaluate
from .index.bm25 import BM25Index
from .search import SearchService


class _ShuffledService:
    """Retrieves correctly, then orders at random. Lower bound for the metric."""

    def __init__(self, service: SearchService, seed: int = 0) -> None:
        self.service = service
        self.random = random.Random(seed)

    def search(self, query: str, *, limit: int = 10, now=None):
        hits = self.service.search(query, limit=limit * 3, now=now)
        self.random.shuffle(hits)
        return hits[:limit]


@dataclass(frozen=True)
class Row:
    name: str
    ndcg: float
    precision: float
    mrr: float
    note: str


def run(corpus_path, queries: list[Query], *, k: int = 10) -> list[Row]:
    documents = load_corpus(corpus_path)

    def service(title_weight: float, half_life: float) -> SearchService:
        index = BM25Index(title_weight=title_weight)
        index.add(documents)
        return SearchService(index, half_life=half_life)

    full = service(2.5, 240.0)
    configurations = [
        ("full", full, "BM25F (title x2.5) + recency decay"),
        ("no title boost", service(1.0, 240.0), "title and body weighted equally"),
        ("no recency decay", service(2.5, 0.0), "relevance only"),
        ("shuffled baseline", _ShuffledService(full), "same candidates, random order"),
    ]

    rows = []
    for name, svc, note in configurations:
        report = evaluate(svc, queries, k=k)
        rows.append(
            Row(name=name, ndcg=report.ndcg, precision=report.precision,
                mrr=report.mrr, note=note)
        )
    return rows


def sweep_title_weight(corpus_path, queries: list[Query], *, k: int = 10,
                       weights=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0)) -> list[tuple[float, float]]:
    """nDCG as a function of the title weight.

    Reported because a single "no title boost" row cannot distinguish "the
    boost is harmful" from "the boost is mistuned". The shape of the curve
    can.
    """
    documents = load_corpus(corpus_path)
    out = []
    for weight in weights:
        index = BM25Index(title_weight=weight)
        index.add(documents)
        report = evaluate(SearchService(index, half_life=240.0), queries, k=k)
        out.append((weight, report.ndcg))
    return out


def format_rows(rows: list[Row], k: int = 10) -> str:
    header = f"{'configuration':<20}{'nDCG@' + str(k):>9}{'P@' + str(k):>9}{'MRR':>8}   note"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.name:<20}{row.ndcg:>9.3f}{row.precision:>9.3f}{row.mrr:>8.3f}   {row.note}"
        )
    return "\n".join(lines)
