"""Retrieval evaluation: nDCG@k, precision@k, recall@k, MRR.

The point of this module is that "the ranking is good" becomes a number you
can regress against, rather than an impression formed by typing three queries
into the UI and liking what came back.

nDCG is the primary metric because it is the only one here that is sensitive
to *position*. Precision@10 cannot tell the difference between the right
answer at rank 1 and the same answer at rank 10; for a search box, that
difference is most of the user experience.

Graded relevance: labels are 2 (directly answers the query), 1 (related and
worth showing), 0 (irrelevant). Binary labels would force every
"related but not ideal" document into one bucket or the other, and blog
search is mostly made of that middle case.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Query:
    text: str
    relevance: dict[str, int]  # url -> graded label

    def gain(self, url: str) -> int:
        return self.relevance.get(url, 0)


@dataclass(frozen=True)
class Report:
    ndcg: float
    precision: float
    recall: float
    mrr: float
    k: int
    n_queries: int
    per_query: dict[str, float]

    def format(self) -> str:
        lines = [
            f"queries       {self.n_queries}",
            f"nDCG@{self.k:<9d}{self.ndcg:.3f}",
            f"precision@{self.k:<4d}{self.precision:.3f}",
            f"recall@{self.k:<7d}{self.recall:.3f}",
            f"MRR           {self.mrr:.3f}",
            "",
            "per query:",
        ]
        for text, score in sorted(self.per_query.items(), key=lambda kv: kv[1]):
            lines.append(f"  {score:.3f}  {text}")
        return "\n".join(lines)


def dcg(gains: list[int]) -> float:
    # 2^g - 1 numerator: the standard exponential formulation, which makes a
    # grade-2 document worth meaningfully more than two grade-1 documents
    # rather than exactly the same.
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved_urls: list[str], query: Query, k: int) -> float:
    gains = [query.gain(url) for url in retrieved_urls[:k]]
    ideal = sorted(query.relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    # A query with no relevant documents at all is undefined rather than zero.
    # Scoring it 0 would punish the system for a gap in the labels.
    return dcg(gains) / denominator if denominator > 0 else 0.0


def load_queries(path: str | Path) -> list[Query]:
    raw = json.loads(Path(path).read_text())
    return [Query(text=q["query"], relevance=q["relevant"]) for q in raw]


def evaluate(service, queries: list[Query], *, k: int = 10) -> Report:
    ndcgs, precisions, recalls, rrs = [], [], [], []
    per_query: dict[str, float] = {}

    for query in queries:
        hits = service.search(query.text, limit=k)
        urls = [hit.document.url for hit in hits]

        score = ndcg_at_k(urls, query, k)
        ndcgs.append(score)
        per_query[query.text] = score

        relevant_found = sum(1 for url in urls if query.gain(url) > 0)
        total_relevant = sum(1 for g in query.relevance.values() if g > 0)
        precisions.append(relevant_found / k)
        recalls.append(relevant_found / total_relevant if total_relevant else 0.0)

        rr = 0.0
        for position, url in enumerate(urls, start=1):
            if query.gain(url) > 0:
                rr = 1.0 / position
                break
        rrs.append(rr)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return Report(
        ndcg=mean(ndcgs),
        precision=mean(precisions),
        recall=mean(recalls),
        mrr=mean(rrs),
        k=k,
        n_queries=len(queries),
        per_query=per_query,
    )
