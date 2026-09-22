"""Final ranking: relevance, then recency.

The index returns a relevance score. This module decides how much a post's
age is allowed to move it.

    final = relevance * (1 - floor + floor * 2^(-age_days / half_life))

The decay is *multiplicative and floored* rather than additive, which is the
decision worth explaining. An additive recency bonus can promote an
irrelevant-but-recent post above a relevant one, because the bonus doesn't
know anything about the query. A multiplicative factor can only ever reorder
documents that already matched, so relevance stays the dominant signal.

`floor` caps the damage: at floor=0.5 the oldest possible post is worth half
its relevance, never zero. Blog archives contain posts that are both ancient
and definitive, and a decay that tends to zero buries them permanently.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .documents import Document

DEFAULT_HALF_LIFE_DAYS = 240.0
DEFAULT_FLOOR = 0.5


@dataclass(frozen=True)
class Hit:
    document: Document
    relevance: float
    recency: float
    score: float


def recency_factor(
    age_days: float | None,
    *,
    half_life: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = DEFAULT_FLOOR,
) -> float:
    """Multiplier in [1 - floor, 1]. Undated posts are treated as neutral.

    Treating an unknown date as "old" would penalise every feed that omits
    publication dates — which is a property of the feed, not the article.
    """
    if age_days is None:
        return 1.0
    if half_life <= 0:
        return 1.0
    return (1.0 - floor) + floor * (2.0 ** (-age_days / half_life))


def rank(
    scored: list[tuple[Document, float]],
    *,
    now: datetime | None = None,
    half_life: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = DEFAULT_FLOOR,
    limit: int | None = None,
) -> list[Hit]:
    hits = []
    for document, relevance in scored:
        factor = recency_factor(
            document.age_days(now), half_life=half_life, floor=floor
        )
        hits.append(
            Hit(
                document=document,
                relevance=relevance,
                recency=factor,
                score=relevance * factor,
            )
        )
    hits.sort(key=lambda h: (-h.score, h.document.doc_id))
    return hits[:limit] if limit else hits
