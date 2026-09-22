"""Query pipeline: index lookup, then re-rank."""
from __future__ import annotations

from datetime import datetime

from .index import SearchIndex
from .rank import DEFAULT_FLOOR, DEFAULT_HALF_LIFE_DAYS, Hit, rank

# The index is asked for more candidates than the caller wants, because
# recency re-ranking can promote a document from outside the top-k. Asking
# for exactly k would make the decay unable to change the result set at all,
# only its order.
OVERFETCH = 4


class SearchService:
    def __init__(
        self,
        index: SearchIndex,
        *,
        half_life: float = DEFAULT_HALF_LIFE_DAYS,
        floor: float = DEFAULT_FLOOR,
    ) -> None:
        self.index = index
        self.half_life = half_life
        self.floor = floor

    def search(
        self, query: str, *, limit: int = 10, now: datetime | None = None
    ) -> list[Hit]:
        candidates = self.index.search(query, limit=limit * OVERFETCH)
        return rank(
            candidates,
            now=now,
            half_life=self.half_life,
            floor=self.floor,
            limit=limit,
        )
