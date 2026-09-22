"""Document model and text normalisation.

Everything downstream — indexing, ranking, evaluation — operates on `Document`,
never on raw feed entries. Keeping the parse boundary here means a malformed
feed can only produce a bad Document, not corrupt the index.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Deliberately short. A long stopword list hurts more than it helps on blog
# titles, where words like "how" and "why" carry real intent ("how to profile
# python" is not the same query as "profile python").
STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the to was were with".split()
)

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_WS = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove tags and collapse whitespace. Feeds ship HTML in summaries."""
    return _WS.sub(" ", _TAG.sub(" ", text or "")).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase, fold accents, split on word characters, drop stopwords.

    No stemming: see the Limitations section of the README. Intra-word dots,
    apostrophes and hyphens are kept so that "asyncio.run", "don't" and
    "walk-forward" survive as single tokens rather than fragmenting.
    """
    folded = unicodedata.normalize("NFKD", (text or "").lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [t for t in _WORD.findall(folded) if t not in STOPWORDS]


def content_hash(title: str, body: str) -> str:
    """Identity for deduplication.

    Hashing content rather than URL is the point: syndicated posts appear on
    several URLs (canonical site, Medium mirror, aggregator) with identical
    text, and a URL-keyed index stores each of them separately.
    """
    norm = " ".join(tokenize(title)) + "\x00" + " ".join(tokenize(body))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    url: str
    source: str
    published: datetime | None = None
    tokens_title: tuple[str, ...] = field(default=(), repr=False)
    tokens_body: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def build(
        cls,
        *,
        title: str,
        body: str,
        url: str,
        source: str,
        published: datetime | None = None,
    ) -> "Document":
        title = strip_html(title)
        body = strip_html(body)
        if published is not None and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return cls(
            doc_id=content_hash(title, body),
            title=title,
            body=body,
            url=url,
            source=source,
            published=published,
            tokens_title=tuple(tokenize(title)),
            tokens_body=tuple(tokenize(body)),
        )

    def age_days(self, now: datetime | None = None) -> float | None:
        if self.published is None:
            return None
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.published).total_seconds() / 86400.0)
