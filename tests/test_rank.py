from datetime import datetime, timedelta, timezone

from src.documents import Document
from src.rank import DEFAULT_FLOOR, rank, recency_factor

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def doc(url, days_old=None):
    published = NOW - timedelta(days=days_old) if days_old is not None else None
    return Document.build(title=url, body="body", url=url, source="s", published=published)


def test_fresh_content_is_undamped():
    assert recency_factor(0.0) == 1.0


def test_decay_reaches_the_half_life_midpoint():
    """At one half-life the decayable portion has halved, not the whole score."""
    assert recency_factor(240.0, half_life=240.0) == 0.75


def test_decay_never_falls_below_the_floor():
    assert recency_factor(100_000.0) >= 1.0 - DEFAULT_FLOOR


def test_undated_documents_are_treated_as_neutral():
    """A feed omitting dates is a property of the feed, not of the article."""
    assert recency_factor(None) == 1.0


def test_zero_half_life_disables_decay():
    assert recency_factor(500.0, half_life=0.0) == 1.0


def test_decay_is_monotonic_in_age():
    factors = [recency_factor(age) for age in (0, 30, 120, 365, 1000)]
    assert factors == sorted(factors, reverse=True)


def test_recency_breaks_ties_between_equally_relevant_documents():
    hits = rank([(doc("old", 720), 10.0), (doc("new", 1), 10.0)], now=NOW)
    assert [h.document.url for h in hits] == ["new", "old"]


def test_recency_cannot_overturn_a_large_relevance_gap():
    """The reason decay is multiplicative and floored rather than additive.

    A recent but weakly matching document must not displace a strong match;
    an additive bonus would allow exactly that.
    """
    hits = rank([(doc("strong", 3650), 10.0), (doc("weak", 0), 4.0)], now=NOW)
    assert [h.document.url for h in hits] == ["strong", "weak"]


def test_score_is_relevance_times_recency():
    hit = rank([(doc("u", 240), 8.0)], now=NOW)[0]
    assert abs(hit.score - hit.relevance * hit.recency) < 1e-9


def test_limit_is_respected():
    hits = rank([(doc(f"u{i}", i), float(i)) for i in range(10)], now=NOW, limit=3)
    assert len(hits) == 3
