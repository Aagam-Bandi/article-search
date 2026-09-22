from datetime import datetime, timedelta, timezone

from src.documents import Document, content_hash, strip_html, tokenize


def test_strip_html_removes_tags_and_collapses_whitespace():
    assert strip_html("<p>hello   <b>world</b></p>") == "hello world"


def test_tokenize_drops_stopwords_and_folds_case():
    assert tokenize("The Quick and THE Dead") == ["quick", "dead"]


def test_tokenize_folds_accents():
    assert tokenize("naïve café") == ["naive", "cafe"]


def test_tokenize_keeps_intra_word_punctuation():
    """asyncio.run and walk-forward must not fragment into separate tokens."""
    assert tokenize("asyncio.run and walk-forward and don't") == [
        "asyncio.run",
        "walk-forward",
        "don't",
    ]


def test_same_content_at_different_urls_has_the_same_id():
    """The dedup guarantee. Syndicated posts must collapse to one document."""
    a = Document.build(title="A Post", body="Body text", url="https://a.com/1", source="a")
    b = Document.build(title="A Post", body="Body text", url="https://b.com/mirror", source="b")
    assert a.doc_id == b.doc_id


def test_different_content_has_different_ids():
    a = Document.build(title="A Post", body="Body text", url="https://a.com/1", source="a")
    b = Document.build(title="A Post", body="Other text", url="https://a.com/2", source="a")
    assert a.doc_id != b.doc_id


def test_content_hash_ignores_html_and_case():
    assert content_hash("Hello", "World") == content_hash("hello", "world")


def test_naive_datetimes_are_treated_as_utc():
    doc = Document.build(
        title="t", body="b", url="u", source="s", published=datetime(2026, 1, 1)
    )
    assert doc.published.tzinfo is timezone.utc


def test_age_days_is_never_negative_for_future_dates():
    future = datetime.now(timezone.utc) + timedelta(days=30)
    doc = Document.build(title="t", body="b", url="u", source="s", published=future)
    assert doc.age_days() == 0.0


def test_age_days_is_none_when_undated():
    doc = Document.build(title="t", body="b", url="u", source="s")
    assert doc.age_days() is None
