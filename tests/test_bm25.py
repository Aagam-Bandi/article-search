import math

from src.documents import Document
from src.index.bm25 import BM25Index


def doc(title, body, url):
    return Document.build(title=title, body=body, url=url, source="test")


def test_empty_index_returns_nothing():
    assert BM25Index().search("anything") == []


def test_empty_query_returns_nothing():
    index = BM25Index()
    index.add([doc("Python", "about python", "u1")])
    assert index.search("") == []


def test_stopword_only_query_returns_nothing():
    index = BM25Index()
    index.add([doc("Python", "about python", "u1")])
    assert index.search("the and of") == []


def test_matching_document_ranks_above_non_matching():
    index = BM25Index()
    index.add([doc("Python", "python asyncio", "u1"), doc("Rust", "rust macros", "u2")])
    results = index.search("asyncio")
    assert [d.url for d, _ in results] == ["u1"]


def test_duplicate_documents_are_added_once():
    index = BM25Index()
    first = index.add([doc("Same", "same body", "u1")])
    second = index.add([doc("Same", "same body", "u2")])
    assert (first, second, len(index)) == (1, 0, 1)


def test_title_match_outranks_body_match():
    """The reason BM25F exists here rather than a single concatenated field."""
    index = BM25Index(title_weight=2.5, body_weight=1.0)
    index.add(
        [
            doc("Profiling Python", "general notes " * 20, "title-hit"),
            doc("General notes", "profiling python " + "filler " * 20, "body-hit"),
        ]
    )
    ranked = [d.url for d, _ in index.search("profiling")]
    assert ranked[0] == "title-hit"


def test_idf_stays_positive_for_very_common_terms():
    """Regression: the +1 guard in the IDF formula.

    Without it, a term appearing in more than half the corpus gets a negative
    IDF and documents containing it score below documents that don't.
    """
    index = BM25Index()
    index.add([doc(f"Post {i}", "common term here", f"u{i}") for i in range(10)])
    scores = [s for _, s in index.search("common")]
    assert scores and all(s > 0 for s in scores)


def test_length_normalisation_prefers_the_shorter_document():
    index = BM25Index()
    index.add(
        [
            doc("A", "target", "short"),
            doc("B", "target " + "padding " * 100, "long"),
        ]
    )
    ranked = [d.url for d, _ in index.search("target")]
    assert ranked == ["short", "long"]


def test_scores_are_finite():
    index = BM25Index()
    index.add([doc("A", "x", "u1")])
    assert all(math.isfinite(s) for _, s in index.search("x"))


def test_limit_is_respected():
    index = BM25Index()
    index.add([doc(f"Post {i}", "shared term", f"u{i}") for i in range(20)])
    assert len(index.search("shared", limit=5)) == 5


def test_ranking_is_deterministic_for_tied_scores():
    index = BM25Index()
    index.add([doc(f"Same title {i}", "identical body", f"u{i}") for i in range(5)])
    first = [d.url for d, _ in index.search("identical")]
    second = [d.url for d, _ in index.search("identical")]
    assert first == second
