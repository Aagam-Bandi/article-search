import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.corpus import load_corpus, save_corpus
from src.documents import Document
from src.index import build_index
from src.index.bm25 import BM25Index
from src.search import SearchService

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_build_index_returns_bm25_by_default():
    assert isinstance(build_index(), BM25Index)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        build_index("solr")


def test_service_returns_hits_with_component_scores():
    index = BM25Index()
    index.add([Document.build(title="Asyncio", body="event loop", url="u", source="s")])
    hit = SearchService(index).search("asyncio")[0]
    assert hit.relevance > 0 and 0 < hit.recency <= 1.0


def test_overfetch_lets_recency_promote_from_outside_the_top_k():
    """If the index were asked for exactly k, decay could only reorder, never
    change, the result set."""
    documents = [
        Document.build(
            title="Target term", body="filler " * (i + 1), url=f"u{i}", source="s",
            published=NOW - timedelta(days=3650 if i < 3 else 1),
        )
        for i in range(6)
    ]
    index = BM25Index()
    index.add(documents)
    hits = SearchService(index).search("target", limit=2, now=NOW)
    assert any(h.document.url in {"u3", "u4", "u5"} for h in hits)


def test_empty_query_returns_no_hits():
    index = BM25Index()
    index.add([Document.build(title="A", body="b", url="u", source="s")])
    assert SearchService(index).search("") == []


def test_corpus_round_trips_through_disk(tmp_path):
    original = [
        Document.build(title="T", body="B", url="u", source="s", published=NOW)
    ]
    path = tmp_path / "c.json"
    save_corpus(original, path)
    assert load_corpus(path)[0].doc_id == original[0].doc_id


def test_shipped_corpus_and_queries_are_consistent():
    """Every labelled URL must exist in the corpus, or nDCG is measured against
    documents that can never be retrieved."""
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    urls = {d.url for d in corpus}
    queries = json.loads((ROOT / "eval" / "queries.json").read_text())
    labelled = {url for q in queries for url in q["relevant"]}
    assert labelled <= urls


def test_shipped_corpus_has_no_duplicate_ids():
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    assert len({d.doc_id for d in corpus}) == len(corpus)
