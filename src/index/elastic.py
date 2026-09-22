"""Elasticsearch backend.

Optional. Selected with `--backend elasticsearch`; everything else in the
repository, including the whole test suite, runs without it. The import of
the client lives inside __init__ so that `pip install -r requirements.txt`
without an ES cluster still leaves the package importable.

The field weighting mirrors BM25Index so results are comparable between
backends rather than silently different.
"""
from __future__ import annotations

from ..documents import Document

INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "english"},
            "body": {"type": "text", "analyzer": "english"},
            "url": {"type": "keyword"},
            "source": {"type": "keyword"},
            "published": {"type": "date"},
        }
    }
}


class ElasticIndex:
    def __init__(
        self,
        *,
        url: str = "http://localhost:9200",
        index_name: str = "articles",
        title_weight: float = 2.5,
        body_weight: float = 1.0,
        client=None,
    ) -> None:
        if client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "the elasticsearch backend needs `pip install elasticsearch`; "
                    "the default bm25 backend needs nothing"
                ) from exc
            client = Elasticsearch(url)
        self.client = client
        self.index_name = index_name
        self.title_weight = title_weight
        self.body_weight = body_weight
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=INDEX_SETTINGS)

    def __len__(self) -> int:
        return int(self.client.count(index=self.index_name)["count"])

    def add(self, documents: list[Document]) -> int:
        added = 0
        for doc in documents:
            # doc_id is the content hash, so re-indexing the same article
            # overwrites rather than duplicates — the dedup guarantee holds
            # here for the same reason it does in BM25Index.
            existed = self.client.exists(index=self.index_name, id=doc.doc_id)
            self.client.index(
                index=self.index_name,
                id=doc.doc_id,
                document={
                    "title": doc.title,
                    "body": doc.body,
                    "url": doc.url,
                    "source": doc.source,
                    "published": doc.published.isoformat() if doc.published else None,
                },
            )
            if not existed:
                added += 1
        self.client.indices.refresh(index=self.index_name)
        return added

    def search(self, query: str, *, limit: int = 10) -> list[tuple[Document, float]]:
        from datetime import datetime

        response = self.client.search(
            index=self.index_name,
            size=limit,
            query={
                "multi_match": {
                    "query": query,
                    "fields": [
                        f"title^{self.title_weight}",
                        f"body^{self.body_weight}",
                    ],
                }
            },
        )
        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            published = (
                datetime.fromisoformat(src["published"]) if src.get("published") else None
            )
            results.append(
                (
                    Document.build(
                        title=src["title"],
                        body=src["body"],
                        url=src["url"],
                        source=src["source"],
                        published=published,
                    ),
                    float(hit["_score"]),
                )
            )
        return results
