"""Generate the synthetic sample corpus shipped in data/ and eval/.

Why synthetic: the evaluation harness and the latency benchmark need a fixed
corpus with known relevance labels, reproducible in CI with no network. Real
blog posts cannot be redistributed, and a corpus that changes whenever a feed
updates makes nDCG incomparable between runs.

The generator is deterministic. Run it to regenerate; the output is committed
so that `pytest` and `python bench.py` work on a fresh clone.

Relevance labels fall out of the structure rather than being hand-assigned:
an article belonging to the query's topic is graded 2, one belonging to a
declared neighbour topic is graded 1, everything else 0.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPOCH = datetime(2026, 9, 1, tzinfo=timezone.utc)

# topic -> (vocabulary that defines it, neighbouring topics)
TOPICS: dict[str, tuple[list[str], list[str]]] = {
    "bm25-ranking": (
        ["bm25", "okapi", "inverse document frequency", "term saturation",
         "length normalisation", "postings list", "lexical retrieval"],
        ["vector-search", "search-evaluation"],
    ),
    "vector-search": (
        ["embedding", "cosine similarity", "approximate nearest neighbour",
         "hnsw graph", "vector database", "dense retrieval", "quantisation"],
        ["bm25-ranking", "rag-pipelines"],
    ),
    "search-evaluation": (
        ["ndcg", "discounted cumulative gain", "precision at k", "relevance judgement",
         "graded labels", "mean reciprocal rank", "offline evaluation"],
        ["bm25-ranking", "rag-pipelines"],
    ),
    "rag-pipelines": (
        ["retrieval augmented generation", "chunking strategy", "context window",
         "grounding", "hallucination", "query rewriting", "reranker"],
        ["vector-search", "search-evaluation"],
    ),
    "async-python": (
        ["asyncio", "event loop", "coroutine", "semaphore", "concurrency limit",
         "await", "task group"],
        ["python-performance"],
    ),
    "python-performance": (
        ["profiling", "cprofile", "flame graph", "memory allocation",
         "cpython interpreter", "bytecode", "hot loop"],
        ["async-python"],
    ),
    "postgres-indexing": (
        ["btree index", "query planner", "sequential scan", "explain analyze",
         "partial index", "vacuum", "bloat"],
        ["database-sharding"],
    ),
    "database-sharding": (
        ["shard key", "horizontal partitioning", "rebalancing", "consistent hashing",
         "cross shard join", "replication lag"],
        ["postgres-indexing"],
    ),
}

SHAPES = [
    ("A practical guide to {v}", "This post works through {v} from first principles. "
     "We start with why {v} matters, then build a small example, measure it, and "
     "discuss where the approach stops being appropriate."),
    ("Debugging {v} in production", "A short incident write-up. Our service degraded "
     "and {v} turned out to be the cause. This covers how we found it, what the fix "
     "was, and the monitoring we added so the next occurrence is obvious."),
    ("Notes on {v}", "Working notes rather than a tutorial. Observations collected "
     "while implementing {v}, including two things the documentation does not make "
     "obvious and one benchmark that contradicted what we expected."),
    ("{v}: what the benchmarks actually show", "Benchmarks for {v} are widely quoted "
     "and rarely reproduced. We reran them on our own workload and got materially "
     "different numbers. This explains the discrepancy and what it means in practice."),
    ("Why we moved away from {v}", "We used {v} for two years and recently replaced it. "
     "This is an honest account of what it was good at, the constraint that eventually "
     "made it untenable, and what the migration cost."),
]

SOURCES = ["https://eng.example.com/feed", "https://notes.example.org/atom",
           "https://blog.example.net/rss"]

# Distractors: articles that mention another topic's vocabulary exactly once,
# in passing, while being about something else entirely. They are graded 0.
#
# Without these, every document a query retrieves is relevant, precision@10 is
# ~1.0 by construction, and a shuffled ranking scores almost as well as a real
# one — which makes the whole evaluation unable to detect a ranking change.
# Passing mentions are also the realistic failure case for lexical retrieval:
# "we considered BM25 and rejected it" is a strong lexical match and a bad result.
DISTRACTOR = (
    "Hiring notes from our {year} engineering round",
    "An account of how we restructured interviewing this year: what we removed, "
    "what we kept, and the outcomes six months on. One candidate asked about "
    "{v}, which prompted a longer discussion we have written up separately. "
    "Most of this post is about process rather than technology.",
)


def build():
    articles, ordinal = [], 0
    for topic, (vocabulary, _) in TOPICS.items():
        # Two articles per term, under different shapes, so that a topic is
        # represented by more documents than any single query needs. Without
        # that, recall@10 is trivially 1.0 and the metric says nothing.
        for term in vocabulary * 2:
            shape_title, shape_body = SHAPES[ordinal % len(SHAPES)]
            # Body repeats the topic vocabulary so documents within a topic are
            # lexically related without being identical.
            context = " ".join(v for v in vocabulary if v != term)
            articles.append(
                {
                    "title": shape_title.format(v=term),
                    "body": shape_body.format(v=term)
                    + f" Related material covers {context}.",
                    "url": f"https://example.com/{topic}/{ordinal}",
                    "source": SOURCES[ordinal % len(SOURCES)],
                    "published": (EPOCH - timedelta(days=7 * ordinal)).isoformat(),
                    "topic": topic,
                }
            )
            ordinal += 1

    # One distractor per vocabulary term, attributed to an unrelated topic.
    topic_names = list(TOPICS)
    for index, (topic, (vocabulary, neighbours)) in enumerate(TOPICS.items()):
        # Place distractors under a topic that is neither this one nor a
        # declared neighbour, so they are unambiguously graded 0.
        host = next(
            name for name in topic_names[index + 1 :] + topic_names[:index]
            if name != topic and name not in neighbours
        )
        for term in vocabulary:
            title, body = DISTRACTOR
            articles.append(
                {
                    "title": title.format(year=2024 + ordinal % 3),
                    "body": body.format(v=term),
                    "url": f"https://example.com/{host}/distractor-{ordinal}",
                    "source": SOURCES[ordinal % len(SOURCES)],
                    "published": (EPOCH - timedelta(days=3 * ordinal)).isoformat(),
                    "topic": host,
                }
            )
            ordinal += 1

    queries = []
    for topic, (vocabulary, neighbours) in TOPICS.items():
        relevance = {}
        for article in articles:
            if article["topic"] == topic:
                relevance[article["url"]] = 2
            elif article["topic"] in neighbours:
                relevance[article["url"]] = 1
        # Three phrasings per topic, drawn from different vocabulary terms,
        # so the score is not dominated by one lucky query formulation.
        for term in (vocabulary[0], vocabulary[2], vocabulary[-1]):
            queries.append(
                {
                    "query": term,
                    "topic": topic,
                    "relevant": relevance,
                }
            )

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "data" / "corpus.json").write_text(json.dumps(articles, indent=2) + "\n")
    (ROOT / "eval" / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
    print(f"{len(articles)} articles across {len(TOPICS)} topics")
    print(f"{len(queries)} labelled queries")


if __name__ == "__main__":
    build()
