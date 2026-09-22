#!/usr/bin/env python3
"""Query latency benchmark.

Reports p50/p95/p99 rather than a mean. A mean latency hides the tail, and
the tail is what a user actually notices: a search box whose median is 8ms
and whose p99 is 900ms feels broken, and its mean looks excellent.

Measures query time only. Index construction is reported separately because
it happens once at startup, not per request.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from src.corpus import load_corpus
from src.evaluate import load_queries
from src.index.bm25 import BM25Index
from src.search import SearchService

ROOT = Path(__file__).resolve().parent


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def synthesise(documents, target: int):
    """Grow the corpus to `target` documents without creating duplicates.

    Each copy gets a distinct token appended so its content hash differs —
    otherwise dedup would silently collapse the whole thing back to the
    original size and the benchmark would measure nothing.
    """
    from src.documents import Document

    out = []
    copy_index = 0
    while len(out) < target:
        for doc in documents:
            if len(out) >= target:
                break
            out.append(
                Document.build(
                    title=f"{doc.title} (variant {copy_index})",
                    body=f"{doc.body} variant{copy_index}",
                    url=f"{doc.url}?v={copy_index}",
                    source=doc.source,
                    published=doc.published,
                )
            )
        copy_index += 1
    return out


def run_scale(args) -> int:
    base = load_corpus(args.corpus)
    queries = [q.text for q in load_queries(args.queries)]
    print(f"{'documents':>10}{'build ms':>11}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}")
    print("-" * 48)
    for size in (1_000, 10_000, 50_000, 100_000):
        documents = synthesise(base, size)
        start = time.perf_counter()
        index = BM25Index()
        index.add(documents)
        build_ms = (time.perf_counter() - start) * 1000
        service = SearchService(index)
        for text in queries:
            service.search(text, limit=10)
        timings = []
        for _ in range(max(1, args.repeat // 10)):
            for text in queries:
                start = time.perf_counter()
                service.search(text, limit=10)
                timings.append((time.perf_counter() - start) * 1000)
        print(f"{len(documents):>10,}{build_ms:>11.0f}"
              f"{statistics.median(timings):>9.2f}"
              f"{percentile(timings, 0.95):>9.2f}"
              f"{percentile(timings, 0.99):>9.2f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.json")
    parser.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries.json")
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--scale", action="store_true",
                        help="report latency at several corpus sizes")
    args = parser.parse_args(argv)

    if args.scale:
        return run_scale(args)

    documents = load_corpus(args.corpus)

    start = time.perf_counter()
    index = BM25Index()
    index.add(documents)
    build_ms = (time.perf_counter() - start) * 1000

    service = SearchService(index)
    queries = [q.text for q in load_queries(args.queries)]

    for text in queries:  # warm-up, excluded from the numbers
        service.search(text, limit=10)

    timings: list[float] = []
    for _ in range(args.repeat):
        for text in queries:
            start = time.perf_counter()
            service.search(text, limit=10)
            timings.append((time.perf_counter() - start) * 1000)

    print(f"corpus         {len(documents)} documents")
    print(f"index build    {build_ms:.1f} ms")
    print(f"queries        {len(timings)} ({len(queries)} distinct x {args.repeat})")
    print(f"p50            {statistics.median(timings):.2f} ms")
    print(f"p95            {percentile(timings, 0.95):.2f} ms")
    print(f"p99            {percentile(timings, 0.99):.2f} ms")
    print(f"max            {max(timings):.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
