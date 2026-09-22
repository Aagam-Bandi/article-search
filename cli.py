#!/usr/bin/env python3
"""Command line entry point.

    python cli.py search "bm25"            search the sample corpus
    python cli.py evaluate                 nDCG/precision/recall over eval/queries.json
    python cli.py crawl data/feeds.json    fetch real feeds into data/corpus.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.corpus import load_corpus, save_corpus
from src.evaluate import evaluate, load_queries
from src.index import build_index
from src.search import SearchService

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "corpus.json"
DEFAULT_QUERIES = ROOT / "eval" / "queries.json"


def make_service(corpus: Path, backend: str, half_life: float) -> SearchService:
    index = build_index(backend)
    index.add(load_corpus(corpus))
    return SearchService(index, half_life=half_life)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--backend", default="bm25", choices=["bm25", "elasticsearch"])
    parser.add_argument("--half-life", type=float, default=240.0,
                        help="recency half-life in days; 0 disables decay")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-n", "--limit", type=int, default=10)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p_eval.add_argument("-k", type=int, default=10)

    p_ablate = sub.add_parser("ablate")
    p_ablate.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p_ablate.add_argument("-k", type=int, default=10)

    p_crawl = sub.add_parser("crawl")
    p_crawl.add_argument("feeds", type=Path)
    p_crawl.add_argument("-o", "--out", type=Path, default=DEFAULT_CORPUS)

    args = parser.parse_args(argv)

    if args.command == "crawl":
        import json

        from src.feeds import fetch_all

        urls = json.loads(args.feeds.read_text())
        result, _ = fetch_all(urls)
        for url, error in result.failures.items():
            print(f"  FAILED  {url}: {error}", file=sys.stderr)
        count = save_corpus(result.documents, args.out)
        print(f"{count} documents from {len(urls) - len(result.failures)}/{len(urls)} feeds "
              f"-> {args.out}")
        return 0 if result.ok else 1

    if args.command == "ablate":
        from src.ablation import format_rows, run

        from src.ablation import sweep_title_weight

        queries = load_queries(args.queries)
        print(format_rows(run(args.corpus, queries, k=args.k), k=args.k))
        print("\ntitle weight sweep:")
        for weight, score in sweep_title_weight(args.corpus, queries, k=args.k):
            print(f"  {weight:>4.1f}  nDCG@{args.k} {score:.3f}")
        return 0

    service = make_service(args.corpus, args.backend, args.half_life)

    if args.command == "search":
        query = " ".join(args.query)
        hits = service.search(query, limit=args.limit)
        if not hits:
            print(f"no results for {query!r}")
            return 0
        print(f"{len(hits)} results for {query!r}\n")
        for position, hit in enumerate(hits, start=1):
            print(f"{position:2d}. {hit.document.title}")
            print(f"    {hit.document.url}")
            print(f"    score {hit.score:.3f}  "
                  f"(relevance {hit.relevance:.3f} x recency {hit.recency:.3f})")
        return 0

    if args.command == "evaluate":
        report = evaluate(service, load_queries(args.queries), k=args.k)
        print(report.format())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
