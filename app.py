#!/usr/bin/env python3
"""Flask search API and a single-page UI.

The index is built once at startup and held in memory. That is a deliberate
constraint rather than an oversight: at the corpus sizes this is built for
(see the benchmark in the README) an in-process index answers in single-digit
milliseconds, and the alternative — a network hop to a search cluster per
request — would dominate the latency budget for no benefit.

The point at which that stops being true is documented in the README's
Limitations section, along with what to do instead.
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from src.corpus import load_corpus
from src.index import build_index
from src.search import SearchService

ROOT = Path(__file__).resolve().parent

PAGE = """<!doctype html>
<title>article search</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 15px/1.55 system-ui, sans-serif; max-width: 46rem; margin: 3rem auto;
        padding: 0 1rem; }
 form { display: flex; gap: .5rem; margin-bottom: 1.5rem; }
 input { flex: 1; padding: .6rem .8rem; font-size: 1rem; border: 1px solid #8887;
         border-radius: 6px; background: transparent; color: inherit; }
 button { padding: .6rem 1.1rem; font-size: 1rem; border-radius: 6px;
          border: 1px solid #8887; background: transparent; color: inherit;
          cursor: pointer; }
 li { margin-bottom: 1.1rem; list-style: none; }
 ul { padding: 0; }
 .meta { font-size: .8rem; opacity: .65; }
 .empty { opacity: .65; }
</style>
<h1>article search</h1>
<form action="/" method="get">
  <input name="q" value="{{ query|e }}" placeholder="search {{ size }} articles"
         autofocus>
  <button type="submit">search</button>
</form>
{% if query %}
  {% if hits %}
  <ul>
  {% for hit in hits %}
    <li>
      <a href="{{ hit.document.url }}">{{ hit.document.title }}</a>
      <div class="meta">
        {{ hit.document.source }} &middot; score {{ '%.2f'|format(hit.score) }}
        (relevance {{ '%.2f'|format(hit.relevance) }} &times;
         recency {{ '%.2f'|format(hit.recency) }})
      </div>
    </li>
  {% endfor %}
  </ul>
  {% else %}
  <p class="empty">No results for &ldquo;{{ query|e }}&rdquo;.</p>
  {% endif %}
{% endif %}
"""


def create_app(corpus_path: str | Path | None = None, backend: str | None = None) -> Flask:
    corpus_path = corpus_path or os.environ.get("CORPUS", ROOT / "data" / "corpus.json")
    backend = backend or os.environ.get("BACKEND", "bm25")

    index = build_index(backend)
    index.add(load_corpus(corpus_path))
    service = SearchService(index)

    app = Flask(__name__)
    app.config["SERVICE"] = service

    @app.get("/")
    def home():
        query = request.args.get("q", "").strip()
        hits = service.search(query, limit=10) if query else []
        return render_template_string(PAGE, query=query, hits=hits, size=len(index))

    @app.get("/api/search")
    def api_search():
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify(error="q parameter is required"), 400
        try:
            limit = min(max(int(request.args.get("limit", 10)), 1), 50)
        except ValueError:
            return jsonify(error="limit must be an integer"), 400
        hits = service.search(query, limit=limit)
        return jsonify(
            query=query,
            count=len(hits),
            results=[
                {
                    "title": hit.document.title,
                    "url": hit.document.url,
                    "source": hit.document.source,
                    "published": (
                        hit.document.published.isoformat()
                        if hit.document.published else None
                    ),
                    "score": round(hit.score, 4),
                    "relevance": round(hit.relevance, 4),
                    "recency": round(hit.recency, 4),
                }
                for hit in hits
            ],
        )

    @app.get("/health")
    def health():
        return jsonify(status="ok", documents=len(index), backend=backend)

    return app


if __name__ == "__main__":
    create_app().run(port=int(os.environ.get("PORT", 5000)), debug=False)
