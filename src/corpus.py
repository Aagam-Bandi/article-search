"""Loading documents from disk."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .documents import Document


def load_corpus(path: str | Path) -> list[Document]:
    raw = json.loads(Path(path).read_text())
    return [
        Document.build(
            title=item["title"],
            body=item["body"],
            url=item["url"],
            source=item.get("source", ""),
            published=(
                datetime.fromisoformat(item["published"])
                if item.get("published")
                else None
            ),
        )
        for item in raw
    ]


def save_corpus(documents: list[Document], path: str | Path) -> int:
    payload = [
        {
            "title": d.title,
            "body": d.body,
            "url": d.url,
            "source": d.source,
            "published": d.published.isoformat() if d.published else None,
        }
        for d in documents
    ]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return len(payload)
