"""RSS/Atom ingestion.

Three properties this module is built around, each of which cost the naive
version something:

1. **Per-feed failure isolation.** Feeds break constantly — DNS failures,
   expired certificates, 500s, XML that doesn't parse. A crawl that raises on
   the first bad feed indexes nothing; `fetch_all` records the failure and
   keeps going, and the caller gets both the documents and the failures.

2. **Conditional GET.** Feeds are re-fetched often and change rarely. Sending
   back the ETag/Last-Modified from the previous fetch lets the server answer
   304 with no body, which is both faster and the polite way to poll someone
   else's server repeatedly.

3. **Dedup by content, not URL.** Handled in `Document.build` via the content
   hash — the same post syndicated to three URLs collapses to one document.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .documents import Document

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


@dataclass
class FeedState:
    """Cache validators from the previous fetch of one feed."""

    etag: str | None = None
    last_modified: str | None = None


@dataclass
class CrawlResult:
    documents: list[Document] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _text(element, *paths: str) -> str:
    for path in paths:
        found = element.find(path, NS)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
        if found is not None and found.get("href"):
            return found.get("href").strip()
    return ""


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    try:  # RFC 822, as used by RSS
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        pass
    try:  # ISO 8601, as used by Atom
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_feed(xml_text: str, *, source: str) -> list[Document]:
    """Parse RSS 2.0 or Atom into Documents.

    Entries missing both a title and a body are skipped rather than indexed
    as empty documents — an empty document matches nothing but still inflates
    the corpus size that IDF is computed against.
    """
    root = ET.fromstring(xml_text)
    entries = root.findall(".//item") or root.findall(".//atom:entry", NS)
    documents = []
    for entry in entries:
        title = _text(entry, "title", "atom:title")
        body = _text(
            entry,
            "content:encoded",
            "description",
            "atom:content",
            "atom:summary",
        )
        url = _text(entry, "link", "atom:link[@rel='alternate']", "atom:link")
        published = _parse_date(
            _text(entry, "pubDate", "atom:published", "atom:updated")
        )
        if not (title or body):
            continue
        documents.append(
            Document.build(
                title=title, body=body, url=url, source=source, published=published
            )
        )
    return documents


def fetch_feed(url: str, *, state: FeedState | None = None, opener=None, timeout=15):
    """Fetch one feed with conditional-GET headers.

    Returns (documents, new_state, unchanged). `opener` is injected so tests
    exercise this without network access.
    """
    import urllib.request

    state = state or FeedState()
    headers = {"User-Agent": "article-search/1.0 (+https://github.com/Aagam-Bandi)"}
    if state.etag:
        headers["If-None-Match"] = state.etag
    if state.last_modified:
        headers["If-Modified-Since"] = state.last_modified

    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers=headers)
    try:
        response = opener(request, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not swallowed
        if getattr(exc, "code", None) == 304:
            return [], state, True
        raise

    status = getattr(response, "status", 200)
    if status == 304:
        return [], state, True

    payload = response.read()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    new_state = FeedState(
        etag=response.headers.get("ETag") if hasattr(response, "headers") else None,
        last_modified=(
            response.headers.get("Last-Modified") if hasattr(response, "headers") else None
        ),
    )
    return parse_feed(payload, source=url), new_state, False


def fetch_all(urls, *, states=None, opener=None) -> tuple[CrawlResult, dict[str, FeedState]]:
    """Crawl every feed, isolating failures per feed."""
    states = dict(states or {})
    result = CrawlResult()
    for url in urls:
        try:
            documents, new_state, unchanged = fetch_feed(
                url, state=states.get(url), opener=opener
            )
        except Exception as exc:  # noqa: BLE001 - one broken feed must not stop the crawl
            result.failures[url] = f"{type(exc).__name__}: {exc}"
            continue
        states[url] = new_state
        if unchanged:
            result.unchanged.append(url)
        result.documents.extend(documents)
    return result, states
