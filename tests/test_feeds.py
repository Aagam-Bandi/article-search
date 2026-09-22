import io

from src.feeds import FeedState, fetch_all, fetch_feed, parse_feed

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>First Post</title>
    <description>&lt;p&gt;Body of the first post&lt;/p&gt;</description>
    <link>https://example.com/1</link>
    <pubDate>Mon, 01 Sep 2025 10:00:00 +0000</pubDate>
  </item>
  <item>
    <title>Second Post</title>
    <description>Body of the second</description>
    <link>https://example.com/2</link>
  </item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Atom Post</title>
    <summary>Atom body text</summary>
    <link rel="alternate" href="https://example.org/a"/>
    <published>2025-09-01T10:00:00Z</published>
  </entry>
</feed>"""

EMPTY_ENTRY = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><link>https://example.com/blank</link></item>
  <item><title>Real</title><description>text</description><link>https://e.com/r</link></item>
</channel></rss>"""


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body = body.encode()
        self.status = status
        self.headers = headers or {}

    def read(self):
        return self._body


def test_parses_rss():
    docs = parse_feed(RSS, source="s")
    assert [d.title for d in docs] == ["First Post", "Second Post"]


def test_strips_html_from_rss_descriptions():
    assert parse_feed(RSS, source="s")[0].body == "Body of the first post"


def test_parses_rfc822_dates():
    assert parse_feed(RSS, source="s")[0].published.year == 2025


def test_missing_date_is_none_rather_than_an_error():
    assert parse_feed(RSS, source="s")[1].published is None


def test_parses_atom_including_link_href():
    doc = parse_feed(ATOM, source="s")[0]
    assert (doc.title, doc.url) == ("Atom Post", "https://example.org/a")


def test_entries_with_neither_title_nor_body_are_skipped():
    """An empty document matches nothing but still inflates the IDF corpus size."""
    assert [d.title for d in parse_feed(EMPTY_ENTRY, source="s")] == ["Real"]


def test_conditional_get_sends_cached_validators():
    seen = {}

    def opener(request, timeout=None):
        seen.update(request.headers)
        return FakeResponse(RSS)

    fetch_feed("https://e.com/f", state=FeedState(etag='"abc"'), opener=opener)
    assert seen.get("If-none-match") == '"abc"'


def test_304_yields_no_documents_and_keeps_state():
    state = FeedState(etag='"abc"')
    docs, new_state, unchanged = fetch_feed(
        "https://e.com/f", state=state,
        opener=lambda request, timeout=None: FakeResponse("", status=304),
    )
    assert (docs, unchanged, new_state.etag) == ([], True, '"abc"')


def test_new_validators_are_captured_for_the_next_fetch():
    _, state, _ = fetch_feed(
        "https://e.com/f",
        opener=lambda request, timeout=None: FakeResponse(RSS, headers={"ETag": '"z"'}),
    )
    assert state.etag == '"z"'


def test_one_broken_feed_does_not_stop_the_crawl():
    """Regression: the crawl must be resilient to individual feed failures."""

    def opener(request, timeout=None):
        if "bad" in request.full_url:
            raise OSError("connection reset")
        return FakeResponse(RSS)

    result, _ = fetch_all(
        ["https://good.com/f", "https://bad.com/f", "https://good2.com/f"], opener=opener
    )
    assert len(result.documents) == 4          # two feeds x two items
    assert list(result.failures) == ["https://bad.com/f"]
    assert result.ok is False


def test_malformed_xml_is_reported_as_a_failure_not_raised():
    result, _ = fetch_all(
        ["https://e.com/f"],
        opener=lambda request, timeout=None: FakeResponse("<not xml"),
    )
    assert "https://e.com/f" in result.failures
