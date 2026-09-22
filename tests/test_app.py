import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_reports_corpus_size(client):
    payload = client.get("/health").get_json()
    assert payload["status"] == "ok" and payload["documents"] > 0


def test_api_requires_a_query(client):
    assert client.get("/api/search").status_code == 400


def test_api_returns_scored_results(client):
    payload = client.get("/api/search?q=bm25").get_json()
    assert payload["count"] > 0
    first = payload["results"][0]
    assert {"title", "url", "score", "relevance", "recency"} <= set(first)


def test_api_rejects_a_non_integer_limit(client):
    assert client.get("/api/search?q=bm25&limit=many").status_code == 400


def test_api_clamps_an_oversized_limit(client):
    payload = client.get("/api/search?q=bm25&limit=9999").get_json()
    assert payload["count"] <= 50


def test_home_page_renders_results(client):
    body = client.get("/?q=bm25").get_data(as_text=True)
    assert "score" in body


def test_home_page_handles_no_results(client):
    body = client.get("/?q=zzzzzznotaword").get_data(as_text=True)
    assert "No results" in body
