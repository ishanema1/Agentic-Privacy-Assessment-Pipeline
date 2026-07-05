"""
Tests for confluence_client.py.

Fully offline — the Confluence API is mocked via a fake `requests.Session`,
so no real credentials or network access are required to run this.

Run with: pytest test_confluence_client.py -v
"""

import pytest

from confluence_client import ConfluenceAuthError, ConfluenceClient


class FakeResponse:
    def __init__(self, status_code: int, json_data: dict, headers = None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """
    Minimal stand-in for requests.Session. `responses` is a list of
    FakeResponse objects returned in order, one per call to .get().
    """

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.headers: dict = {}
        self.auth = None
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        return self._responses.pop(0)


SAMPLE_STORAGE_HTML = (
    "<h1>Data Sharing Overview</h1>"
    "<p>This customer requests anonymized vehicle trajectory data.</p>"
    "<ul><li>Format: CSV</li><li>Frequency: daily batch</li></ul>"
)


def make_client(responses: list[FakeResponse]) -> ConfluenceClient:
    return ConfluenceClient(
        base_url="https://example.atlassian.net",
        email="pipeline-bot@example.com",
        api_token="fake-token-not-real",
        session=FakeSession(responses),
    )


def test_extract_text_strips_markup_and_preserves_structure():
    text = ConfluenceClient._extract_text(SAMPLE_STORAGE_HTML)
    assert "Data Sharing Overview" in text
    assert "<p>" not in text
    assert "Format: CSV" in text
    # Each block-level element should land on its own line.
    lines = text.splitlines()
    assert "Data Sharing Overview" in lines


def test_extract_text_handles_empty_input():
    assert ConfluenceClient._extract_text("") == ""
    assert ConfluenceClient._extract_text(None) == ""


def test_get_customer_docs_happy_path():
    responses = [
        FakeResponse(200, {"results": [{"id": "space-1"}]}),   # space lookup
        FakeResponse(200, {"results": [{"id": "page-1"}]}),    # page list in space
        FakeResponse(200, {                                     # page detail
            "title": "Architecture Overview",
            "version": {"number": 3},
            "_links": {"webui": "/spaces/CUST014/pages/page-1"},
            "body": {"storage": {"value": SAMPLE_STORAGE_HTML}},
        }),
    ]
    client = make_client(responses)

    docs = client.get_customer_docs(space_key="CUST014")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Architecture Overview"
    assert doc.version == 3
    assert "anonymized vehicle trajectory data" in doc.content
    assert doc.url == "https://example.atlassian.net/wiki/spaces/CUST014/pages/page-1"


def test_get_customer_docs_raises_for_unknown_space():
    responses = [FakeResponse(200, {"results": []})]
    client = make_client(responses)

    with pytest.raises(ValueError):
        client.get_customer_docs(space_key="DOES-NOT-EXIST")


def test_auth_error_raises_confluence_auth_error():
    responses = [FakeResponse(401, {})]
    client = make_client(responses)

    with pytest.raises(ConfluenceAuthError):
        client._get("/wiki/api/v2/spaces")


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr("confluence_client.time.sleep", lambda seconds: None)

    responses = [
        FakeResponse(429, {}, headers={"Retry-After": "1"}),
        FakeResponse(200, {"results": [{"id": "space-1"}]}),
    ]
    client = make_client(responses)

    result = client._get("/wiki/api/v2/spaces", params={"keys": "CUST014"})
    assert result["results"][0]["id"] == "space-1"


def test_as_context_block_includes_title_and_source_url():
    responses = [
        FakeResponse(200, {"results": [{"id": "space-1"}]}),
        FakeResponse(200, {"results": [{"id": "page-1"}]}),
        FakeResponse(200, {
            "title": "Data Spec",
            "version": {"number": 1},
            "_links": {"webui": "/spaces/CUST014/pages/page-1"},
            "body": {"storage": {"value": "<p>Some content.</p>"}},
        }),
    ]
    client = make_client(responses)
    doc = client.get_customer_docs(space_key="CUST014")[0]

    block = doc.as_context_block()
    assert "### Data Spec" in block
    assert "https://example.atlassian.net/wiki/spaces/CUST014/pages/page-1" in block
    assert "Some content." in block
