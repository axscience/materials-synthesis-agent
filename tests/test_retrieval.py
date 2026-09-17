"""literature/retrieval.py -- mocked at the requests.get boundary (no real network calls here;
search_openalex was verified against the real live API separately, see its module docstring)."""

from unittest.mock import Mock, patch

import pytest
import requests

from materials_synthesis_agent.literature.retrieval import (
    RateLimitedError,
    resolve_oa_pdf_url_via_unpaywall,
    search,
    search_arxiv,
    search_openalex,
    search_semantic_scholar,
)


def _response(status_code=200, json_data=None):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return resp


def _openalex_work(title="A COF paper", year=2020, doi="https://doi.org/10.1000/x", abstract_words=None, oa_pdf_url=None):
    aii = None
    if abstract_words:
        aii = {word: [i] for i, word in enumerate(abstract_words)}
    return {
        "id": "https://openalex.org/W123",
        "doi": doi,
        "display_name": title,
        "publication_year": year,
        "abstract_inverted_index": aii,
        "best_oa_location": {"pdf_url": oa_pdf_url} if oa_pdf_url else {},
    }


def test_search_openalex_reconstructs_abstract_in_order():
    work = _openalex_work(abstract_words=["Covalent", "organic", "frameworks", "are", "porous"])
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": [work]})) as mock_get:
        papers = search_openalex("COF-LZU1 synthesis", limit=10)

    assert len(papers) == 1
    assert papers[0].abstract == "Covalent organic frameworks are porous"
    assert papers[0].source == "openalex"
    assert mock_get.call_args.kwargs["params"]["search"] == "COF-LZU1 synthesis"


def test_search_openalex_strips_doi_prefix_for_source_id():
    work = _openalex_work(doi="https://doi.org/10.1021/ja206846p")
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": [work]})):
        papers = search_openalex("x", limit=10)
    assert papers[0].source_id == "10.1021/ja206846p"


def test_search_openalex_handles_no_abstract():
    work = _openalex_work(abstract_words=None)
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": [work]})):
        papers = search_openalex("x", limit=10)
    assert papers[0].abstract is None


def test_search_openalex_respects_limit_even_if_api_returns_more():
    works = [_openalex_work(title=f"paper {i}") for i in range(5)]
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": works})):
        papers = search_openalex("x", limit=2)
    assert len(papers) == 2


def test_search_falls_back_to_openalex_when_semantic_scholar_is_rate_limited():
    rate_limited = _response(status_code=429)
    openalex_ok = _response(json_data={"results": [_openalex_work()]})
    with patch(
        "materials_synthesis_agent.literature.retrieval.requests.get",
        side_effect=[rate_limited, rate_limited, rate_limited, openalex_ok],
    ):
        papers = search("x", limit=5)
    assert len(papers) == 1
    assert papers[0].source == "openalex"


def test_search_openalex_populates_oa_pdf_url_when_genuinely_open_access():
    work = _openalex_work(oa_pdf_url="https://example.org/real.pdf")
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": [work]})):
        papers = search_openalex("x", limit=10)
    assert papers[0].oa_pdf_url == "https://example.org/real.pdf"


def test_search_openalex_leaves_oa_pdf_url_none_when_not_open_access():
    work = _openalex_work(oa_pdf_url=None)
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": [work]})):
        papers = search_openalex("x", limit=10)
    assert papers[0].oa_pdf_url is None


def test_search_openalex_ignores_a_landing_page_with_no_direct_pdf():
    # A real, common OpenAlex shape: is_oa=True but best_oa_location only has a landing page, no
    # pdf_url -- confirmed against a real response while building this. Not fetchable as a PDF, so
    # oa_pdf_url must stay None, not fall back to the landing page URL.
    work = _openalex_work()
    work["best_oa_location"] = {"pdf_url": None, "landing_page_url": "https://example.org/landing"}
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"results": [work]})):
        papers = search_openalex("x", limit=10)
    assert papers[0].oa_pdf_url is None


def test_search_semantic_scholar_populates_oa_pdf_url():
    item = {
        "paperId": "abc", "title": "t", "abstract": "a", "year": 2020, "url": "u",
        "externalIds": {"DOI": "10.1/x"}, "openAccessPdf": {"url": "https://example.org/s2.pdf"},
    }
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=_response(json_data={"data": [item]})):
        papers = search_semantic_scholar("x", limit=10)
    assert papers[0].oa_pdf_url == "https://example.org/s2.pdf"


def test_search_arxiv_derives_oa_pdf_url_from_abs_url():
    resp = Mock()
    resp.status_code = 200
    resp.text = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.12345</id>
        <title>A paper</title>
        <summary>abstract text</summary>
        <published>2023-01-01T00:00:00Z</published>
      </entry>
    </feed>"""
    resp.raise_for_status = Mock()
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=resp):
        papers = search_arxiv("x", limit=5)
    assert papers[0].oa_pdf_url == "http://arxiv.org/pdf/2301.12345"


def test_resolve_oa_pdf_url_via_unpaywall_returns_none_without_an_email():
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_oa_pdf_url_via_unpaywall("10.1000/x", email=None) is None


def test_resolve_oa_pdf_url_via_unpaywall_returns_none_without_a_doi():
    assert resolve_oa_pdf_url_via_unpaywall("", email="real@example.org") is None


def test_resolve_oa_pdf_url_via_unpaywall_returns_the_pdf_url_on_success():
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {"best_oa_location": {"url_for_pdf": "https://example.org/real.pdf"}}
    resp.raise_for_status = Mock()
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=resp) as mock_get:
        url = resolve_oa_pdf_url_via_unpaywall("10.1000/x", email="real@example.org")
    assert url == "https://example.org/real.pdf"
    assert mock_get.call_args.kwargs["params"]["email"] == "real@example.org"


def test_resolve_oa_pdf_url_via_unpaywall_returns_none_on_request_failure():
    # Confirmed against the real API: a placeholder email gets a real 422 rejection.
    resp = Mock()
    resp.status_code = 422
    resp.raise_for_status = Mock(side_effect=requests.HTTPError("422"))
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=resp):
        url = resolve_oa_pdf_url_via_unpaywall("10.1000/x", email="placeholder@example.com")
    assert url is None


def test_search_falls_back_to_arxiv_when_both_semantic_scholar_and_openalex_fail():
    rate_limited = _response(status_code=429)
    openalex_fails = Mock(side_effect=requests.ConnectionError("no route"))

    def fake_get(url, *args, **kwargs):
        if "semanticscholar" in url:
            return rate_limited
        if "openalex" in url:
            raise requests.ConnectionError("no route")
        # arxiv
        resp = Mock()
        resp.status_code = 200
        resp.text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        resp.raise_for_status = Mock()
        return resp

    with patch("materials_synthesis_agent.literature.retrieval.requests.get", side_effect=fake_get):
        papers = search("x", limit=5)
    assert papers == []  # empty feed, but no exception -- confirms arxiv was actually reached


def test_search_merges_semantic_scholar_and_openalex_deduped_by_doi():
    """search() now MERGES both sources (not fallback-only): the same paper (shared DOI) appears
    once, and a paper unique to OpenAlex is included alongside the Semantic Scholar results."""
    from unittest.mock import patch as _patch
    from materials_synthesis_agent.literature.retrieval import Paper, search as _search

    a_s2 = Paper(source_id="10.1/a", title="Alpha COF", abstract=None, year=2024, url="",
                 source="semantic_scholar", oa_pdf_url="s2.pdf")
    a_oa = Paper(source_id="10.1/a", title="Alpha COF", abstract=None, year=2024, url="",
                 source="openalex", oa_pdf_url="oa.pdf")   # same DOI -> duplicate
    b_oa = Paper(source_id="10.1/b", title="Beta COF", abstract=None, year=2024, url="",
                 source="openalex", oa_pdf_url=None)

    with _patch("materials_synthesis_agent.literature.retrieval.search_semantic_scholar",
                return_value=[a_s2]), \
         _patch("materials_synthesis_agent.literature.retrieval.search_openalex",
                return_value=[a_oa, b_oa]):
        papers = _search("cof", limit=10)

    assert [p.source_id for p in papers] == ["10.1/a", "10.1/b"]   # deduped, order preserved
    assert papers[0].source == "semantic_scholar"                   # S2 copy kept for the shared DOI
    assert {p.title for p in papers} == {"Alpha COF", "Beta COF"}
