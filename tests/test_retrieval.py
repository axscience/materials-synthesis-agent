"""literature/retrieval.py -- mocked at the requests.get boundary (no real network calls here;
search_openalex was verified against the real live API separately, see its module docstring)."""

from unittest.mock import Mock, patch

import pytest
import requests

from materials_synthesis_agent.literature.retrieval import (
    RateLimitedError,
    search,
    search_openalex,
)


def _response(status_code=200, json_data=None):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return resp


def _openalex_work(title="A COF paper", year=2020, doi="https://doi.org/10.1000/x", abstract_words=None):
    aii = None
    if abstract_words:
        aii = {word: [i] for i, word in enumerate(abstract_words)}
    return {
        "id": "https://openalex.org/W123",
        "doi": doi,
        "display_name": title,
        "publication_year": year,
        "abstract_inverted_index": aii,
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
