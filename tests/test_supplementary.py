"""Tests for supplementary-information (SI) discovery and extraction in literature/fulltext.py.
The SI is where a COF's exact conditions usually live, and is often open even when the article is
paywalled -- these check the landing-page scrape and the SI-first combined excerpt, without hitting
the network."""

from types import SimpleNamespace

import materials_synthesis_agent.literature.fulltext as ft
from materials_synthesis_agent.literature.retrieval import Paper


def _fake_landing(monkeypatch, html):
    def fake_get(url, timeout_s=None, timeout=None, headers=None):
        return SimpleNamespace(
            status_code=200, text=html, url="https://pubs.example.org/doi/10.1021/x",
            headers={"content-type": "text/html"}, raise_for_status=lambda: None,
        )
    monkeypatch.setattr(ft.requests, "get", fake_get)


def test_resolve_si_urls_finds_common_publisher_patterns(monkeypatch):
    html = """
    <a href="/doi/suppl/10.1021/x/suppl_file/x_si_001.pdf">Supporting Information</a>
    <a href="/action/downloadSupplement?doi=10.1002/y&file=y-sup-0001.pdf">SI</a>
    <a href="https://static-content.springer.com/esm/art%3A10.1038/z/MediaObjects/z_MOESM1_ESM.pdf">Supp</a>
    <a href="/normal/article.pdf">Full article (NOT si)</a>
    """
    _fake_landing(monkeypatch, html)
    urls = ft.resolve_si_pdf_urls("10.1021/x")
    assert len(urls) == 3
    assert not any("normal/article.pdf" in u for u in urls)
    assert all(u.startswith("http") for u in urls)  # resolved to absolute


def test_resolve_si_urls_orders_pdfs_before_query_download_links(monkeypatch):
    html = """
    <a href="/action/downloadSupplement?doi=10.1002/y&file=y.pdf">query-style</a>
    <a href="/suppl/direct_si.pdf">direct pdf</a>
    """
    _fake_landing(monkeypatch, html)
    urls = ft.resolve_si_pdf_urls("10.1002/y")
    assert urls[0].endswith("direct_si.pdf")  # a plain .pdf ranks ahead of the ?query download


def test_resolve_si_urls_returns_empty_on_non_html(monkeypatch):
    def fake_get(url, **k):
        return SimpleNamespace(status_code=200, text="%PDF-1.5", url=url,
                               headers={"content-type": "application/pdf"}, raise_for_status=lambda: None)
    monkeypatch.setattr(ft.requests, "get", fake_get)
    assert ft.resolve_si_pdf_urls("10.1021/x") == []


def test_full_text_excerpt_puts_supplementary_first(monkeypatch):
    paper = Paper(source_id="10.1021/x", title="t", abstract="abs", year=2024,
                  url="http://x", source="openalex", oa_pdf_url="http://x/main.pdf")
    monkeypatch.setattr(ft, "get_supplementary_excerpt", lambda p, max_chars=8000: "SI: TAPB 30 mg, PDA 20 mg, 120 C, 72 h")
    monkeypatch.setattr(ft, "fetch_pdf_text", lambda url, timeout_s=30.0: "MAIN BODY ... Experimental Section ... methods")
    monkeypatch.setattr(ft, "extract_relevant_section", lambda text, max_chars=8000: "MAIN: experimental methods")
    out = ft.get_full_text_excerpt(paper)
    assert "SUPPLEMENTARY INFORMATION" in out
    assert out.index("SUPPLEMENTARY INFORMATION") < out.index("MAIN TEXT")  # SI first


def test_full_text_excerpt_returns_none_when_nothing_available(monkeypatch):
    paper = Paper(source_id="nodoi", title="t", abstract="abs", year=2024, url="", source="arxiv", oa_pdf_url=None)
    monkeypatch.setattr(ft, "get_supplementary_excerpt", lambda p, max_chars=8000: None)
    assert ft.get_full_text_excerpt(paper) is None


def test_supplementary_excerpt_none_when_no_si_links(monkeypatch):
    paper = Paper(source_id="10.1021/x", title="t", abstract="a", year=2024, url="http://x", source="openalex")
    monkeypatch.setattr(ft, "resolve_si_pdf_urls", lambda landing, timeout_s=30.0: [])
    assert ft.get_supplementary_excerpt(paper) is None
