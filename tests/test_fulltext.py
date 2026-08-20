"""literature/fulltext.py -- PDF parsing tested against a real, hand-built minimal PDF (not a
mock at the PDF-parsing layer), network mocked at requests.get. The end-to-end OA-fetch chain was
separately verified against real live papers while building this (see the module docstring and
CHANGELOG-equivalent commit message) -- a real Nature Communications COF paper's Experimental
Section, with genuine quantitative synthesis conditions, was correctly located and extracted.
"""

from unittest.mock import Mock, patch

import pytest

from materials_synthesis_agent.literature.fulltext import (
    extract_relevant_section,
    fetch_pdf_text,
    get_full_text_excerpt,
)
from materials_synthesis_agent.literature.retrieval import Paper


def _build_minimal_pdf(text_lines: list[str]) -> bytes:
    """A small, real, valid PDF -- hand-built rather than pulled in via a PDF-authoring dependency
    this package doesn't otherwise need. Exercises the real pypdf parsing path, not a mock of it."""
    content_ops = []
    y = 700
    for line in text_lines:
        esc = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_ops.append(f"BT /F1 12 Tf 50 {y} Td ({esc}) Tj ET")
        y -= 20
    content = "\n".join(content_ops).encode()

    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 800]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        f"<</Length {len(content)}>>stream\n".encode() + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj".encode() + obj + b"endobj\n"

    xref_start = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer<</Size {n}/Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF".encode()
    return bytes(out)


def _pdf_response(pdf_bytes: bytes, content_type: str = "application/pdf"):
    resp = Mock()
    resp.status_code = 200
    resp.content = pdf_bytes
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = Mock()
    return resp


# ---- extract_relevant_section (pure logic, no PDF/network involved) ----------------------------


def test_extract_relevant_section_finds_a_strict_line_anchored_heading():
    text = "Intro text here.\n\nExperimental\nSynthesized using 5 mmol reagent A."
    result = extract_relevant_section(text, max_chars=200)
    assert result.startswith("Experimental")
    assert "5 mmol reagent A" in result


def test_extract_relevant_section_prefers_the_last_strict_match():
    # A figure legend can itself land alone on a line and match the strict pattern (confirmed
    # against a real paper -- see the module docstring); the real heading is later.
    text = "Experimental\n(figure axis label)\n\nlots of results text\n\nExperimental\nReal recipe: 5 mmol, 120 C, 24 h."
    result = extract_relevant_section(text, max_chars=200)
    assert "Real recipe" in result


def test_extract_relevant_section_does_not_match_prose_for_the_strict_tier():
    text = "In this paper we describe the synthesis of MOFs and COFs broadly.\n\nExperimental Section\nReal recipe here."
    result = extract_relevant_section(text, max_chars=200)
    assert result.startswith("Experimental Section")


def test_extract_relevant_section_falls_back_to_loose_match():
    # No line-anchored heading anywhere -- extraction lost the line breaks. Loose tier should
    # still find "synthesis of X" in running text as a last resort before giving up entirely.
    text = "Intro. " + ("filler " * 50) + "The synthesis of TAPB-PDA proceeded as follows: 5 mmol, 120 C."
    result = extract_relevant_section(text, max_chars=200)
    assert "synthesis of TAPB-PDA" in result.lower() or "TAPB-PDA proceeded" in result


def test_extract_relevant_section_falls_back_to_leading_excerpt_when_nothing_matches():
    text = "A paper with no recognizable heading at all, just prose about porous materials."
    result = extract_relevant_section(text, max_chars=20)
    assert result == text[:20]


def test_extract_relevant_section_respects_max_chars():
    text = "Experimental\n" + ("x" * 500)
    result = extract_relevant_section(text, max_chars=50)
    assert len(result) <= 50


# ---- fetch_pdf_text (real PDF parsing, mocked network) --------------------------------------


def test_fetch_pdf_text_extracts_real_text_from_a_real_pdf():
    pdf_bytes = _build_minimal_pdf(["Experimental Section", "Synthesized using 5 mmol reagent A in THF."])
    with patch("materials_synthesis_agent.literature.fulltext.requests.get", return_value=_pdf_response(pdf_bytes)):
        text = fetch_pdf_text("https://example.org/paper.pdf")
    assert text is not None
    assert "Experimental Section" in text
    assert "5 mmol reagent A" in text


def test_fetch_pdf_text_returns_none_for_non_pdf_content_type():
    # A bot-challenge/paywall page returned as 200 text/html -- confirmed as a real failure mode
    # against a real publisher (RSC) while building this.
    resp = Mock()
    resp.status_code = 200
    resp.content = b"<html>Just a moment...</html>"
    resp.headers = {"content-type": "text/html; charset=UTF-8"}
    resp.raise_for_status = Mock()
    with patch("materials_synthesis_agent.literature.fulltext.requests.get", return_value=resp):
        text = fetch_pdf_text("https://example.org/blocked.pdf")
    assert text is None


def test_fetch_pdf_text_returns_none_on_network_error():
    import requests

    with patch("materials_synthesis_agent.literature.fulltext.requests.get", side_effect=requests.ConnectionError("no route")):
        text = fetch_pdf_text("https://example.org/paper.pdf")
    assert text is None


def test_fetch_pdf_text_returns_none_for_unparseable_pdf_bytes():
    resp = _pdf_response(b"not actually a pdf at all")
    with patch("materials_synthesis_agent.literature.fulltext.requests.get", return_value=resp):
        text = fetch_pdf_text("https://example.org/broken.pdf")
    assert text is None


def test_fetch_pdf_text_sends_a_browser_like_user_agent():
    # Confirmed necessary against a real publisher host (Nature/Springer 403'd with no UA,
    # succeeded with one) while building this.
    pdf_bytes = _build_minimal_pdf(["hello"])
    with patch("materials_synthesis_agent.literature.fulltext.requests.get", return_value=_pdf_response(pdf_bytes)) as mock_get:
        fetch_pdf_text("https://example.org/paper.pdf")
    assert "User-Agent" in mock_get.call_args.kwargs["headers"]


# ---- get_full_text_excerpt (orchestration) ----------------------------------------------------


def _paper(**overrides):
    base = dict(source_id="10.1000/x", title="t", abstract="a", year=2020, url="u", source="openalex", oa_pdf_url=None)
    base.update(overrides)
    return Paper(**base)


def test_get_full_text_excerpt_returns_none_with_no_oa_pdf_url_and_no_doi_fallback():
    paper = _paper(source_id="not-a-doi", oa_pdf_url=None)
    assert get_full_text_excerpt(paper) is None


def test_get_full_text_excerpt_uses_the_papers_own_oa_pdf_url():
    pdf_bytes = _build_minimal_pdf(["Experimental Section", "Real recipe: 5 mmol, 120 C, 24 h."])
    paper = _paper(oa_pdf_url="https://example.org/real.pdf")
    with patch("materials_synthesis_agent.literature.fulltext.requests.get", return_value=_pdf_response(pdf_bytes)):
        excerpt = get_full_text_excerpt(paper)
    assert excerpt is not None
    assert "Real recipe" in excerpt


def test_get_full_text_excerpt_falls_back_to_unpaywall_when_paper_has_doi_but_no_oa_pdf_url():
    # Unpaywall resolution (retrieval.py) and PDF fetching (fulltext.py) each import `requests`
    # independently, so both call sites need patching, not just one.
    pdf_bytes = _build_minimal_pdf(["Experimental Section", "Recipe from Unpaywall-resolved PDF."])
    unpaywall_resp = Mock()
    unpaywall_resp.status_code = 200
    unpaywall_resp.json.return_value = {"best_oa_location": {"url_for_pdf": "https://example.org/unpaywall.pdf"}}
    unpaywall_resp.raise_for_status = Mock()

    paper = _paper(source_id="10.1000/real-doi", oa_pdf_url=None)
    with patch("materials_synthesis_agent.literature.retrieval.requests.get", return_value=unpaywall_resp):
        with patch("materials_synthesis_agent.literature.fulltext.requests.get", return_value=_pdf_response(pdf_bytes)):
            excerpt = get_full_text_excerpt(paper, unpaywall_email="real@example.org")
    assert excerpt is not None
    assert "Unpaywall-resolved" in excerpt


def test_get_full_text_excerpt_returns_none_when_pdf_fetch_fails():
    paper = _paper(oa_pdf_url="https://example.org/gone.pdf")
    import requests

    with patch("materials_synthesis_agent.literature.fulltext.requests.get", side_effect=requests.ConnectionError()):
        assert get_full_text_excerpt(paper) is None
