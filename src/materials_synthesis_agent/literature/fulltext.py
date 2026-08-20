"""Full-text retrieval: given a `Paper` with a real, open-access `oa_pdf_url`, fetch the PDF,
extract its text, and cut it down to the section that actually has synthesis conditions in it.

Legal/practical scope, stated plainly: this only ever fetches a PDF a source API has already
confirmed is open access (`Paper.oa_pdf_url` -- see retrieval.py). A paywalled paper has no
`oa_pdf_url` and this module never tries to work around that. In practice, a large fraction of COF
chemistry literature is paywalled -- confirmed directly against real OpenAlex results, not assumed
(2 of 3 real papers checked while building this had `is_oa: False`) -- so full text is a real
upgrade for *some* papers, not a universal replacement for abstract-only extraction. And even for
an open-access main text, the exact molar ratios/temperature/time are very often in a separate
Supplementary Information PDF that isn't covered by the same OA grant and often isn't linked from
any of these APIs at all -- a real ceiling this module can't engineer around.

Every function here returns None on failure (no PDF, download error, unparseable PDF, no matching
section) rather than raising -- a caller always has abstract-only extraction to fall back to, and
"couldn't get full text" is a normal, expected outcome, not an error.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Optional

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Headings that mark where a chemistry paper's actual synthesis conditions live. Two tiers, tried
# in order:
#  1. Line-anchored -- the heading stands alone on its own line, the way a real section title
#     survives PDF text extraction. Confirmed directly against a real paper's extracted text
#     (a Nature Communications COF paper): "Experimental" and "Methods" both appear exactly this
#     way. High precision -- this is very unlikely to match running prose.
#  2. Loose, anywhere-in-text -- a fallback for PDFs whose extracted text lost the line breaks that
#     would let tier 1 work. Lower precision: "synthesis of X" legitimately appears in running
#     prose too (confirmed the hard way -- an earlier version of this list matched "synthesis of
#     MOFs" in an introduction paragraph, not a real heading), so it's included only as a last
#     resort before falling back to a leading excerpt, not tried first.
_SECTION_HEADINGS_STRICT = [
    r"(?m)^\s*experimental\s+section\s*$",
    r"(?m)^\s*materials\s+and\s+methods\s*$",
    r"(?m)^\s*experimental\s+procedures?\s*$",
    r"(?m)^\s*experimental\s*$",
    r"(?m)^\s*methods\s*$",
]
_SECTION_HEADINGS_LOOSE = [
    r"experimental\s+section",
    r"materials\s+and\s+methods",
    r"experimental\s+procedures?",
    r"synthesis\s+of\s+",
]


def fetch_pdf_text(url: str, timeout_s: float = 30.0) -> Optional[str]:
    """Downloads a PDF and extracts its text. Returns None on any failure (network error, not
    actually a PDF, encrypted/unparseable PDF) -- never raises, since a full-text fetch failing is
    a normal fallback-to-abstract case, not something that should crash suggest-protocols."""
    # Confirmed directly against real publisher hosts: some (RSC) return a Cloudflare bot-challenge
    # page (403, text/html) to a request with no User-Agent, even for a genuinely open-access PDF;
    # a plain browser-like UA is enough to get the real PDF from those that don't hard-block
    # scripted requests outright (Nature/Springer worked; RSC still 403s regardless -- that's a
    # real, unresolved gap, not a bug in this function).
    headers = {"Accept": "application/pdf", "User-Agent": "Mozilla/5.0 (compatible; materials-synthesis-agent/0.3)"}
    try:
        resp = requests.get(url, timeout=timeout_s, headers=headers)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    if "pdf" not in resp.headers.get("content-type", "").lower():
        # A 200 with an HTML body (a paywall/consent/bot-challenge page dressed as success) is not
        # a PDF -- checked explicitly rather than handing arbitrary HTML to the PDF parser and
        # hoping it fails loudly.
        return None

    try:
        reader = PdfReader(BytesIO(resp.content))
        if reader.is_encrypted:
            return None
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError):
        return None

    return text or None


def extract_relevant_section(full_text: str, max_chars: int = 8000) -> str:
    """Finds the synthesis-relevant section of a full paper's text via a heading search, rather
    than handing the whole paper to the extraction prompt -- cheaper, and it doesn't give the model
    thousands of words of unrelated introduction/results/discussion to plausibly (and wrongly) pull
    a "matching" number from.

    For the strict tier, the LAST line-anchored match wins, not the first -- confirmed directly
    against a real paper's extracted text that this matters: a PXRD figure's axis/legend labels
    ("Experimental", "Refined", "Simulated", "Difference") can themselves land alone on their own
    extracted line, matching the strict pattern well before the real Methods/Experimental Section
    heading. The real heading is a single, structurally late occurrence (Methods conventionally
    follows Results & Discussion in this journal family); repeated figure-legend text tends to
    cluster earlier, near the results figures. This is a heuristic tuned against one real
    confirmed case, not a guaranteed-correct rule for every journal's layout -- a paper that puts
    its real Experimental Section early (common in some ACS/RSC formatting) can still be missed by
    "prefer last," which is exactly why the loose tier and the leading-excerpt fallback exist.

    Falls back to the same phrases matched anywhere in the text, then a leading excerpt (still
    bounded by max_chars), if no strict heading is found -- a paper with no recognized heading at
    all still has real text worth trying."""
    all_strict_matches = [m for pattern in _SECTION_HEADINGS_STRICT for m in re.finditer(pattern, full_text, re.IGNORECASE)]
    if all_strict_matches:
        match = max(all_strict_matches, key=lambda m: m.start())
        return full_text[match.start() : match.start() + max_chars].strip()

    for pattern in _SECTION_HEADINGS_LOOSE:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return full_text[match.start() : match.start() + max_chars].strip()
    return full_text[:max_chars].strip()


def get_full_text_excerpt(paper, max_chars: int = 8000, unpaywall_email: Optional[str] = None) -> Optional[str]:
    """End-to-end: resolve an OA PDF (the paper's own `oa_pdf_url`, or a live Unpaywall lookup as a
    second try when the paper has a DOI but no `oa_pdf_url` already), fetch it, extract text, and
    return the synthesis-relevant excerpt. Returns None at the first point nothing is available --
    caller falls back to `paper.abstract`, same as if this function were never called."""
    from materials_synthesis_agent.literature.retrieval import resolve_oa_pdf_url_via_unpaywall

    pdf_url = paper.oa_pdf_url
    if not pdf_url and paper.source_id and "/" in paper.source_id:  # a DOI-shaped source_id
        pdf_url = resolve_oa_pdf_url_via_unpaywall(paper.source_id, email=unpaywall_email)
    if not pdf_url:
        return None

    full_text = fetch_pdf_text(pdf_url)
    if not full_text:
        return None

    return extract_relevant_section(full_text, max_chars=max_chars)
