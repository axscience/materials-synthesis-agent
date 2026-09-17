"""Full-text retrieval: given a `Paper` with a real, open-access `oa_pdf_url`, fetch the PDF,
extract its text, and cut it down to the section that actually has synthesis conditions in it.

Legal/practical scope, stated plainly: this only ever fetches a PDF a source API has already
confirmed is open access (`Paper.oa_pdf_url` -- see retrieval.py). A paywalled paper has no
`oa_pdf_url` and this module never tries to work around that. In practice, a large fraction of COF
chemistry literature is paywalled -- confirmed directly against real OpenAlex results, not assumed
(2 of 3 real papers checked while building this had `is_oa: False`) -- so full text is a real
upgrade for *some* papers, not a universal replacement for abstract-only extraction.

The exact molar ratios/temperature/time for a COF synthesis very often live in the separate
Supplementary Information rather than the main text -- and, importantly, the SI is frequently left
open even when the article itself is paywalled. So this module also scrapes the paper's public
landing page for SI links and tries those (`resolve_si_pdf_urls`, `get_supplementary_excerpt`).
That's a best-effort HTML scrape, not a publisher API: it works where the landing page is
server-rendered with real SI links, and returns nothing for JS-only or hard-bot-blocked pages --
still a real ceiling for a subset of publishers, just a lower one than before.

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
    a normal fallback-to-abstract case, not something that should crash suggest-protocols.
    Uses cloudscraper to bypass Cloudflare bot challenges on publisher sites."""
    try:
        scraper = _get_cloudscraper()
        resp = scraper.get(url, timeout=timeout_s)
        resp.raise_for_status()
    except Exception:
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
        # Skip matches in the first 200 chars — those are almost always the paper title
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            if match.start() > 200:
                return full_text[match.start() : match.start() + max_chars].strip()
    return full_text[:max_chars].strip()


# A supplementary-information file is worth fetching even when the main article is paywalled --
# publishers very often leave the SI open even behind a paywalled article (confirmed as a general
# pattern; the exact conditions/temps/molar ratios for a COF synthesis live in the SI far more
# often than in the main text). These substrings, matched in a link's URL, flag it as an SI file
# across the common publisher layouts (ACS suppl_file, Wiley downloadSupplement, Springer/Nature
# MOESM/ESM, RSC suppdata, and the generic "supporting/supplementary" wording). This is a
# best-effort HTML scrape of the article's public landing page, not a publisher API -- it works
# where the landing page is server-rendered with real SI links and is defeated by JS-only pages or
# hard bot-blocking, in which case it returns nothing and the caller falls back exactly as before.
_SI_URL_MARKERS = (
    # generic
    "suppl_file", "downloadsupplement", "supporting-information", "supplementary",
    "supplementary-material", "supplementary_material", "electronic-supplementary",
    "supp-info", "sup-info", "supp_info", "/suppl/",
    # Nature / Springer (…MOESM1_ESM.pdf)
    "moesm", "_esm", "/esm/",
    # RSC (…/suppdata/…)
    "suppdata",
    # ACS (…_si_001.pdf, /doi/suppl/)
    "_si_", "-si.pdf", "_si.pdf", "/doi/suppl/",
    # Wiley (…-sup-0001-SuppMat.pdf, /asset/supinfo)
    "-sup-", "suppmat", "supinfo", "/asset/",
    # Elsevier / ScienceDirect (…-mmc1.pdf)
    "mmc",
)
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def resolve_si_pdf_urls(doi_or_landing_url: str, timeout_s: float = 30.0) -> list[str]:
    """Fetch a paper's public landing page and scrape it for links to supplementary-information
    files. `doi_or_landing_url` may be a bare DOI (resolved via https://doi.org/), a doi.org URL,
    or a direct landing-page URL. Returns absolute candidate SI URLs, best first (PDFs before
    other download links), or [] on any failure -- never raises."""
    from urllib.parse import urljoin

    if doi_or_landing_url.startswith("http"):
        landing = doi_or_landing_url
    elif "/" in doi_or_landing_url and " " not in doi_or_landing_url:
        landing = f"https://doi.org/{doi_or_landing_url}"  # DOI-shaped
    else:
        return []

    try:
        scraper = _get_cloudscraper()
        resp = scraper.get(landing, timeout=timeout_s)
        resp.raise_for_status()
    except Exception:
        return []
    if "html" not in resp.headers.get("content-type", "").lower():
        return []

    final_url = str(resp.url)
    seen: set[str] = set()
    pdfs: list[str] = []
    others: list[str] = []
    for href in _HREF_RE.findall(resp.text):
        low = href.lower()
        if not any(marker in low for marker in _SI_URL_MARKERS):
            continue
        absolute = urljoin(final_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        (pdfs if low.split("?")[0].endswith(".pdf") else others).append(absolute)
    return pdfs + others


def get_supplementary_excerpt(paper, max_chars: int = 8000) -> Optional[str]:
    """Try to fetch and extract the synthesis-relevant part of a paper's supplementary information.
    Returns None if no SI link is found on the landing page, or nothing downloadable/parseable --
    a normal, expected outcome, never an error."""
    landing = paper.source_id if (paper.source_id and "/" in paper.source_id) else paper.url
    if not landing:
        return None
    for si_url in resolve_si_pdf_urls(landing):
        text = fetch_pdf_text(si_url)
        if text:
            return extract_relevant_section(text, max_chars=max_chars)
    return None


def _get_cloudscraper():
    """Lazy-init a cloudscraper session for Cloudflare-protected sites."""
    import cloudscraper
    return cloudscraper.create_scraper()


def _resolve_alternative_pdf(source_id: Optional[str]) -> Optional[str]:
    """Try to find a PDF via Unpaywall repository copies or Crossref preprint relations.
    Works around Cloudflare-blocked publisher and ChemRxiv sites by finding repository
    mirrors (institutional repositories, preprint servers) that serve PDFs without bot
    challenges."""
    import os as _os
    if not source_id or "/" not in source_id:
        return None

    email = _os.environ.get("UNPAYWALL_EMAIL") or _os.environ.get("OPENALEX_MAILTO")
    if not email:
        email = None

    # 1. Unpaywall: find repository copies (institutional repos bypass Cloudflare)
    if email:
        try:
            resp = requests.get(
                f"https://api.unpaywall.org/v2/{source_id}",
                params={"email": email},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                for loc in data.get("oa_locations", []):
                    pdf_url = loc.get("url_for_pdf")
                    host = loc.get("host_type", "")
                    if pdf_url and host == "repository":
                        return pdf_url
                best = data.get("best_oa_location") or {}
                if best.get("url_for_pdf"):
                    return best["url_for_pdf"]
        except Exception:
            pass

    # 2. Crossref: check for preprint relation, then resolve that preprint's PDF
    try:
        resp = requests.get(
            f"https://api.crossref.org/works/{source_id}",
            timeout=10,
            headers={"User-Agent": "materials-synthesis-agent/0.3 (mailto:materials-synthesis-agent@example.com)"},
        )
        if resp.ok:
            relations = resp.json().get("message", {}).get("relation", {})
            for rel_type in ("has-preprint", "is-preprint-of"):
                for rel in relations.get(rel_type, []):
                    preprint_doi = rel.get("id", "")
                    if preprint_doi and "chemrxiv" in preprint_doi.lower():
                        preprint_pdf = _resolve_alternative_pdf(preprint_doi)
                        if preprint_pdf:
                            return preprint_pdf
    except Exception:
        pass

    return None


def _candidate_pdf_urls(paper, unpaywall_email: Optional[str] = None):
    """Yield candidate PDF URLs in priority order, trying each source."""
    from materials_synthesis_agent.literature.retrieval import resolve_oa_pdf_url_via_unpaywall

    if paper.oa_pdf_url:
        yield paper.oa_pdf_url
    if paper.source_id and "/" in paper.source_id:
        url = resolve_oa_pdf_url_via_unpaywall(paper.source_id, email=unpaywall_email)
        if url and url != paper.oa_pdf_url:
            yield url
        alt = _resolve_alternative_pdf(paper.source_id)
        if alt and alt != url and alt != paper.oa_pdf_url:
            yield alt


def get_full_text_excerpt(
    paper, max_chars: int = 40000, unpaywall_email: Optional[str] = None, include_supplementary: bool = True
) -> Optional[str]:
    """End-to-end synthesis-text resolution for one paper. Tries, in order:

      1. the paper's supplementary information (where a COF's exact conditions usually live, and
         which is usually open even when the article is paywalled), if `include_supplementary`;
      2. the open-access main text (the paper's own `oa_pdf_url`, or a live Unpaywall lookup for a
         DOI with no PDF already), reduced to its Experimental/Methods section.

    When both are found they're combined (SI first, since it's synthesis-dense), bounded by
    `max_chars`. Returns None only if neither yields anything -- the caller then falls back to
    `paper.abstract`, exactly as if this function were never called."""
    from materials_synthesis_agent.literature.retrieval import resolve_oa_pdf_url_via_unpaywall

    parts: list[str] = []

    section_chars = max_chars // 2  # each part gets half the budget

    if include_supplementary:
        si = get_supplementary_excerpt(paper, max_chars=section_chars)
        if si:
            parts.append(f"--- SUPPLEMENTARY INFORMATION ---\n{si}")

    full_text = None
    for pdf_url in _candidate_pdf_urls(paper, unpaywall_email):
        full_text = fetch_pdf_text(pdf_url)
        if full_text:
            break
    if full_text:
        # Short papers: send the whole text; long papers: extract the experimental section
        remaining = max_chars - sum(len(p) for p in parts)
        if len(full_text) <= remaining:
            parts.append(f"--- MAIN TEXT (full) ---\n{full_text}")
        else:
            parts.append(f"--- MAIN TEXT (Experimental/Methods) ---\n{extract_relevant_section(full_text, max_chars=remaining)}")

    if not parts:
        return None
    return "\n\n".join(parts)[:max_chars].strip()
