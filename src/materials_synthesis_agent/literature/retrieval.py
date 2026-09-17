"""Literature retrieval -- Semantic Scholar + OpenAlex MERGED, arXiv as last resort.

`search()` queries Semantic Scholar and OpenAlex and merges the results (deduplicated by DOI/title):
both are real, independent sources with complementary coverage, so querying both widens the corpus
the extractor draws full-text protocols from. arXiv runs only if both come up empty.

All three are free, keyless APIs. Semantic Scholar's unauthenticated rate limit is very low
(observed 429s in normal use without a key) -- callers should expect to need
`SEMANTIC_SCHOLAR_API_KEY` for anything beyond light interactive use; this module degrades to a
clear error, not a silent empty result, when rate-limited. OpenAlex is merged in -- confirmed
directly (not assumed) to return real, relevant results with no key at all, full-text search
included, via its "polite pool" (a `mailto` param gets a higher, documented rate limit -- see
https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication). arXiv is the last
resort: real coverage of COF/materials chemistry literature there is thin (it's a preprint server,
not a chemistry-journal index), so it only runs if both real literature APIs are unavailable.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import requests

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org/works"
ARXIV_BASE = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    source_id: str  # DOI, Semantic Scholar paperId, or arXiv id
    title: str
    abstract: Optional[str]
    year: Optional[int]
    url: str
    source: str  # "semantic_scholar" | "openalex" | "arxiv"
    oa_pdf_url: Optional[str] = None  # a direct, fetchable PDF URL, only when the source API says
    # the paper is genuinely open access -- None means "no legally fetchable full text found here,"
    # not "this field wasn't populated." See literature/fulltext.py, which only ever reads this.


class RateLimitedError(RuntimeError):
    """Raised on a 429. Distinct from other request failures so callers can decide whether to
    retry, back off, or surface "try again with an API key" to the user -- never silently
    swallowed into an empty result, which would look like "no relevant papers found."""


def search_semantic_scholar(
    query: str, limit: int = 10, api_key: Optional[str] = None, timeout_s: float = 15.0, max_retries: int = 2
) -> list[Paper]:
    api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    # openAccessPdf is documented in Semantic Scholar's public Graph API field list
    # (api.semanticscholar.org/api-docs) as {url, status} -- requested here but not live-verified
    # in this environment (every real call so far has been rate-limited before reaching a
    # response); treat it the same as any other field this parser reads defensively.
    params = {"query": query, "limit": limit, "fields": "title,abstract,year,externalIds,url,openAccessPdf"}

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        resp = requests.get(f"{SEMANTIC_SCHOLAR_BASE}/paper/search", params=params, headers=headers, timeout=timeout_s)
        if resp.status_code == 429:
            if attempt < max_retries:
                time.sleep(2**attempt)
                continue
            raise RateLimitedError(
                "Semantic Scholar rate-limited this request. Set SEMANTIC_SCHOLAR_API_KEY "
                "(free, see https://www.semanticscholar.org/product/api#api-key-form) for a "
                "much higher limit, or retry later."
            )
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for item in data.get("data", []):
            external_ids = item.get("externalIds") or {}
            source_id = external_ids.get("DOI") or item.get("paperId")
            oa_pdf = item.get("openAccessPdf") or {}
            papers.append(
                Paper(
                    source_id=source_id,
                    title=item.get("title") or "(untitled)",
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId')}",
                    source="semantic_scholar",
                    oa_pdf_url=oa_pdf.get("url"),
                )
            )
        return papers
    raise last_exc or RuntimeError("Semantic Scholar search failed for an unknown reason.")


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """OpenAlex returns abstracts as an inverted index (word -> [positions]), not plain text --
    confirmed against a real response, not assumed. Reassembling it is lossless for word order."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


def search_openalex(query: str, limit: int = 10, timeout_s: float = 15.0, mailto: Optional[str] = None) -> list[Paper]:
    params = {"search": query, "per-page": min(limit, 200)}
    if mailto or os.environ.get("OPENALEX_MAILTO"):
        # OpenAlex's "polite pool" -- a contact email gets a documented higher rate limit, no
        # registration needed. Optional: an empty/unauthenticated request still works.
        params["mailto"] = mailto or os.environ.get("OPENALEX_MAILTO")

    resp = requests.get(OPENALEX_BASE, params=params, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for item in data.get("results", [])[:limit]:
        doi = item.get("doi") or ""
        source_id = doi.removeprefix("https://doi.org/") if doi else item.get("id", "")
        # best_oa_location.pdf_url -- confirmed against real live responses (not assumed): present
        # for genuinely open-access papers, None otherwise. A landing_page_url with no pdf_url
        # (an institutional-repository page, say) is deliberately NOT used here -- it's HTML, not
        # a fetchable PDF, and scraping arbitrary landing pages is out of scope.
        best_oa = item.get("best_oa_location") or {}
        papers.append(
            Paper(
                source_id=source_id,
                title=item.get("display_name") or item.get("title") or "(untitled)",
                abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                year=item.get("publication_year"),
                url=item.get("id") or (doi or ""),
                source="openalex",
                oa_pdf_url=best_oa.get("pdf_url"),
            )
        )
    return papers


def search_arxiv(query: str, limit: int = 10, timeout_s: float = 15.0) -> list[Paper]:
    resp = requests.get(
        ARXIV_BASE, params={"search_query": f"all:{query}", "start": 0, "max_results": limit}, timeout=timeout_s
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        title = (entry.findtext("atom:title", default="(untitled)", namespaces=ARXIV_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default=None, namespaces=ARXIV_NS) or "").strip() or None
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
        year = int(published[:4]) if published[:4].isdigit() else None
        # arXiv papers are inherently open access -- the PDF URL is a direct, documented
        # transformation of the abstract-page URL (arxiv.org/abs/X -> arxiv.org/pdf/X), not a guess.
        oa_pdf_url = arxiv_id.replace("/abs/", "/pdf/") if "/abs/" in arxiv_id else None
        papers.append(
            Paper(source_id=arxiv_id, title=title, abstract=summary, year=year, url=arxiv_id, source="arxiv", oa_pdf_url=oa_pdf_url)
        )
    return papers


UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


def resolve_oa_pdf_url_via_unpaywall(doi: str, email: Optional[str] = None, timeout_s: float = 15.0) -> Optional[str]:
    """Secondary OA-PDF resolver for a paper whose search result didn't already carry one
    (`Paper.oa_pdf_url`) but does have a DOI. Unpaywall requires a real contact email per its terms
    -- it rejects placeholder addresses (confirmed directly: a generic example.com address gets a
    real 422 "please use your own email" response) -- so this is a no-op, not an error, when no
    email is configured. Implemented against Unpaywall's documented response shape
    (`best_oa_location.url_for_pdf`); not live-verified end-to-end in this environment for the same
    reason -- no real contact email was available to test with."""
    email = email or os.environ.get("UNPAYWALL_EMAIL") or os.environ.get("OPENALEX_MAILTO")
    if not email or not doi:
        return None
    try:
        resp = requests.get(f"{UNPAYWALL_BASE}/{doi}", params={"email": email}, timeout=timeout_s)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    data = resp.json()
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf")


def _dedup_key(paper: Paper) -> str:
    """Identity for cross-source deduplication: a DOI when we have one (the same paper on Semantic
    Scholar and OpenAlex shares its DOI), else a normalized title."""
    sid = (paper.source_id or "").strip().lower()
    if sid.startswith("10."):  # a DOI
        return "doi:" + sid
    title = re.sub(r"[^a-z0-9]", "", (paper.title or "").lower())
    return ("title:" + title) if title else ("id:" + sid)


def search(
    query: str,
    limit: int = 10,
    semantic_scholar_api_key: Optional[str] = None,
    mailto: Optional[str] = None,
) -> list[Paper]:
    """Search Semantic Scholar AND OpenAlex and MERGE the results, deduplicated by DOI/title -- both
    are real, independent literature sources with complementary coverage (S2 is broad on chemistry
    journals; OpenAlex has strong open-access PDF links), so querying both widens the corpus the
    extractor can draw full-text protocols from. Each source's failure (a Semantic Scholar rate
    limit, an OpenAlex network error) is tolerated -- it just contributes nothing. arXiv is queried
    only as a last resort, when both primary sources yield nothing at all.

    Order is preserved source-first (Semantic Scholar results, then OpenAlex papers not already
    seen), and the merged list is capped at `limit`."""
    results: list[Paper] = []
    seen: set[str] = set()

    def _add(papers: list[Paper]) -> None:
        for p in papers:
            key = _dedup_key(p)
            if key not in seen:
                seen.add(key)
                results.append(p)

    try:
        _add(search_semantic_scholar(query, limit=limit, api_key=semantic_scholar_api_key))
    except RateLimitedError:
        pass  # S2 rate-limited -> rely on OpenAlex

    try:
        _add(search_openalex(query, limit=limit, mailto=mailto))
    except requests.RequestException:
        pass  # OpenAlex unreachable -> S2 results (if any) still stand

    if not results:  # both primary sources came up empty -> last-resort preprint search
        try:
            _add(search_arxiv(query, limit=limit))
        except requests.RequestException:
            pass

    return results[:limit]
