"""Literature retrieval -- Semantic Scholar (primary), OpenAlex (fallback), arXiv (last resort).

All three are free, keyless APIs. Semantic Scholar's unauthenticated rate limit is very low
(observed 429s in normal use without a key) -- callers should expect to need
`SEMANTIC_SCHOLAR_API_KEY` for anything beyond light interactive use; this module degrades to a
clear error, not a silent empty result, when rate-limited. OpenAlex is tried next -- confirmed
directly (not assumed) to return real, relevant results with no key at all, full-text search
included, via its "polite pool" (a `mailto` param gets a higher, documented rate limit -- see
https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication). arXiv is the last
resort: real coverage of COF/materials chemistry literature there is thin (it's a preprint server,
not a chemistry-journal index), so it only runs if both real literature APIs are unavailable.
"""

from __future__ import annotations

import os
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


class RateLimitedError(RuntimeError):
    """Raised on a 429. Distinct from other request failures so callers can decide whether to
    retry, back off, or surface "try again with an API key" to the user -- never silently
    swallowed into an empty result, which would look like "no relevant papers found."""


def search_semantic_scholar(
    query: str, limit: int = 10, api_key: Optional[str] = None, timeout_s: float = 15.0, max_retries: int = 2
) -> list[Paper]:
    api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "limit": limit, "fields": "title,abstract,year,externalIds,url"}

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
            papers.append(
                Paper(
                    source_id=source_id,
                    title=item.get("title") or "(untitled)",
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId')}",
                    source="semantic_scholar",
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
        papers.append(
            Paper(
                source_id=source_id,
                title=item.get("display_name") or item.get("title") or "(untitled)",
                abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                year=item.get("publication_year"),
                url=item.get("id") or (doi or ""),
                source="openalex",
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
        papers.append(
            Paper(source_id=arxiv_id, title=title, abstract=summary, year=year, url=arxiv_id, source="arxiv")
        )
    return papers


def search(query: str, limit: int = 10, semantic_scholar_api_key: Optional[str] = None) -> list[Paper]:
    """Search Semantic Scholar first (broader coverage of chemistry journals), fall back to
    OpenAlex if Semantic Scholar is rate-limited, then arXiv as a last resort if OpenAlex's request
    itself fails outright (a real network/HTTP error, not just an empty result -- an empty result
    is a real "no papers found," not something to escalate past). Each fallback is a real,
    independent literature source, not a silent empty result standing in for one."""
    try:
        return search_semantic_scholar(query, limit=limit, api_key=semantic_scholar_api_key)
    except RateLimitedError:
        pass

    try:
        return search_openalex(query, limit=limit)
    except requests.RequestException:
        return search_arxiv(query, limit=limit)
