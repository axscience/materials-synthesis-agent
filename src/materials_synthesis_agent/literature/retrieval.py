"""Literature retrieval -- Semantic Scholar (primary) and arXiv (fallback/supplement).

Both are free, keyless APIs. Semantic Scholar's unauthenticated rate limit is very low (observed
429s in normal use without a key) -- callers should expect to need `SEMANTIC_SCHOLAR_API_KEY` for
anything beyond light interactive use; this module degrades to a clear error, not a silent empty
result, when rate-limited.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import requests

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    source_id: str  # DOI, Semantic Scholar paperId, or arXiv id
    title: str
    abstract: Optional[str]
    year: Optional[int]
    url: str
    source: str  # "semantic_scholar" | "arxiv"


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
    """Search Semantic Scholar first (broader coverage of chemistry journals), fall back to arXiv
    if Semantic Scholar is rate-limited -- surfaced as a real fallback, not silently swallowed."""
    try:
        return search_semantic_scholar(query, limit=limit, api_key=semantic_scholar_api_key)
    except RateLimitedError:
        return search_arxiv(query, limit=limit)
