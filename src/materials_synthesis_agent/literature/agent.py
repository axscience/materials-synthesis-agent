"""Orchestrates retrieval + extraction: given a Target, search relevant papers and extract N
citation-grounded candidate protocols. This is what the CLI's `suggest-protocols` command and
materials-copilot's literature job both call.
"""

from __future__ import annotations

from typing import Optional

from materials_synthesis_agent.literature.extraction import extract_protocol
from materials_synthesis_agent.literature.retrieval import Paper, search
from materials_synthesis_agent.llm import LLMClient
from materials_synthesis_agent.schema import ProtocolCandidate, Target


def build_query(target: Target) -> str:
    """A name-targeted search ("COF-5 covalent organic framework synthesis") finds the specific
    paper(s) that made that exact material far more reliably than a functional-group/linkage query
    would, so `target.name` wins whenever it's set. Adding "covalent organic framework" to named
    queries disambiguates common abbreviations (e.g. "COF-1" alone matches unrelated papers).

    The fallback query for unnamed/hypothesized COFs combines the linkage chemistry with "covalent
    organic framework synthesis" and any functional groups, producing queries like
    "imine covalent organic framework synthesis TAPB PDA" that hit the real literature."""
    if target.name:
        return f"{target.name} covalent organic framework synthesis"
    parts = [target.linkage_chemistry, "covalent organic framework synthesis"]
    if target.functional_groups:
        parts.extend(target.functional_groups)
    if target.application and target.application.lower() not in ("general", ""):
        parts.append(target.application)
    return " ".join(parts)


def _resolve_excerpt(paper: Paper, use_full_text: bool, unpaywall_email: Optional[str]) -> Optional[str]:
    if not use_full_text:
        return None
    from materials_synthesis_agent.literature.fulltext import get_full_text_excerpt

    return get_full_text_excerpt(paper, unpaywall_email=unpaywall_email)


def estimate_generation_cost(
    target: Target,
    papers: list[Paper],
    model: Optional[str] = None,
    use_full_text: bool = False,
    unpaywall_email: Optional[str] = None,
) -> float:
    """`use_full_text=True` actually resolves and fetches each paper's OA PDF here (no $ cost, but
    real network/time cost) so the estimate reflects the real, larger full-text prompt size rather
    than under-quoting against an abstract-only estimate the real call won't use."""
    from materials_synthesis_agent.literature.extraction import estimate_extraction_cost

    total = 0.0
    for p in papers:
        excerpt = _resolve_excerpt(p, use_full_text, unpaywall_email)
        total += estimate_extraction_cost(target, p, model=model, full_text_excerpt=excerpt)
    return total


def generate_protocols(
    target: Target,
    n: int = 5,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    search_limit: Optional[int] = None,
    use_full_text: bool = False,
    unpaywall_email: Optional[str] = None,
) -> list[ProtocolCandidate]:
    """Search for papers relevant to `target`, extract a protocol from each, and return up to `n`
    successful extractions. A paper that doesn't describe a usable protocol (extract_protocol
    returns None) is skipped, not counted toward `n` -- callers get real candidates, not padding.

    `use_full_text=True` tries to fetch each paper's real, open-access full text (see
    literature/fulltext.py) and extract from its Experimental/Methods section instead of just the
    abstract -- falls back to abstract-only per paper when no OA PDF is found, a download fails, or
    the PDF doesn't parse, never raises for that reason."""
    papers = search(build_query(target), limit=search_limit or n * 3)

    candidates: list[ProtocolCandidate] = []
    for paper in papers:
        if len(candidates) >= n:
            break
        excerpt = _resolve_excerpt(paper, use_full_text, unpaywall_email)
        result = extract_protocol(target, paper, client=client, model=model, dry_run=False, full_text_excerpt=excerpt)
        if result is not None:
            candidates.append(result)
    return candidates
