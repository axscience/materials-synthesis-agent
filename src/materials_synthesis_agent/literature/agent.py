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
    return f"{target.linkage_chemistry} {' '.join(target.functional_groups)} synthesis {target.application}"


def estimate_generation_cost(target: Target, papers: list[Paper], model: Optional[str] = None) -> float:
    from materials_synthesis_agent.literature.extraction import estimate_extraction_cost

    return sum(estimate_extraction_cost(target, p, model=model) for p in papers)


def generate_protocols(
    target: Target,
    n: int = 5,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    search_limit: Optional[int] = None,
) -> list[ProtocolCandidate]:
    """Search for papers relevant to `target`, extract a protocol from each, and return up to `n`
    successful extractions. A paper that doesn't describe a usable protocol (extract_protocol
    returns None) is skipped, not counted toward `n` -- callers get real candidates, not padding."""
    papers = search(build_query(target), limit=search_limit or n * 3)

    candidates: list[ProtocolCandidate] = []
    for paper in papers:
        if len(candidates) >= n:
            break
        result = extract_protocol(target, paper, client=client, model=model, dry_run=False)
        if result is not None:
            candidates.append(result)
    return candidates
