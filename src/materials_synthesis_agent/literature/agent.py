"""Orchestrates retrieval + extraction: given a Target, search relevant papers and extract N
citation-grounded candidate protocols. This is what the CLI's `suggest-protocols` command calls.

Search proceeds hierarchically by linkage chemistry (see literature/linkage_fallback.py): the exact
COF first, then any COF with the same linkage, then -- only with the user's go-ahead -- COFs with a
related-but-different linkage. Candidates found beyond the target's own linkage are tagged with a
provenance note so a weaker-prior protocol is never silently treated as an exact match.
"""

from __future__ import annotations

from typing import Callable, Optional

from materials_synthesis_agent.literature.extraction import extract_protocol
from materials_synthesis_agent.literature.linkage_fallback import (
    SearchTier,
    build_search_tiers,
    related_linkages,
)
from materials_synthesis_agent.literature.retrieval import Paper, search
from materials_synthesis_agent.llm import LLMClient
from materials_synthesis_agent.schema import ProtocolCandidate, Target


def build_query(target: Target) -> str:
    """The single most-specific query for a target -- its first search tier. Kept for callers (and
    tests) that want one query; the full hierarchical plan is `build_search_tiers`."""
    return build_search_tiers(target)[0].query


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
    """`use_full_text=True` actually resolves and fetches each paper's OA full text / SI here (no $
    cost, but real network/time cost) so the estimate reflects the real, larger prompt size rather
    than under-quoting against an abstract-only estimate the real call won't use."""
    from materials_synthesis_agent.literature.extraction import estimate_extraction_cost

    total = 0.0
    for p in papers:
        excerpt = _resolve_excerpt(p, use_full_text, unpaywall_email)
        total += estimate_extraction_cost(target, p, model=model, full_text_excerpt=excerpt)
    return total


def _extract_tier(
    target: Target,
    tier: SearchTier,
    need: int,
    client: Optional[LLMClient],
    model: Optional[str],
    search_limit: int,
    use_full_text: bool,
    unpaywall_email: Optional[str],
) -> list[ProtocolCandidate]:
    """Search one tier and extract up to `need` protocols from it, tagging each with the tier's
    provenance (so a related-linkage candidate is clearly marked a weaker prior)."""
    papers = search(tier.query, limit=search_limit)
    out: list[ProtocolCandidate] = []
    for paper in papers:
        if len(out) >= need:
            break
        excerpt = _resolve_excerpt(paper, use_full_text, unpaywall_email)
        result = extract_protocol(target, paper, client=client, model=model, dry_run=False, full_text_excerpt=excerpt)
        if result is not None:
            note = f"Extracted from literature on {tier.label}."
            if tier.beyond_target_linkage:
                note += (
                    f" NOTE: this is a related but DIFFERENT linkage chemistry ({tier.linkage}) than the "
                    f"target ({target.linkage_chemistry}) -- treat as a weaker prior, not an exact precedent."
                )
            result.provenance_note = note
            out.append(result)
    return out


def generate_protocols(
    target: Target,
    n: int = 5,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    search_limit: Optional[int] = None,
    use_full_text: bool = False,
    unpaywall_email: Optional[str] = None,
    confirm_expand: Optional[Callable[[list[str]], bool]] = None,
) -> list[ProtocolCandidate]:
    """Search for papers relevant to `target` and return up to `n` extracted candidate protocols,
    proceeding hierarchically by linkage chemistry (see module docstring).

    Tiers within the target's own linkage (the exact COF, then same-linkage COFs) always run. Before
    searching a *different* (related) linkage, `confirm_expand` is called once with the ordered list
    of related linkages that would be tried; if it returns False (or is None), the search stops at
    the target's own linkage rather than silently broadening. Anything found beyond the target's
    linkage carries a provenance note marking it a weaker prior.

    `use_full_text=True` tries each paper's real open-access full text and supplementary information
    (see literature/fulltext.py), falling back to its abstract per paper when nothing is available.

    A paper that doesn't describe a usable protocol is skipped, not counted toward `n` -- callers
    get real candidates, not padding. Returns fewer than `n` (possibly zero) when the literature,
    within whatever linkage scope the user allowed, simply doesn't have more."""
    limit = search_limit or n * 3
    tiers = build_search_tiers(target)
    candidates: list[ProtocolCandidate] = []
    asked_to_expand = False

    for tier in tiers:
        if len(candidates) >= n:
            break
        if tier.beyond_target_linkage and not asked_to_expand:
            asked_to_expand = True
            alternatives = related_linkages(target)
            if confirm_expand is None or not confirm_expand(alternatives):
                break  # user declined (or no confirmer) -- stay within the target's own linkage
        candidates.extend(
            _extract_tier(target, tier, n - len(candidates), client, model, limit, use_full_text, unpaywall_email)
        )
    return candidates[:n]
