"""Hierarchical literature fallback by linkage chemistry.

When there's no literature on the exact target COF, the next-best evidence is a COF made by the
same bond-forming chemistry, then one made by a *related* chemistry. This module encodes that
hierarchy and turns a Target into an ordered list of search tiers, most-specific first:

  tier 0  the exact material (by name), if the target is named
  tier 1  any COF with the SAME linkage chemistry
  tier 2+ COFs with progressively-less-similar linkage chemistries

Tiers 0 and 1 stay within the target's own linkage. Tiers 2+ cross into a *different* chemistry --
a genuinely weaker prior for the target -- so the orchestrator asks the user before running them
and tags anything they produce with a provenance note.

The similarity ordering below is a chemistry-informed heuristic, not a rigorous metric, and it is
deliberately easy to edit -- reasonable chemists will rank "somewhat similar" differently. The
imine ordering follows the reversible-imine (Schiff-base) family first, since those share the same
C=N dynamic-covalent bond and reaction conditions most closely, then broader relatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from materials_synthesis_agent.schema import Target

# base linkage -> ordered list of related linkages, most similar first. Keys are matched as
# substrings of a target's (free-text) linkage_chemistry, so "imine condensation" -> "imine".
LINKAGE_SIMILARITY: dict[str, list[str]] = {
    "imine": ["hydrazone", "azine", "beta-ketoenamine", "imide"],
    "hydrazone": ["imine", "azine", "beta-ketoenamine"],
    "azine": ["hydrazone", "imine"],
    "beta-ketoenamine": ["imine", "hydrazone"],
    "ketoenamine": ["imine", "hydrazone"],
    "imide": ["imine", "amide"],
    "boronate ester": ["boroxine", "borazine"],
    "boroxine": ["boronate ester", "borazine"],
    "borazine": ["boronate ester", "boroxine"],
    "triazine": ["imine"],  # CTFs are fairly distinct; imine is the closest common alternative
}

# Ordered so multi-word keys are tested before their substrings (so "boronate ester" wins over a
# hypothetical "boron" key, and "beta-ketoenamine" over "ketoenamine").
_BASE_LINKAGES_BY_SPECIFICITY = sorted(LINKAGE_SIMILARITY, key=len, reverse=True)


def base_linkage(linkage_chemistry: str) -> Optional[str]:
    """Which known base linkage the target's (possibly verbose) linkage_chemistry names, or None if
    it isn't one we have a similarity ordering for. Substring match, case-insensitive."""
    text = (linkage_chemistry or "").lower()
    for base in _BASE_LINKAGES_BY_SPECIFICITY:
        if base in text:
            return base
    return None


@dataclass(frozen=True)
class SearchTier:
    label: str  # human-readable, shown to the user
    query: str
    linkage: Optional[str]  # the base linkage this tier searches, or None (name-only tier)
    beyond_target_linkage: bool  # True once we've left the target's own linkage chemistry


def _linkage_query(linkage: str, target: Target) -> str:
    parts = [linkage, "covalent organic framework synthesis"]
    if target.functional_groups:
        parts.extend(target.functional_groups)
    if target.application and target.application.lower() not in ("general", ""):
        parts.append(target.application)
    # Dedupe case-insensitively, preserving order -- functional groups often repeat the linkage
    # word (e.g. a "imine"-linkage target with "imine" also listed as a functional group).
    seen: set[str] = set()
    deduped = [p for p in parts if not (p.lower() in seen or seen.add(p.lower()))]
    return " ".join(deduped)


def build_search_tiers(target: Target) -> list[SearchTier]:
    """Ordered search tiers for a target, most-specific first. See module docstring."""
    tiers: list[SearchTier] = []
    base = base_linkage(target.linkage_chemistry)

    # tier 0: the exact named material.
    if target.name:
        tiers.append(SearchTier(
            label=f"the exact COF ({target.name})",
            query=f"{target.name} covalent organic framework synthesis",
            linkage=base,
            beyond_target_linkage=False,
        ))

    # tier 1: same linkage chemistry.
    if base:
        tiers.append(SearchTier(
            label=f"COFs with the same linkage chemistry ({base})",
            query=_linkage_query(base, target),
            linkage=base,
            beyond_target_linkage=False,
        ))
    else:
        # No recognized base linkage -- fall back to the free-text linkage as a single same-linkage
        # tier, and there are no "related linkage" tiers to offer (we don't know the neighbors).
        tiers.append(SearchTier(
            label=f"COFs with the stated linkage chemistry ({target.linkage_chemistry})",
            query=_linkage_query(target.linkage_chemistry, target),
            linkage=None,
            beyond_target_linkage=False,
        ))

    # tiers 2+: related-but-different linkages, most similar first.
    for related in LINKAGE_SIMILARITY.get(base or "", []):
        tiers.append(SearchTier(
            label=f"COFs with a related linkage chemistry ({related})",
            query=_linkage_query(related, target),
            linkage=related,
            beyond_target_linkage=True,
        ))

    return tiers


def related_linkages(target: Target) -> list[str]:
    """The ordered list of related-but-different linkages that would be searched beyond the target's
    own -- what the orchestrator shows the user when asking whether to expand."""
    return [t.linkage for t in build_search_tiers(target) if t.beyond_target_linkage and t.linkage]
