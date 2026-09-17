"""Orchestrates retrieval + extraction: given a Target, search relevant papers and extract N
citation-grounded candidate protocols. This is what the CLI's `suggest-protocols` command calls.

Search proceeds hierarchically by linkage chemistry (see literature/linkage_fallback.py): the exact
COF first, then any COF with the same linkage, then -- only with the user's go-ahead -- COFs with a
related-but-different linkage. Candidates found beyond the target's own linkage are tagged with a
provenance note so a weaker-prior protocol is never silently treated as an exact match.

Within-linkage tiers (Tier 0 / 0.5 / 1) always run -- even when an earlier tier already found enough
protocols, because different papers contribute different experiments and the GP benefits from more
distinct data points. After all within-linkage tiers are searched, protocols are deduplicated by
building-block fingerprint while preserving every experiment from every merged source, so the
optimizer sees the full literature data landscape without redundant protocol entries.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from materials_synthesis_agent.literature.extraction import extract_protocol
from materials_synthesis_agent.literature.linkage_fallback import (
    SearchTier,
    build_search_tiers,
    related_linkages,
)
from materials_synthesis_agent.literature.retrieval import Paper, search
from materials_synthesis_agent.llm import LLMClient
from materials_synthesis_agent.schema import LiteratureExperiment, MeasuredOutcome, ProtocolCandidate, Target

logger = logging.getLogger(__name__)


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
    use_full_text: bool = True,
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
    require_full_text: bool = False,
    refine: bool = False,
) -> list[ProtocolCandidate]:
    """Search one tier and extract up to `need` protocols from it, tagging each with the tier's
    provenance (so a related-linkage candidate is clearly marked a weaker prior).

    With `require_full_text=True`, papers whose complete open-access PDF (main text / SI) can't be
    fetched are skipped entirely rather than extracted from their abstract -- abstracts give building
    blocks but rarely the numeric conditions/outcomes the GP needs, so this trades quantity for the
    data quality that actually seeds the optimizer."""
    papers = search(tier.query, limit=search_limit)
    out: list[ProtocolCandidate] = []
    for paper in papers:
        if len(out) >= need:
            break
        excerpt = _resolve_excerpt(paper, use_full_text, unpaywall_email)
        if require_full_text and not excerpt:
            continue  # no complete PDF -> skip; don't extract from an abstract
        result = extract_protocol(target, paper, client=client, model=model, dry_run=False, full_text_excerpt=excerpt)
        if result is not None:
            if refine and excerpt:
                from materials_synthesis_agent.literature.extraction import refine_protocol
                result = refine_protocol(result, target, paper, client=client, model=model,
                                         full_text_excerpt=excerpt)
            note = f"Extracted from literature on {tier.label}."
            if tier.beyond_target_linkage:
                note += (
                    f" NOTE: this is a related but DIFFERENT linkage chemistry ({tier.linkage}) than the "
                    f"target ({target.linkage_chemistry}) -- treat as a weaker prior, not an exact precedent."
                )
            result.provenance_note = note
            out.append(result)
    return out


def _count_on_metric(candidates: list[ProtocolCandidate], metric_name: str) -> int:
    """Count the data points that would actually seed the GP: literature experiments and single
    measured outcomes whose metric matches the objective. This is what 'N experiments extracted'
    means for the optimizer -- protocols that report some other property don't count."""
    from materials_synthesis_agent.literature.outcomes import _metric_matches

    total = 0
    for c in candidates:
        total += sum(1 for e in c.literature_experiments if _metric_matches(e.outcome.metric_name, metric_name))
        total += sum(1 for o in c.measured_outcomes if _metric_matches(o.metric_name, metric_name))
    return total


def _protocol_fingerprint(candidate: ProtocolCandidate) -> str:
    """Group key for deduplication: same building blocks = same material family.
    Within a family, coarse-bucket the temperature to separate genuinely different conditions.
    Protocols from different linkage chemistries (cross-linkage search results) are never merged,
    even if they happen to share conditions — they represent distinct material systems."""
    bb_key = tuple(sorted(candidate.building_blocks.keys()))
    solvent = (candidate.solvent.value if candidate.solvent else "").lower().strip()[:30]
    temp_bucket = ""
    if candidate.temperature_c:
        try:
            t = float(candidate.temperature_c.value.split()[0].rstrip("°CcMm"))
            temp_bucket = str(round(t / 20) * 20)
        except (ValueError, IndexError):
            temp_bucket = candidate.temperature_c.value[:10]
    prov = candidate.provenance_note or ""
    return f"{bb_key}|{solvent}|{temp_bucket}|{prov}"


def deduplicate_protocols(
    candidates: list[ProtocolCandidate],
) -> list[ProtocolCandidate]:
    """Collapse protocols with the same building blocks and similar conditions into one
    representative per group. The protocol with the highest citation coverage (and most
    experiments) survives; experiments and measured outcomes from all members of the group
    are merged onto the survivor, so no data point is lost for the GP.

    This is the core of the "search broadly, dedup narrowly" strategy: the search casts a wide
    net across tiers, and this function narrows to unique condition families while preserving
    every (conditions -> outcome) data point the literature provides."""
    if not candidates:
        return []

    groups: dict[str, list[ProtocolCandidate]] = {}
    for c in candidates:
        key = _protocol_fingerprint(c)
        groups.setdefault(key, []).append(c)

    deduplicated: list[ProtocolCandidate] = []
    for key, group in groups.items():
        group.sort(key=lambda c: (len(c.literature_experiments), c.citation_coverage()), reverse=True)
        winner = group[0]

        if len(group) > 1:
            seen_labels = {exp.label for exp in winner.literature_experiments}
            seen_outcomes = {(o.metric_name, o.value) for o in winner.measured_outcomes}

            for donor in group[1:]:
                for exp in donor.literature_experiments:
                    if exp.label not in seen_labels:
                        winner.literature_experiments.append(exp)
                        seen_labels.add(exp.label)
                for outcome in donor.measured_outcomes:
                    outcome_key = (outcome.metric_name, outcome.value)
                    if outcome_key not in seen_outcomes:
                        winner.measured_outcomes.append(outcome)
                        seen_outcomes.add(outcome_key)

            winner.provenance_note = (
                (winner.provenance_note or "") +
                f" Merged from {len(group)} papers with similar conditions; "
                f"all {len(winner.literature_experiments)} experiments preserved."
            ).strip()
            logger.info(
                "Dedup group %s: merged %d protocols into 1 (%d experiments, %d outcomes)",
                key[:40], len(group), len(winner.literature_experiments), len(winner.measured_outcomes),
            )

        deduplicated.append(winner)
    return deduplicated


def generate_protocols(
    target: Target,
    n: int = 8,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    search_limit: Optional[int] = None,
    use_full_text: bool = True,
    unpaywall_email: Optional[str] = None,
    confirm_expand: Optional[Callable[[list[str]], bool]] = None,
    require_full_text: bool = False,
    target_experiments: int = 0,
    refine: bool = False,
) -> list[ProtocolCandidate]:
    """Search for papers relevant to `target` and return up to `n` deduplicated candidate protocols,
    proceeding hierarchically by linkage chemistry (see module docstring).

    **Within-linkage tiers always run in full** -- Tier 0 (exact name), Tier 0.5 (monomer names),
    and Tier 1 (same linkage) each get their own search quota, regardless of how many protocols
    earlier tiers found. This ensures the GP benefits from the widest possible set of experiments
    even when the exact-name search already turns up several protocols.

    Before searching a *different* (related) linkage, `confirm_expand` is called once with the
    ordered list of related linkages that would be tried; if it returns False (or is None), the
    search stops at the target's own linkage rather than silently broadening.

    After extraction, protocols are deduplicated by building-block fingerprint (same material +
    similar conditions = one protocol family), with all experiments from merged protocols preserved
    on the surviving representative. This gives the optimizer maximum data points with minimum
    redundancy in the protocol list presented to the user.

    `use_full_text=True` (the default) tries each paper's real open-access full text and
    supplementary information, falling back to its abstract per paper when nothing is available.

    Returns fewer than `n` (possibly zero) when the literature doesn't have enough."""
    per_tier_limit = search_limit or max(n, 8)
    per_tier_extract = max(n, 6)
    if target_experiments:
        # Data-point-driven mode: extract as many protocols per tier as we might need, and keep
        # searching (auto-expanding into related linkages) until we have >= target_experiments
        # on-metric data points -- the number the GP actually needs to be confident.
        per_tier_extract = max(per_tier_extract, target_experiments)
    tiers = build_search_tiers(target)
    candidates: list[ProtocolCandidate] = []
    asked_to_expand = False

    for tier in tiers:
        if tier.beyond_target_linkage:
            if target_experiments:
                # Auto-expand into related linkages only while still short of the data-point target.
                if _count_on_metric(candidates, target.metric_name) >= target_experiments:
                    break
            elif not asked_to_expand:
                asked_to_expand = True
                if len(candidates) >= n:
                    break
                alternatives = related_linkages(target)
                if confirm_expand is None or not confirm_expand(alternatives):
                    break
        tier_results = _extract_tier(
            target, tier, per_tier_extract, client, model, per_tier_limit, use_full_text,
            unpaywall_email, require_full_text=require_full_text, refine=refine,
        )
        candidates.extend(tier_results)
        got = _count_on_metric(candidates, target.metric_name)
        logger.info("Tier '%s': extracted %d protocols (%d total, %d on-metric data points)",
                    tier.label, len(tier_results), len(candidates), got)
        if target_experiments and got >= target_experiments:
            break

    pre_dedup = len(candidates)
    candidates = deduplicate_protocols(candidates)
    logger.info("Deduplication: %d protocols -> %d unique families", pre_dedup, len(candidates))

    total = _count_on_metric(candidates, target.metric_name)
    logger.info("On-metric data points across all protocols: %d (target: %s)",
                total, target_experiments or "15-30")
    if target_experiments and total < target_experiments:
        logger.info("Literature exhausted at %d on-metric points (< target %d) -- returning what "
                    "was found.", total, target_experiments)

    # In data-point mode keep every candidate (each carries data points); otherwise cap at n.
    return candidates if target_experiments else candidates[:n]
