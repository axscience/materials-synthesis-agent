"""Discovery Harness — the thin layer that unifies the synthesis and inverse-design loops behind
one planner, one campaign state, and one set of blocking gates.

Design stance (see the harness spec): three LLM roles (interpret / extract / plan), a registry of
*deterministic* tools that wrap the existing modules, gates that block instead of a critic that
opines, and a SQL warm-start prior. Campaign state here is metadata + pointers; the loop data
(targets, protocols, experiments, suggestions, decisions) stays in the existing `storage.Store` as
the single source of truth, never duplicated.
"""

from materials_synthesis_agent.harness.campaign import (
    Campaign,
    CampaignStatus,
    CharacterizationResult,
    PropertyComparison,
    Session,
    SynthesisOutcome,
    Turn,
)
from materials_synthesis_agent.harness.store import CampaignStore

__all__ = [
    "Campaign",
    "CampaignStatus",
    "CampaignStore",
    "CharacterizationResult",
    "PropertyComparison",
    "Session",
    "SynthesisOutcome",
    "Turn",
]
