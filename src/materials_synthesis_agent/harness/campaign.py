"""Campaign, Session, and characterization state models for the Discovery Harness.

A Campaign is the central, resumable state object for one user project. It deliberately holds
*metadata and pointers*, not copies of the loop data: the Target, its ProtocolCandidates,
Experiments, BOSuggestions, and Decisions all live in the existing `storage.Store` keyed by
`target_id`, which stays the single source of truth. Campaign adds only what the Store has no notion
of -- the project's identity, the optimizer's parameter space, the calibration correction in force,
the design-side results, characterizations, and the session list.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CampaignStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class SynthesisOutcome(str, Enum):
    SUCCESS = "success"        # target properties met
    PARTIAL = "partial"        # material formed, properties below target
    AMORPHOUS = "amorphous"    # no crystallinity
    NO_PRODUCT = "no_product"  # reaction didn't yield solid
    UNKNOWN = "unknown"        # characterization inconclusive


class PropertyComparison(BaseModel):
    """One predicted-vs-measured comparison. `within_prediction` and `deviation_sigma` are computed,
    never guessed -- they feed the honest-uncertainty story, not decorate it."""

    property_name: str
    predicted: Optional[float] = None
    predicted_uncertainty: Optional[float] = None
    measured: float
    measured_uncertainty: Optional[float] = None
    within_prediction: bool = False
    deviation_sigma: Optional[float] = None

    @classmethod
    def build(
        cls,
        property_name: str,
        measured: float,
        predicted: Optional[float] = None,
        predicted_uncertainty: Optional[float] = None,
        measured_uncertainty: Optional[float] = None,
    ) -> "PropertyComparison":
        within = False
        sigma: Optional[float] = None
        if predicted is not None:
            # Combine predicted and measured uncertainty in quadrature; guard the zero-sigma case.
            pu = predicted_uncertainty or 0.0
            mu = measured_uncertainty or 0.0
            spread = (pu**2 + mu**2) ** 0.5
            if spread > 0:
                sigma = abs(measured - predicted) / spread
                within = sigma <= 1.0
            else:
                within = measured == predicted
        return cls(
            property_name=property_name,
            predicted=predicted,
            predicted_uncertainty=predicted_uncertainty,
            measured=measured,
            measured_uncertainty=measured_uncertainty,
            within_prediction=within,
            deviation_sigma=sigma,
        )


class CharacterizationResult(BaseModel):
    """What Loop 3 produces: the comparison between what we predicted and what the bench measured.
    In v1 the numbers are entered by a human via `log-result`; the automated `char.score_pxrd` model
    (spec M5) will later fill `crystallinity_assessment`/`structure_confirmed` directly."""

    id: str = Field(default_factory=_new_id)
    experiment_id: str
    candidate_id: Optional[str] = None
    property_comparisons: list[PropertyComparison] = Field(default_factory=list)
    structure_confirmed: Optional[bool] = None
    crystallinity_assessment: Optional[str] = None
    outcome: SynthesisOutcome = SynthesisOutcome.UNKNOWN
    failure_hypothesis: Optional[str] = None
    next_action_suggestion: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class Turn(BaseModel):
    """One message in a session. `tool_calls` holds compact summaries of what ran behind the scenes
    (name + cost + status), not the raw tool payloads -- those live in the Store."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=_now)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    cost: float = 0.0


class Session(BaseModel):
    """One conversation with the harness. Persisted for resume, audit, and Markdown export."""

    id: str = Field(default_factory=_new_id)
    campaign_id: str
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None
    turns: list[Turn] = Field(default_factory=list)
    # LLM-generated summary of older turns, so a long session's context stays bounded.
    compressed_history: Optional[str] = None
    context_window_turns: int = 20

    @property
    def total_cost(self) -> float:
        return sum(t.cost for t in self.turns)


class Campaign(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    status: CampaignStatus = CampaignStatus.ACTIVE
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # Pointer into the Store -- the Target and everything hung off it (protocols, experiments,
    # suggestions, decisions) are read from the Store by this id, never copied here.
    target_id: Optional[str] = None

    # Optimizer configuration the Store has no notion of.
    space: Optional[dict[str, Any]] = None      # ParameterSpace definition (list of specs)
    variance_scale: float = 1.0                  # calibration correction in force, if any

    # Denormalized for the cross-campaign warm prior's index (also on the Target, copied here so the
    # prior query never has to load every Target).
    linkage_chemistry: Optional[str] = None

    # Loop 1 (design) -- InverseDesignResult dicts, kept as dicts so the harness has no hard
    # dependency on inverse_design_agent being installed.
    design_results: list[dict[str, Any]] = Field(default_factory=list)
    selected_candidates: list[str] = Field(default_factory=list)

    # Loop 3 (characterization) -- ids into the characterizations table.
    characterization_ids: list[str] = Field(default_factory=list)

    # Sessions belonging to this campaign, newest last.
    session_ids: list[str] = Field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now()
