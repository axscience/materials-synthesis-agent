"""Versioned, portable data models -- the contract between this package and any consumer
(the local CLI, the local web UI, or materials-copilot's Postgres layer).

Guardrails this module exists to enforce (see CLAUDE.md):
  - No Metric reaches the optimizer without a stated uncertainty, or it is flagged low_confidence.
  - Every literature-derived field on a ProtocolCandidate traces to a Citation, or is marked
    inferred=True. Never silently presented as sourced fact.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class ProtocolSource(str, Enum):
    LITERATURE = "literature"
    BO_SUGGESTED = "bo_suggested"
    MANUAL = "manual"


class MetricConfidence(str, Enum):
    QUANTIFIED = "quantified"
    LOW_CONFIDENCE = "low_confidence"


class ObjectiveDirection(str, Enum):
    """Whether the metric should be pushed up or down. Many materials metrics are minimize goals
    (particle size, defect density, reaction time, cost), so this can't be assumed to be maximize."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Citation(BaseModel):
    """Provenance for a single extracted field."""

    source_id: str  # e.g. a DOI, arXiv id, or Semantic Scholar paper id
    title: str
    excerpt: Optional[str] = Field(
        default=None, description="The specific text the field was extracted from, if available."
    )


class FieldValue(BaseModel):
    """A single protocol field with its provenance.

    `inferred=True` means the extraction model reasoned to this value rather than reading it
    directly from a citation -- e.g. inferring a typical modulator equivalence when a paper didn't
    state one explicitly. Never render an inferred field the same way as a cited one.
    """

    value: str
    citation: Optional[Citation] = None
    inferred: bool = False

    @model_validator(mode="after")
    def _grounded_or_flagged(self) -> "FieldValue":
        if self.citation is None and not self.inferred:
            raise ValueError(
                "A FieldValue must either carry a citation or be explicitly flagged inferred=True. "
                "Presenting an extracted value as sourced fact with no traceable source is exactly "
                "what CLAUDE.md guardrail #2 forbids."
            )
        return self


class Target(BaseModel):
    id: str = Field(default_factory=_new_id)
    functional_groups: list[str]
    linkage_chemistry: str
    application: str
    metric_name: str
    metric_measurement_method: str
    objective_direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    created_at: datetime = Field(default_factory=_now)

    @property
    def maximize(self) -> bool:
        return self.objective_direction == ObjectiveDirection.MAXIMIZE


class ProtocolCandidate(BaseModel):
    """A versioned, append-only synthesis protocol suggestion.

    A revision is a new ProtocolCandidate with `revises` set to the prior version's id -- this
    object is never mutated in place once created.
    """

    id: str = Field(default_factory=_new_id)
    target_id: str
    version: int = 1
    revises: Optional[str] = None
    source: ProtocolSource

    building_blocks: dict[str, FieldValue] = Field(
        default_factory=dict, description="name -> SMILES, with provenance"
    )
    stoichiometry: dict[str, FieldValue] = Field(default_factory=dict)
    solvent: Optional[FieldValue] = None
    modulator: Optional[FieldValue] = None
    temperature_c: Optional[FieldValue] = None
    time_hours: Optional[FieldValue] = None
    concentration_molar: Optional[FieldValue] = None

    feasibility_flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)

    def all_fields(self) -> list[FieldValue]:
        """Every FieldValue on this candidate, for grounding audits and UI rendering."""
        out: list[FieldValue] = list(self.building_blocks.values()) + list(
            self.stoichiometry.values()
        )
        for f in (self.solvent, self.modulator, self.temperature_c, self.time_hours,
                  self.concentration_molar):
            if f is not None:
                out.append(f)
        return out

    def citation_coverage(self) -> float:
        """Fraction of fields that carry a real citation (not just inferred). Useful as a quick
        trust signal to surface in a UI."""
        fields = self.all_fields()
        if not fields:
            return 0.0
        grounded = sum(1 for f in fields if f.citation is not None)
        return grounded / len(fields)


class Metric(BaseModel):
    """A measured result. `uncertainty=None` is allowed (a user can log a bare number), but
    `confidence` is always derived, never left to a caller or the UI to infer."""

    name: str
    value: float
    uncertainty: Optional[float] = None
    measurement_method: str

    @property
    def confidence(self) -> MetricConfidence:
        return (
            MetricConfidence.QUANTIFIED
            if self.uncertainty is not None
            else MetricConfidence.LOW_CONFIDENCE
        )


class Experiment(BaseModel):
    id: str = Field(default_factory=_new_id)
    project_target_id: str
    protocol_candidate_id: str
    deviations: dict[str, str] = Field(
        default_factory=dict, description="What was actually different from the logged protocol."
    )
    metrics: list[Metric] = Field(default_factory=list)
    run_at: datetime = Field(default_factory=_now)


class BOSuggestion(BaseModel):
    id: str = Field(default_factory=_new_id)
    target_id: str
    protocol_candidate_id: str  # the suggested next protocol
    expected_improvement: float
    uncertainty: float
    rationale: str
    pareto_set_id: Optional[str] = Field(
        default=None, description="Groups suggestions that form one Pareto front (v0.2, multi-objective)."
    )
    generated_at: datetime = Field(default_factory=_now)


class Decision(BaseModel):
    """A record of a choice -- never evidence, never moves a belief. Purely a trajectory log
    (context -> options considered -> chosen -> expected vs. actual outcome)."""

    id: str = Field(default_factory=_new_id)
    context: str
    options_considered: list[str] = Field(default_factory=list)
    chosen_protocol_candidate_id: str
    rationale: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    followed_from: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    def record_outcome(self, actual_outcome: str) -> "Decision":
        return self.model_copy(update={"actual_outcome": actual_outcome})
