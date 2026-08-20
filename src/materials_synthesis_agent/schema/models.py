"""Versioned, portable data models -- the contract between this package and any consumer
(the local CLI or the local web UI).

Invariants this module enforces:
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
                "presenting an extracted value as sourced fact with no traceable source."
            )
        return self


class TargetObjective(BaseModel):
    """One metric to optimize, with how it's measured and which way to push it. A Target with two
    or more of these is a multi-objective (Pareto) problem."""

    name: str
    measurement_method: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    @property
    def maximize(self) -> bool:
        return self.direction == ObjectiveDirection.MAXIMIZE


class Target(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: Optional[str] = Field(
        default=None,
        description="A specific COF name (e.g. 'COF-5'), when the researcher is targeting a known "
        "material rather than describing one by functional groups/linkage chemistry. When set, "
        "the literature search queries by name directly -- see literature.agent.build_query.",
    )
    functional_groups: list[str]
    linkage_chemistry: str
    application: str
    metric_name: str
    metric_measurement_method: str
    objective_direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    # Empty for single-objective projects (the metric_name/objective_direction fields above are the
    # source of truth then). Populated with 2+ entries for a multi-objective/Pareto project. Read
    # via `all_objectives`, never directly, so both cases are handled uniformly.
    objectives: list[TargetObjective] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)

    @property
    def maximize(self) -> bool:
        return self.objective_direction == ObjectiveDirection.MAXIMIZE

    @property
    def all_objectives(self) -> list[TargetObjective]:
        """The objectives to optimize, uniformly. Falls back to the single legacy metric fields
        when `objectives` is empty, so single-objective projects (including ones saved before
        multi-objective existed) keep working unchanged."""
        if self.objectives:
            return self.objectives
        return [
            TargetObjective(
                name=self.metric_name,
                measurement_method=self.metric_measurement_method,
                direction=self.objective_direction,
            )
        ]

    @property
    def is_multi_objective(self) -> bool:
        return len(self.all_objectives) > 1


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
        default_factory=dict, description="monomer abbreviation -> SMILES, with provenance"
    )
    monomer_roles: dict[str, FieldValue] = Field(
        default_factory=dict, description="monomer abbreviation -> structural role (node/linker/core)"
    )
    stoichiometry: dict[str, FieldValue] = Field(default_factory=dict)
    synthesis_method: Optional[FieldValue] = None
    solvent: Optional[FieldValue] = None
    catalyst: Optional[FieldValue] = None
    modulator: Optional[FieldValue] = None
    temperature_c: Optional[FieldValue] = None
    time_hours: Optional[FieldValue] = None
    concentration_molar: Optional[FieldValue] = None
    atmosphere: Optional[FieldValue] = None
    activation_method: Optional[FieldValue] = None
    purification: Optional[FieldValue] = None
    yield_percent: Optional[FieldValue] = None
    characterization_notes: Optional[FieldValue] = None

    measured_outcomes: list[MeasuredOutcome] = Field(
        default_factory=list,
        description="Quantitative results reported in the same paper as the protocol — "
        "PXRD crystallinity, BET surface area, yield, etc. Real measurements that can "
        "seed the GP as observations.",
    )

    feasibility_flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)

    def all_fields(self) -> list[FieldValue]:
        """Every FieldValue on this candidate, for grounding audits and UI rendering."""
        out: list[FieldValue] = list(self.building_blocks.values()) + list(
            self.monomer_roles.values()
        ) + list(self.stoichiometry.values())
        for f in (self.synthesis_method, self.solvent, self.catalyst, self.modulator,
                  self.temperature_c, self.time_hours, self.concentration_molar,
                  self.atmosphere, self.activation_method, self.purification,
                  self.yield_percent, self.characterization_notes):
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


class StructureMatch(BaseModel):
    """A CIF matched against a known-COF structure database (structure/database.py), citation-backed
    via the database's own paper mapping -- never handed to Target.name without the caller
    surfacing `source_database`/`matched_id` alongside it. Distinct from a guess: a match here
    traces to a specific, already-published structure, the same grounding standard `Citation`
    enforces for literature-extracted fields."""

    matched_name: str
    matched_id: str
    source_database: str
    paper_doi: Optional[str] = None
    paper_title: Optional[str] = None


class LinkageClassification(BaseModel):
    """A structural guess at linkage chemistry from bond-graph analysis (structure/linkage.py),
    carrying its own evidence -- never handed to Target.linkage_chemistry without the caller
    surfacing `confidence` and `evidence` alongside it, the same discipline FieldValue enforces for
    literature-extracted fields. `linkage_chemistry=""` and `confidence=0.0` means nothing matched,
    not that a guess was suppressed."""

    linkage_chemistry: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class MeasuredOutcome(BaseModel):
    """A quantitative result reported in a paper alongside its synthesis protocol — e.g. a PXRD
    crystallinity ratio, a BET surface area, or an isolated yield. These are real, published
    measurements that can seed the GP as genuine observations (with appropriate noise), unlike
    LiteratureAnchors which only bias the search start.

    `uncertainty` is often not reported in papers; when absent, the optimizer assigns a conservative
    default noise to downweight the observation rather than trusting it blindly."""

    metric_name: str = Field(description="e.g. 'crystallinity', 'BET_surface_area', 'yield'")
    value: float
    unit: str = Field(description="e.g. 'm²/g', '%', 'ratio'")
    uncertainty: Optional[float] = None
    measurement_method: str = Field(description="e.g. 'PXRD peak area ratio', 'N₂ adsorption at 77 K'")
    citation: Optional[Citation] = None
    inferred: bool = False


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
    expected_improvement: float = 0.0  # single-objective only; 0 for a Pareto-set member
    uncertainty: float = 0.0
    rationale: str
    pareto_set_id: Optional[str] = Field(
        default=None, description="Groups suggestions that form one Pareto front (multi-objective)."
    )
    predicted_values: Optional[dict[str, float]] = Field(
        default=None, description="objective name -> predicted value (multi-objective Pareto members)."
    )
    generated_at: datetime = Field(default_factory=_now)


class ParsedRequest(BaseModel):
    """What an LLM understood from a researcher's free-text request (`nl.parser.parse_request`).

    Same discipline as `FieldValue`, adapted from "grounded in a paper" to "grounded in what the
    user typed": a field the model filled in without the user actually stating it must be named in
    `inferred_fields`, and anything essential the model could not determine -- and did NOT guess at
    -- goes in `clarifications_needed` instead. A `ParsedRequest` is an intermediate, editable
    object; nothing downstream should treat it as a settled `Target` without the caller checking
    these two lists first.
    """

    raw_text: str
    cof_name: Optional[str] = None
    cif_path: Optional[str] = Field(
        default=None,
        description="A file path mentioned in the text. Validated to exist on disk by the parser "
        "before being trusted -- the model can misread a path out of prose, so an unverified path "
        "is never handed downstream as if it were real.",
    )
    functional_groups: list[str] = Field(default_factory=list)
    linkage_chemistry: Optional[str] = None
    application: Optional[str] = None
    objectives: list[TargetObjective] = Field(default_factory=list)
    inferred_fields: list[str] = Field(
        default_factory=list, description="Names of fields above the model filled in by inference/default rather than reading directly from the text."
    )
    clarifications_needed: list[str] = Field(
        default_factory=list, description="Plain-language questions about anything essential that's missing or ambiguous, which the model did NOT guess at."
    )


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
