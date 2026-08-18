import pytest
from pydantic import ValidationError

from materials_synthesis_agent.schema import (
    Citation,
    FieldValue,
    Metric,
    MetricConfidence,
    ObjectiveDirection,
    Target,
)


def _target(**overrides):
    base = dict(
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )
    base.update(overrides)
    return Target(**base)


def test_target_defaults_to_maximize():
    t = _target()
    assert t.objective_direction == ObjectiveDirection.MAXIMIZE
    assert t.maximize is True


def test_target_minimize_sets_maximize_false():
    t = _target(objective_direction=ObjectiveDirection.MINIMIZE)
    assert t.objective_direction == ObjectiveDirection.MINIMIZE
    assert t.maximize is False


def test_target_direction_round_trips_through_json():
    t = _target(objective_direction=ObjectiveDirection.MINIMIZE)
    restored = Target.model_validate_json(t.model_dump_json())
    assert restored.maximize is False


def test_field_value_requires_citation_or_inferred():
    with pytest.raises(ValidationError):
        FieldValue(value="120", citation=None, inferred=False)


def test_field_value_with_citation_is_valid():
    fv = FieldValue(value="120", citation=Citation(source_id="10.1/x", title="paper"))
    assert fv.citation is not None
    assert fv.inferred is False


def test_field_value_inferred_without_citation_is_valid():
    fv = FieldValue(value="120", inferred=True)
    assert fv.citation is None
    assert fv.inferred is True


def test_metric_confidence_quantified_when_uncertainty_given():
    m = Metric(name="crystallinity", value=0.8, uncertainty=0.05, measurement_method="PXRD")
    assert m.confidence == MetricConfidence.QUANTIFIED


def test_metric_confidence_low_when_uncertainty_missing():
    m = Metric(name="crystallinity", value=0.8, uncertainty=None, measurement_method="PXRD")
    assert m.confidence == MetricConfidence.LOW_CONFIDENCE


def test_protocol_candidate_citation_coverage():
    from materials_synthesis_agent.schema import ProtocolCandidate, ProtocolSource

    candidate = ProtocolCandidate(
        target_id="t1",
        source=ProtocolSource.LITERATURE,
        solvent=FieldValue(value="dioxane", citation=Citation(source_id="x", title="p")),
        modulator=FieldValue(value="aniline", inferred=True),
    )
    assert candidate.citation_coverage() == 0.5


def test_protocol_candidate_citation_coverage_empty_is_zero():
    from materials_synthesis_agent.schema import ProtocolCandidate, ProtocolSource

    candidate = ProtocolCandidate(target_id="t1", source=ProtocolSource.MANUAL)
    assert candidate.citation_coverage() == 0.0
