import pytest
from pydantic import ValidationError

from materials_synthesis_agent.schema import Citation, FieldValue, Metric, MetricConfidence


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
