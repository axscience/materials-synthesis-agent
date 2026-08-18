import pytest

from materials_synthesis_agent.cli.params import params_to_new_candidate, protocol_to_params
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec
from materials_synthesis_agent.schema import FieldValue, ProtocolCandidate, ProtocolSource

SPACE = ParameterSpace(
    [
        ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
        ParameterSpec("solvent", "categorical", categories=("dioxane", "mesitylene")),
    ]
)


def test_protocol_to_params_coerces_types():
    candidate = ProtocolCandidate(
        target_id="t1",
        source=ProtocolSource.MANUAL,
        temperature_c=FieldValue(value="120", inferred=True),
        solvent=FieldValue(value="dioxane", inferred=True),
    )
    params = protocol_to_params(candidate, SPACE)
    assert params == {"temperature_c": 120.0, "solvent": "dioxane"}


def test_protocol_to_params_raises_on_missing_field():
    candidate = ProtocolCandidate(target_id="t1", source=ProtocolSource.MANUAL, solvent=FieldValue(value="dioxane", inferred=True))
    with pytest.raises(ValueError):
        protocol_to_params(candidate, SPACE)  # temperature_c missing


def test_protocol_to_params_raises_on_unparsable_numeric_field():
    candidate = ProtocolCandidate(
        target_id="t1",
        source=ProtocolSource.MANUAL,
        temperature_c=FieldValue(value="not a number", inferred=True),
        solvent=FieldValue(value="dioxane", inferred=True),
    )
    with pytest.raises(ValueError):
        protocol_to_params(candidate, SPACE)


def test_params_to_new_candidate_flags_every_field_inferred():
    candidate = params_to_new_candidate({"temperature_c": 90.0, "solvent": "mesitylene"}, target_id="t1")
    assert candidate.source == ProtocolSource.BO_SUGGESTED
    assert candidate.temperature_c.inferred is True
    assert candidate.temperature_c.citation is None
    assert candidate.solvent.value == "mesitylene"
