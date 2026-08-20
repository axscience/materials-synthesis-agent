from materials_synthesis_agent.literature.outcomes import (
    _extract_param_value,
    outcomes_to_observations,
    protocol_to_params,
)
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec
from materials_synthesis_agent.schema import (
    Citation,
    FieldValue,
    MeasuredOutcome,
    ProtocolCandidate,
    ProtocolSource,
)


def _space():
    return ParameterSpace([
        ParameterSpec("temperature_c", "continuous", bounds=(80.0, 200.0)),
        ParameterSpec("time_hours", "continuous", bounds=(12.0, 96.0)),
        ParameterSpec("solvent", "categorical", categories=("dioxane/mesitylene", "n-BuOH/o-DCB", "DMF")),
    ])


def _fv(val):
    return FieldValue(
        value=val,
        citation=Citation(source_id="10.1234/x", title="T", excerpt="e"),
        inferred=False,
    )


def _candidate(temp="120", time="72", solvent="dioxane/mesitylene 1:1", outcomes=None):
    return ProtocolCandidate(
        target_id="t",
        source=ProtocolSource.LITERATURE,
        building_blocks={},
        solvent=_fv(solvent),
        temperature_c=_fv(temp),
        time_hours=_fv(time),
        measured_outcomes=outcomes or [],
    )


def test_extract_param_value_temperature():
    c = _candidate(temp="120°C")
    assert _extract_param_value(c, "temperature_c") == 120.0


def test_extract_param_value_time():
    c = _candidate(time="72 hours")
    assert _extract_param_value(c, "time_hours") == 72.0


def test_extract_param_value_missing():
    c = _candidate()
    assert _extract_param_value(c, "nonexistent_param") is None


def test_extract_param_value_solvent():
    c = _candidate(solvent="DMF")
    assert _extract_param_value(c, "solvent") == "DMF"


def test_protocol_to_params_maps_all():
    space = _space()
    c = _candidate(temp="120", time="48", solvent="DMF")
    params = protocol_to_params(c, space)
    assert params is not None
    assert params["temperature_c"] == 120.0
    assert params["time_hours"] == 48.0
    assert params["solvent"] == "DMF"


def test_protocol_to_params_clamps_to_bounds():
    space = _space()
    c = _candidate(temp="250", time="48", solvent="DMF")
    params = protocol_to_params(c, space)
    assert params["temperature_c"] == 200.0  # clamped to upper bound


def test_protocol_to_params_returns_none_when_too_many_missing():
    space = ParameterSpace([
        ParameterSpec("temperature_c", "continuous", bounds=(80.0, 200.0)),
        ParameterSpec("time_hours", "continuous", bounds=(12.0, 96.0)),
        ParameterSpec("solvent", "categorical", categories=("dioxane/mesitylene",)),
        ParameterSpec("nonexistent_a", "continuous", bounds=(0.0, 1.0)),
        ParameterSpec("nonexistent_b", "continuous", bounds=(0.0, 1.0)),
    ])
    c = _candidate(temp="120")
    c.time_hours = None
    c.solvent = None
    result = protocol_to_params(c, space)
    assert result is None


def test_protocol_to_params_fills_missing_with_midpoint():
    space = _space()
    c = _candidate(temp="120", time="48")
    c.solvent = None
    params = protocol_to_params(c, space)
    assert params is not None
    assert params["solvent"] == "dioxane/mesitylene"  # first category as default


def test_outcomes_to_observations_matches_metric():
    space = _space()
    outcomes = [
        MeasuredOutcome(metric_name="crystallinity", value=0.72, unit="ratio", measurement_method="PXRD"),
        MeasuredOutcome(metric_name="BET_surface_area", value=410.0, unit="m²/g", measurement_method="N₂"),
    ]
    c = _candidate(temp="120", time="48", solvent="DMF", outcomes=outcomes)
    obs = outcomes_to_observations(c, space, "crystallinity")
    assert len(obs) == 1
    assert obs[0].value == 0.72


def test_outcomes_to_observations_case_insensitive():
    space = _space()
    outcomes = [
        MeasuredOutcome(metric_name="BET_surface_area", value=410.0, unit="m²/g", measurement_method="N₂"),
    ]
    c = _candidate(temp="120", time="48", solvent="DMF", outcomes=outcomes)
    obs = outcomes_to_observations(c, space, "bet surface area")
    assert len(obs) == 1
    assert obs[0].value == 410.0


def test_outcomes_to_observations_empty_when_no_match():
    space = _space()
    outcomes = [
        MeasuredOutcome(metric_name="yield", value=85.0, unit="%", measurement_method="gravimetric"),
    ]
    c = _candidate(temp="120", time="48", solvent="DMF", outcomes=outcomes)
    obs = outcomes_to_observations(c, space, "crystallinity")
    assert obs == []


def test_outcomes_to_observations_preserves_uncertainty():
    space = _space()
    outcomes = [
        MeasuredOutcome(metric_name="crystallinity", value=0.72, unit="ratio",
                        measurement_method="PXRD", uncertainty=0.05),
    ]
    c = _candidate(temp="120", time="48", solvent="DMF", outcomes=outcomes)
    obs = outcomes_to_observations(c, space, "crystallinity")
    assert obs[0].uncertainty == 0.05


def test_outcomes_to_observations_none_uncertainty_when_not_reported():
    space = _space()
    outcomes = [
        MeasuredOutcome(metric_name="crystallinity", value=0.72, unit="ratio", measurement_method="PXRD"),
    ]
    c = _candidate(temp="120", time="48", solvent="DMF", outcomes=outcomes)
    obs = outcomes_to_observations(c, space, "crystallinity")
    assert obs[0].uncertainty is None
