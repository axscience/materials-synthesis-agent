"""Tests for extracting a paper's (conditions -> outcome) experiments and seeding the GP with the
subset the user selects."""

from materials_synthesis_agent.literature.extraction import EXTRACTION_TOOL_SCHEMA, extract_protocol
from materials_synthesis_agent.literature.outcomes import (
    experiment_to_params,
    selected_experiments_to_observations,
)
from materials_synthesis_agent.literature.retrieval import Paper
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec
from materials_synthesis_agent.schema import (
    Citation,
    FieldValue,
    LiteratureExperiment,
    MeasuredOutcome,
    ProtocolCandidate,
    ProtocolSource,
    Target,
)

SPACE = ParameterSpace([
    ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
    ParameterSpec("solvent", "categorical", categories=("dioxane", "mesitylene", "DMAc")),
])
CIT = Citation(source_id="10.x", title="p")


def _target():
    return Target(name="X", functional_groups=["imine"], linkage_chemistry="imine condensation",
                  application="x", metric_name="crystallinity", metric_measurement_method="PXRD peak area ratio")


class _Bridge:
    def __init__(self, out): self._out = out
    def call_tool(self, **kw): return self._out
    def chat(self, *a, **k): raise NotImplementedError


def test_extraction_schema_has_experiments():
    assert "experiments" in EXTRACTION_TOOL_SCHEMA["properties"]


def test_extract_parses_experiments_and_skips_outcomeless_rows():
    paper = Paper(source_id="10.1/x", title="opt paper", abstract="...", year=2023, url="http://x", source="openalex")
    out = {
        "found_protocol": True,
        "building_blocks": {"TFB": {"value": "O=Cc1cc(C=O)cc(C=O)c1", "excerpt": "TFB", "inferred": False}},
        "experiments": [
            {"label": "T1e1", "conditions": {"solvent": {"value": "dioxane", "excerpt": "dioxane", "inferred": False},
                                             "temperature_c": {"value": "120", "excerpt": "120 C", "inferred": False}},
             "outcome": {"metric_name": "crystallinity", "value": 0.62, "unit": "ratio",
                         "measurement_method": "PXRD peak area ratio", "excerpt": "0.62", "inferred": False}},
            {"label": "T1e2", "conditions": {"solvent": {"value": "mesitylene", "excerpt": "mes", "inferred": False}},
             "outcome": {"metric_name": "crystallinity", "value": 0.91, "unit": "ratio",
                         "measurement_method": "PXRD peak area ratio", "excerpt": "0.91", "inferred": False}},
            {"conditions": {}, "outcome": None},  # no outcome -> skipped
        ],
    }
    cand = extract_protocol(_target(), paper, client=_Bridge(out))
    assert len(cand.literature_experiments) == 2
    assert all(not e.selected_for_seeding for e in cand.literature_experiments)  # off until chosen
    assert cand.literature_experiments[0].outcome.value == 0.62
    assert cand.literature_experiments[0].outcome.citation is not None  # grounded via excerpt


def _experiment(sel, solvent, temp, val, metric="crystallinity"):
    return LiteratureExperiment(
        conditions={"solvent": FieldValue(value=solvent, citation=CIT),
                    "temperature_c": FieldValue(value=str(temp), citation=CIT)},
        outcome=MeasuredOutcome(metric_name=metric, value=val, unit="ratio", measurement_method="PXRD", citation=CIT),
        selected_for_seeding=sel,
    )


def test_only_selected_matching_experiments_become_observations():
    cand = ProtocolCandidate(target_id="t", source=ProtocolSource.LITERATURE, literature_experiments=[
        _experiment(True, "dioxane", 120, 0.62),
        _experiment(True, "mesitylene", 100, 0.91),
        _experiment(False, "DMAc", 80, 0.40),          # not selected
        _experiment(True, "dioxane", 90, 1200, metric="BET_surface_area"),  # wrong metric
    ])
    obs = selected_experiments_to_observations(cand, SPACE, "crystallinity")
    assert len(obs) == 2
    assert {round(o.value, 2) for o in obs} == {0.62, 0.91}


def test_experiment_conditions_override_protocol_base():
    # protocol says dioxane/120; this experiment overrides to mesitylene/100 -> params reflect the row.
    cand = ProtocolCandidate(
        target_id="t", source=ProtocolSource.LITERATURE,
        solvent=FieldValue(value="dioxane", inferred=True),
        temperature_c=FieldValue(value="120", inferred=True),
        literature_experiments=[_experiment(True, "mesitylene", 100, 0.91)],
    )
    params = experiment_to_params(cand, cand.literature_experiments[0], SPACE)
    assert params["solvent"] == "mesitylene"
    assert params["temperature_c"] == 100.0


def test_no_selection_yields_no_seeds():
    cand = ProtocolCandidate(target_id="t", source=ProtocolSource.LITERATURE,
                             literature_experiments=[_experiment(False, "dioxane", 120, 0.62)])
    assert selected_experiments_to_observations(cand, SPACE, "crystallinity") == []
