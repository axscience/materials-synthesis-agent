"""Tool registry -- the torch-free handlers, the flywheel feed, and the design-guard."""

import pytest

from materials_synthesis_agent.harness import Campaign, CampaignStore
from materials_synthesis_agent.harness.tools import ToolContext, ToolError, run_tool
from materials_synthesis_agent.harness.warm_prior import PriorStore
from materials_synthesis_agent.schema import (
    FieldValue,
    ProtocolCandidate,
    ProtocolSource,
    Target,
)


def _ctx(tmp_path):
    store = CampaignStore(tmp_path / "campaign.db")
    priors = PriorStore(tmp_path / "priors.db")
    target = Target(
        name="COF-X", functional_groups=["amine", "aldehyde"], linkage_chemistry="imine",
        application="gas storage", metric_name="crystallinity", metric_measurement_method="PXRD",
    )
    store.save_target(target)
    cand = ProtocolCandidate(
        target_id=target.id, source=ProtocolSource.MANUAL,
        building_blocks={"TAPB": FieldValue(value="Nc1ccccc1", inferred=True)},
        temperature_c=FieldValue(value="120", inferred=True),
        solvent=FieldValue(value="dioxane", inferred=True),
    )
    store.save_protocol_candidate(cand)
    campaign = Campaign(name="c", target_id=target.id, linkage_chemistry="imine",
                        space={"specs": [
                            {"name": "temperature_c", "kind": "continuous", "bounds": [20, 150]},
                            {"name": "solvent", "kind": "categorical", "categories": ["dioxane", "mesitylene"]},
                        ]})
    store.save_campaign(campaign)
    return ToolContext(store=store, prior_store=priors, campaign=campaign), target, cand


def test_setup_space_persists_and_summarizes(tmp_path):
    ctx, _, _ = _ctx(tmp_path)
    out, gate = run_tool(ctx, "opt.setup_space", {"specs": [
        {"name": "time_hours", "kind": "continuous", "bounds": [1, 96]},
    ]})
    assert gate.blocked is False
    assert out["parameters"] == ["time_hours"]
    assert ctx.store.get_campaign(ctx.campaign.id).space["specs"][0]["name"] == "time_hours"
    ctx.store.close(); ctx.prior_store.close()


def test_setup_space_rejects_bad_spec(tmp_path):
    ctx, _, _ = _ctx(tmp_path)
    with pytest.raises(ToolError):
        run_tool(ctx, "opt.setup_space", {"specs": [{"name": "t", "kind": "continuous"}]})
    ctx.store.close(); ctx.prior_store.close()


def test_log_result_saves_experiment_and_feeds_prior(tmp_path):
    ctx, target, cand = _ctx(tmp_path)
    out, gate = run_tool(ctx, "store.log_result", {
        "protocol_id": cand.id,
        "metrics": [{"name": "crystallinity", "value": 0.72, "uncertainty": 0.04}],
        "notes": "PXRD matches predicted topology",
    })
    assert gate.blocked is False
    assert out["recorded_to_prior"] is True

    # Experiment persisted, characterization created, prior fed.
    assert len(ctx.store.list_experiments(target.id)) == 1
    assert len(ctx.store.list_characterizations(ctx.campaign.id)) == 1
    wp = ctx.prior_store._rows("imine", "crystallinity")
    assert len(wp) == 1
    assert wp[0][0]["temperature_c"] == 120.0 and wp[0][0]["solvent"] == "dioxane"
    ctx.store.close(); ctx.prior_store.close()


def test_warm_prior_handler_reflects_logged_results(tmp_path):
    ctx, target, cand = _ctx(tmp_path)
    run_tool(ctx, "store.log_result", {"protocol_id": cand.id,
             "metrics": [{"name": "crystallinity", "value": 0.72, "uncertainty": 0.04}]})
    out, _ = run_tool(ctx, "data.warm_prior", {})
    assert out["n_prior_experiments"] == 1
    assert out["anchor_params"][0]["solvent"] == "dioxane"
    ctx.store.close(); ctx.prior_store.close()


def test_design_tools_guarded_when_package_absent(tmp_path):
    ctx, _, _ = _ctx(tmp_path)
    with pytest.raises(ToolError):
        run_tool(ctx, "design.generate_candidates", {})
    ctx.store.close(); ctx.prior_store.close()


def test_unknown_tool_raises_keyerror(tmp_path):
    ctx, _, _ = _ctx(tmp_path)
    with pytest.raises(KeyError):
        run_tool(ctx, "nope.nonexistent", {})
    ctx.store.close(); ctx.prior_store.close()


def test_extract_protocols_requires_llm_client(tmp_path):
    ctx, _, _ = _ctx(tmp_path)  # ctx.llm is None
    with pytest.raises(ToolError):
        run_tool(ctx, "lit.extract_protocols", {"n": 3})
    ctx.store.close(); ctx.prior_store.close()


def test_suggest_next_observations_seed_from_literature(tmp_path):
    """The extractor's experiments seed the GP: with 2 selected literature experiments and NO
    user-logged results, the optimizer still has 2 observations to start from -- this is what lets
    it propose a first protocol from the papers' real data."""
    from materials_synthesis_agent.harness.tools import _build_observations, _build_parameter_space
    from materials_synthesis_agent.schema import LiteratureExperiment, MeasuredOutcome

    ctx, target, _ = _ctx(tmp_path)
    lit = ProtocolCandidate(
        target_id=target.id, source=ProtocolSource.LITERATURE,
        building_blocks={"TAPB": FieldValue(value="Nc1ccccc1", inferred=True)},
        literature_experiments=[
            LiteratureExperiment(
                label="row1",
                conditions={"temperature_c": FieldValue(value="120", inferred=True),
                            "solvent": FieldValue(value="dioxane", inferred=True)},
                outcome=MeasuredOutcome(metric_name="crystallinity", value=0.65, unit="ratio",
                                        measurement_method="PXRD"),
                selected_for_seeding=True,
            ),
            LiteratureExperiment(
                label="row2",
                conditions={"temperature_c": FieldValue(value="130", inferred=True),
                            "solvent": FieldValue(value="dioxane", inferred=True)},
                outcome=MeasuredOutcome(metric_name="crystallinity", value=0.82, unit="ratio",
                                        measurement_method="PXRD"),
                selected_for_seeding=True,
            ),
        ],
    )
    ctx.store.save_protocol_candidate(lit)

    space = _build_parameter_space(ctx.campaign.space["specs"])
    obs = _build_observations(ctx, space, "crystallinity")
    assert len(obs) == 2                       # both literature rows became observations
    assert {round(o.value, 2) for o in obs} == {0.65, 0.82}
    ctx.store.close(); ctx.prior_store.close()


def test_suggest_next_observations_seed_from_single_measured_outcomes(tmp_path):
    """Most COF papers report a single headline value (a measured_outcome), not an optimization
    table. Those must seed the GP too -- the live smoke test showed experiment-only seeding leaves
    it empty. Here two candidates each report a BET value at clean conditions -> two observations."""
    from materials_synthesis_agent.harness.tools import _build_observations, _build_parameter_space
    from materials_synthesis_agent.schema import MeasuredOutcome

    ctx, target, _ = _ctx(tmp_path)  # target metric is "crystallinity"; use a BET campaign here
    # Re-point the campaign's target metric to BET for this test.
    bet_target = Target(
        name="COF-BET", functional_groups=["amine", "aldehyde"], linkage_chemistry="imine",
        application="gas storage", metric_name="BET_surface_area",
        metric_measurement_method="N2 adsorption",
    )
    ctx.store.save_target(bet_target)
    ctx.campaign.target_id = bet_target.id
    ctx.store.save_campaign(ctx.campaign)

    for temp, bet in [("120", 1457.0), ("90", 800.0)]:
        c = ProtocolCandidate(
            target_id=bet_target.id, source=ProtocolSource.LITERATURE,
            building_blocks={"BB": FieldValue(value="Nc1ccccc1", inferred=True)},
            temperature_c=FieldValue(value=temp, inferred=True),
            solvent=FieldValue(value="dioxane", inferred=True),
            measured_outcomes=[MeasuredOutcome(metric_name="BET_surface_area", value=bet,
                                               unit="m2/g", measurement_method="N2 adsorption")],
        )
        ctx.store.save_protocol_candidate(c)

    space = _build_parameter_space(ctx.campaign.space["specs"])
    obs = _build_observations(ctx, space, "BET_surface_area")
    assert len(obs) == 2
    assert {round(o.value) for o in obs} == {1457, 800}
    ctx.store.close(); ctx.prior_store.close()


def test_calibration_correction_widens_when_overconfident():
    from types import SimpleNamespace
    from materials_synthesis_agent.harness.tools import _calibration_correction
    scale, note = _calibration_correction(SimpleNamespace(passes=False, variance_scale=5.8, coverage_80=0.6))
    assert scale == 5.8
    assert "widened" in note and "80%" in note


def test_calibration_correction_noop_when_passing_or_insufficient():
    from types import SimpleNamespace
    from materials_synthesis_agent.harness.tools import _calibration_correction
    assert _calibration_correction(SimpleNamespace(passes=True, variance_scale=1.1, coverage_80=0.82)) == (1.0, "")
    assert _calibration_correction(SimpleNamespace(passes=None, variance_scale=None, coverage_80=None)) == (1.0, "")
    assert _calibration_correction(SimpleNamespace(passes=False, variance_scale=None, coverage_80=0.5)) == (1.0, "")


def test_extrapolation_note_flags_out_of_range_only():
    from types import SimpleNamespace
    from materials_synthesis_agent.harness.tools import _extrapolation_note
    obs = [SimpleNamespace(params={"temperature_c": 100.0}),
           SimpleNamespace(params={"temperature_c": 120.0})]
    specs = [{"name": "temperature_c", "kind": "continuous", "bounds": [20, 180]}]
    assert "temperature_c" in _extrapolation_note({"temperature_c": 160.0}, obs, specs)  # above data
    assert _extrapolation_note({"temperature_c": 110.0}, obs, specs) == ""               # within data
    assert _extrapolation_note({"temperature_c": 160.0}, [], specs) == ""                # no data -> no claim


def test_derive_space_from_extracted_conditions(tmp_path):
    """opt.derive_space builds the parameter space from what the protocols actually report."""
    from materials_synthesis_agent.schema import FieldValue
    ctx, target, _ = _ctx(tmp_path)
    for temp, solv in [("100", "dioxane"), ("120", "mesitylene"), ("130", "dioxane")]:
        ctx.store.save_protocol_candidate(ProtocolCandidate(
            target_id=target.id, source=ProtocolSource.LITERATURE,
            building_blocks={"BB": FieldValue(value="Nc1ccccc1", inferred=True)},
            temperature_c=FieldValue(value=temp, inferred=True),
            solvent=FieldValue(value=solv, inferred=True),
        ))
    out, gate = run_tool(ctx, "opt.derive_space", {})
    assert gate.blocked is False
    assert "temperature_c" in out["parameters"]
    # The campaign now carries the derived space.
    assert any(s["name"] == "temperature_c" for s in ctx.store.get_campaign(ctx.campaign.id).space["specs"])
    ctx.store.close(); ctx.prior_store.close()


def test_derive_space_errors_without_enough_protocols(tmp_path):
    ctx, _, _ = _ctx(tmp_path)  # only the one fixture candidate, no conditions on it
    with pytest.raises(ToolError):
        run_tool(ctx, "opt.derive_space", {})
    ctx.store.close(); ctx.prior_store.close()
