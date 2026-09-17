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
