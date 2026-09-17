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
