"""Planner loop -- driven by a scripted FakeToolCaller that exercises the REAL dispatch (handlers +
gates), so the orchestration is tested end-to-end without a live LLM."""

from materials_synthesis_agent.harness import Campaign, CampaignStore, Session
from materials_synthesis_agent.harness.planner import Planner, PlannerResult
from materials_synthesis_agent.harness.tools import ToolContext
from materials_synthesis_agent.harness.warm_prior import PriorStore
from materials_synthesis_agent.schema import (
    FieldValue,
    ProtocolCandidate,
    ProtocolSource,
    Target,
)


class FakeToolCaller:
    """Scripts a fixed sequence of tool calls, then a final text -- but runs them through the real
    dispatch passed in by the Planner."""

    def __init__(self, calls, final_text):
        self._calls = calls
        self._final = final_text
        self.seen = []

    def run(self, system, messages, tools, dispatch, max_turns=12) -> PlannerResult:
        tool_calls = []
        for name, inp in self._calls:
            outcome = dispatch(name, inp)
            self.seen.append((name, outcome.is_error))
            tool_calls.append({"name": name, "is_error": outcome.is_error, "cost": outcome.cost})
        return PlannerResult(final_text=self._final, tool_calls=tool_calls, turns_used=1)


def _ctx(tmp_path):
    store = CampaignStore(tmp_path / "campaign.db")
    priors = PriorStore(tmp_path / "priors.db")
    target = Target(name="COF-X", functional_groups=["amine", "aldehyde"], linkage_chemistry="imine",
                    application="gas storage", metric_name="crystallinity", metric_measurement_method="PXRD")
    store.save_target(target)
    cand = ProtocolCandidate(
        target_id=target.id, source=ProtocolSource.MANUAL,
        building_blocks={"TAPB": FieldValue(value="Nc1ccccc1", inferred=True)},
        temperature_c=FieldValue(value="120", inferred=True),
        solvent=FieldValue(value="dioxane", inferred=True),
    )
    store.save_protocol_candidate(cand)
    campaign = Campaign(name="c", target_id=target.id, linkage_chemistry="imine",
                        space={"specs": [{"name": "temperature_c", "kind": "continuous", "bounds": [20, 150]},
                                         {"name": "solvent", "kind": "categorical", "categories": ["dioxane", "mesitylene"]}]})
    store.save_campaign(campaign)
    return ToolContext(store=store, prior_store=priors, campaign=campaign), target, cand


def test_planner_runs_tool_and_records_session(tmp_path):
    ctx, target, cand = _ctx(tmp_path)
    caller = FakeToolCaller(
        [("store.log_result", {"protocol_id": cand.id,
                               "metrics": [{"name": "crystallinity", "value": 0.7, "uncertainty": 0.05}]})],
        "Logged your result — crystallinity 0.70 ± 0.05.",
    )
    session = Session(campaign_id=ctx.campaign.id)
    planner = Planner(ctx, caller)

    out = planner.respond("I measured 0.70 crystallinity on candidate 1", session)

    assert out.startswith("Logged your result")
    assert len(session.turns) == 2                          # user + assistant
    assert session.turns[1].tool_calls[0]["name"] == "store.log_result"
    assert session.turns[1].tool_calls[0]["is_error"] is False
    assert len(ctx.store.list_experiments(target.id)) == 1  # the tool actually ran
    assert session.id in ctx.store.get_campaign(ctx.campaign.id).session_ids
    ctx.store.close(); ctx.prior_store.close()


def test_planner_surfaces_tool_error_without_crashing(tmp_path):
    ctx, _, _ = _ctx(tmp_path)
    caller = FakeToolCaller(
        [("design.generate_candidates", {})],
        "Design candidate generation isn't available in this environment.",
    )
    planner = Planner(ctx, caller)
    out = planner.respond("Generate some candidates", Session(campaign_id=ctx.campaign.id))

    assert caller.seen == [("design.generate_candidates", True)]  # error surfaced, not raised
    assert "isn't available" in out
    ctx.store.close(); ctx.prior_store.close()


def test_planner_reports_unknown_tool_as_error(tmp_path):
    ctx, _, _ = _ctx(tmp_path)
    caller = FakeToolCaller([("bogus.tool", {})], "done")
    planner = Planner(ctx, caller)
    planner.respond("do something", None)
    assert caller.seen == [("bogus.tool", True)]
    ctx.store.close(); ctx.prior_store.close()
