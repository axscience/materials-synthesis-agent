"""Markdown session export -- the lab-notebook log, generated from the structured store."""

from materials_synthesis_agent.harness import Campaign, CampaignStore, Session, Turn
from materials_synthesis_agent.harness.session_export import session_to_markdown
from materials_synthesis_agent.schema import (
    Experiment,
    Metric,
    ProtocolCandidate,
    ProtocolSource,
    Target,
)


def _fixture(tmp_path):
    store = CampaignStore(tmp_path / "campaign.db")
    target = Target(name="COF-X", functional_groups=["amine"], linkage_chemistry="imine",
                    application="gas separation", metric_name="crystallinity",
                    metric_measurement_method="PXRD")
    store.save_target(target)
    cand = ProtocolCandidate(target_id=target.id, source=ProtocolSource.MANUAL)
    store.save_protocol_candidate(cand)
    store.save_experiment(Experiment(
        project_target_id=target.id, protocol_candidate_id=cand.id,
        metrics=[Metric(name="crystallinity", value=0.72, uncertainty=0.04, measurement_method="PXRD")],
    ))
    campaign = Campaign(name="imine-cof-gas-sep", target_id=target.id, linkage_chemistry="imine")
    store.save_campaign(campaign)
    return store, campaign


def test_markdown_has_header_state_and_conversation(tmp_path):
    store, campaign = _fixture(tmp_path)
    session = Session(campaign_id=campaign.id, turns=[
        Turn(role="user", content="I measured 0.72 crystallinity"),
        Turn(role="assistant", content="Logged it.",
             tool_calls=[{"name": "store.log_result", "is_error": False, "cost": 0.0}]),
    ])
    md = session_to_markdown(store, campaign, session)

    assert "# Session —" in md
    assert "imine-cof-gas-sep" in md
    assert "## State at start" in md
    assert "Best crystallinity so far: 0.72 ± 0.04" in md
    assert "**You:** I measured 0.72 crystallinity" in md
    assert "store.log_result" in md
    assert "## State at end" in md
    store.close()


def test_markdown_flags_tool_errors(tmp_path):
    store, campaign = _fixture(tmp_path)
    session = Session(campaign_id=campaign.id, turns=[
        Turn(role="user", content="generate candidates"),
        Turn(role="assistant", content="Not available.",
             tool_calls=[{"name": "design.generate_candidates", "is_error": True}]),
    ])
    md = session_to_markdown(store, campaign, session)
    assert "design.generate_candidates (error)" in md
    store.close()


def test_markdown_handles_empty_session(tmp_path):
    store, campaign = _fixture(tmp_path)
    md = session_to_markdown(store, campaign, Session(campaign_id=campaign.id))
    assert "_(no turns recorded)_" in md
    store.close()
