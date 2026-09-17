"""Campaign state + CampaignStore persistence, and the PropertyComparison math (the one bit of
scoring logic in this layer, so it gets the hardest test per TESTING.md)."""

import pytest

from materials_synthesis_agent.harness import (
    Campaign,
    CampaignStatus,
    CampaignStore,
    CharacterizationResult,
    PropertyComparison,
    Session,
    SynthesisOutcome,
    Turn,
)
from materials_synthesis_agent.schema import Target


def _store(tmp_path) -> CampaignStore:
    return CampaignStore(tmp_path / "campaign.db")


def test_campaign_round_trip(tmp_path):
    store = _store(tmp_path)
    c = Campaign(name="imine-cof-gas-sep", linkage_chemistry="imine")
    store.save_campaign(c)

    loaded = store.get_campaign(c.id)
    assert loaded is not None
    assert loaded.name == "imine-cof-gas-sep"
    assert loaded.status == CampaignStatus.ACTIVE
    assert loaded.linkage_chemistry == "imine"
    store.close()


def test_list_campaigns_newest_first(tmp_path):
    store = _store(tmp_path)
    a = Campaign(name="a")
    b = Campaign(name="b")
    store.save_campaign(a)
    store.save_campaign(b)
    b.status = CampaignStatus.PAUSED
    store.save_campaign(b)  # touch b -> most recently updated

    names = [c.name for c in store.list_campaigns()]
    assert names[0] == "b"  # updated most recently
    assert set(names) == {"a", "b"}
    store.close()


def test_sessions_scoped_to_campaign(tmp_path):
    store = _store(tmp_path)
    c1, c2 = Campaign(name="c1"), Campaign(name="c2")
    store.save_campaign(c1)
    store.save_campaign(c2)

    s1 = Session(campaign_id=c1.id, turns=[Turn(role="user", content="hi", cost=0.01)])
    s2 = Session(campaign_id=c1.id)
    s_other = Session(campaign_id=c2.id)
    for s in (s1, s2, s_other):
        store.save_session(s)

    got = store.list_sessions(c1.id)
    assert {s.id for s in got} == {s1.id, s2.id}  # c2's session excluded
    assert store.get_session(s1.id).total_cost == pytest.approx(0.01)
    store.close()


def test_characterizations_scoped_to_campaign(tmp_path):
    store = _store(tmp_path)
    c = Campaign(name="c")
    store.save_campaign(c)
    char = CharacterizationResult(experiment_id="exp-1", outcome=SynthesisOutcome.PARTIAL)
    store.save_characterization(c.id, char)

    got = store.list_characterizations(c.id)
    assert len(got) == 1
    assert got[0].experiment_id == "exp-1"
    assert got[0].outcome == SynthesisOutcome.PARTIAL
    store.close()


def test_subclass_does_not_break_parent_store(tmp_path):
    """CampaignStore must still be a working Store -- the loop data stays in the same file."""
    store = _store(tmp_path)
    target = Target(
        name="COF-X",
        functional_groups=["amine", "aldehyde"],
        linkage_chemistry="imine",
        application="gas storage",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )
    store.save_target(target)
    assert store.get_target(target.id).name == "COF-X"
    store.close()


# --- PropertyComparison math (predicted vs measured) ---

def test_property_comparison_within_one_sigma():
    pc = PropertyComparison.build(
        "BET_surface_area", measured=1820, predicted=1840,
        predicted_uncertainty=50, measured_uncertainty=10,
    )
    # |1820-1840| = 20; spread = sqrt(50^2+10^2) ~= 51 -> ~0.39 sigma -> within
    assert pc.within_prediction is True
    assert pc.deviation_sigma == pytest.approx(20 / (50**2 + 10**2) ** 0.5, rel=1e-6)


def test_property_comparison_outside_prediction():
    pc = PropertyComparison.build(
        "BET_surface_area", measured=1200, predicted=1840,
        predicted_uncertainty=50, measured_uncertainty=10,
    )
    assert pc.within_prediction is False
    assert pc.deviation_sigma > 1.0


def test_property_comparison_zero_spread_is_exact_match_only():
    exact = PropertyComparison.build("yield", measured=85.0, predicted=85.0,
                                     predicted_uncertainty=0, measured_uncertainty=0)
    off = PropertyComparison.build("yield", measured=84.0, predicted=85.0,
                                   predicted_uncertainty=0, measured_uncertainty=0)
    assert exact.within_prediction is True
    assert off.within_prediction is False
    assert exact.deviation_sigma is None  # no spread -> sigma undefined, not fabricated


def test_property_comparison_no_prediction_is_measurement_only():
    pc = PropertyComparison.build("CO2_uptake", measured=88.0)
    assert pc.predicted is None
    assert pc.within_prediction is False
    assert pc.deviation_sigma is None
