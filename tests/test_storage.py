import pytest

from materials_synthesis_agent.schema import (
    BOSuggestion,
    Decision,
    Experiment,
    FieldValue,
    Metric,
    ProtocolCandidate,
    ProtocolSource,
    Target,
)
from materials_synthesis_agent.storage import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


@pytest.fixture
def target(store):
    t = Target(
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="crystallinity",
        metric_measurement_method="PXRD",
    )
    store.save_target(t)
    return t


def test_target_round_trip(store, target):
    loaded = store.get_target(target.id)
    assert loaded == target


def test_protocol_candidate_round_trip(store, target):
    candidate = ProtocolCandidate(
        target_id=target.id,
        source=ProtocolSource.MANUAL,
        solvent=FieldValue(value="dioxane", inferred=True),
    )
    store.save_protocol_candidate(candidate)
    loaded = store.list_protocol_candidates(target.id)
    assert len(loaded) == 1
    assert loaded[0].id == candidate.id
    assert loaded[0].solvent.value == "dioxane"


def test_experiments_joined_through_protocol_candidates(store, target):
    candidate = ProtocolCandidate(target_id=target.id, source=ProtocolSource.MANUAL)
    store.save_protocol_candidate(candidate)
    exp = Experiment(
        project_target_id=target.id,
        protocol_candidate_id=candidate.id,
        metrics=[Metric(name="crystallinity", value=0.7, uncertainty=0.05, measurement_method="PXRD")],
    )
    store.save_experiment(exp)
    loaded = store.list_experiments(target.id)
    assert len(loaded) == 1
    assert loaded[0].metrics[0].value == 0.7


def test_experiments_empty_when_no_candidates(store, target):
    assert store.list_experiments(target.id) == []


def test_decision_trajectory_walks_followed_from_chain(store):
    d1 = Decision(
        context="first", chosen_protocol_candidate_id="c1", rationale="r1", expected_outcome="e1"
    )
    store.save_decision(d1)
    d2 = Decision(
        context="second", chosen_protocol_candidate_id="c2", rationale="r2", expected_outcome="e2",
        followed_from=d1.id,
    )
    store.save_decision(d2)
    trajectory = store.reconstruct_trajectory()
    assert [d.context for d in trajectory] == ["first", "second"]


def test_latest_decision(store):
    assert store.latest_decision() is None
    d1 = Decision(context="a", chosen_protocol_candidate_id="c", rationale="r", expected_outcome="e")
    store.save_decision(d1)
    assert store.latest_decision().id == d1.id


def test_usage_idempotency_key_prevents_double_charge(store):
    first = store.record_usage("llm_call", 0.05, idempotency_key="job-1")
    second = store.record_usage("llm_call", 0.05, idempotency_key="job-1")
    assert first is True
    assert second is False  # same key -- retried job must not double-charge
    assert store.total_usage_cost() == pytest.approx(0.05)


def test_bo_suggestion_round_trip(store, target):
    candidate = ProtocolCandidate(target_id=target.id, source=ProtocolSource.BO_SUGGESTED)
    store.save_protocol_candidate(candidate)
    suggestion = BOSuggestion(
        target_id=target.id,
        protocol_candidate_id=candidate.id,
        expected_improvement=0.1,
        uncertainty=0.02,
        rationale="test",
    )
    store.save_bo_suggestion(suggestion)
    loaded = store.list_bo_suggestions(target.id)
    assert len(loaded) == 1
    assert loaded[0].expected_improvement == pytest.approx(0.1)
