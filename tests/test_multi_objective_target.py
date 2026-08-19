"""Tests for the multi-objective Target schema and the end-to-end multi-objective suggest flow
(experiments with several metrics -> MultiObjectiveOptimizer -> Pareto set), exercised through the
storage + optimize layers the way the CLI/web do it."""

import torch

from materials_synthesis_agent.optimize import MultiObjectiveOptimizer, MultiObservation, Objective, ParameterSpace, ParameterSpec
from materials_synthesis_agent.schema import ObjectiveDirection, Target, TargetObjective


def _multi_target():
    return Target(
        functional_groups=["imine"],
        linkage_chemistry="imine condensation",
        application="CO2 capture",
        metric_name="yield",
        metric_measurement_method="isolated mass",
        objectives=[
            TargetObjective(name="yield", measurement_method="isolated mass", direction=ObjectiveDirection.MAXIMIZE),
            TargetObjective(name="cost", measurement_method="USD/g", direction=ObjectiveDirection.MINIMIZE),
        ],
    )


def test_single_objective_target_is_not_multi():
    t = Target(functional_groups=["x"], linkage_chemistry="y", application="z", metric_name="crystallinity", metric_measurement_method="PXRD")
    assert t.is_multi_objective is False
    assert [o.name for o in t.all_objectives] == ["crystallinity"]


def test_multi_objective_target_reports_all_objectives_with_directions():
    t = _multi_target()
    assert t.is_multi_objective is True
    assert [(o.name, o.maximize) for o in t.all_objectives] == [("yield", True), ("cost", False)]


def test_multi_objective_target_round_trips_through_json():
    restored = Target.model_validate_json(_multi_target().model_dump_json())
    assert restored.is_multi_objective is True
    assert [o.name for o in restored.all_objectives] == ["yield", "cost"]


def test_end_to_end_pareto_set_reflects_a_real_tradeoff():
    torch.manual_seed(5)
    space = ParameterSpace([ParameterSpec("temp", "continuous", bounds=(20, 150)),
                            ParameterSpec("solvent", "categorical", categories=("dioxane", "DMAc"))])
    t = _multi_target()

    # Ground truth with a genuine yield/cost conflict: hotter -> higher yield AND higher cost.
    def yield_fn(p):
        return (p["temp"] - 20) / 130
    def cost_fn(p):
        return 5 + (p["temp"] - 20) / 130 * 15

    seeds = [
        {"temp": 40, "solvent": "dioxane"}, {"temp": 90, "solvent": "DMAc"},
        {"temp": 120, "solvent": "dioxane"}, {"temp": 60, "solvent": "DMAc"},
    ]
    obs = [
        MultiObservation(params=p, values={"yield": yield_fn(p), "cost": cost_fn(p)},
                         uncertainties={"yield": 0.02, "cost": 0.3})
        for p in seeds
    ]
    optimizer = MultiObjectiveOptimizer(space, [Objective(name=o.name, maximize=o.maximize) for o in t.all_objectives])
    pareto = optimizer.suggest_next(obs, n_suggestions=4)

    assert len(pareto) >= 2  # a real spread of tradeoffs, not one point
    # Higher predicted yield must come with higher predicted cost across the returned set (the
    # conflict the ground truth encodes) -- i.e. the set is a genuine tradeoff front.
    ordered = sorted(pareto, key=lambda s: s.predicted_values["yield"])
    costs_in_yield_order = [s.predicted_values["cost"] for s in ordered]
    assert costs_in_yield_order == sorted(costs_in_yield_order), (
        "cost should rise with yield across the Pareto set given the ground-truth conflict"
    )
