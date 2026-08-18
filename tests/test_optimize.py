"""Validates the BO engine against known synthetic functions with known optima -- CLAUDE.md
guardrail: "a validation test suite (not just a passing unit test) confirms the GP's
recommendations behave correctly on known synthetic functions before it's trusted." Also pins down
a real bug found during development (missing input normalization on the GP -- see
optimize/single_objective.py and optimize/multi_objective.py's Normalize input_transform, and the
comment explaining why it's required) so it can't silently regress.
"""

import torch

from materials_synthesis_agent.optimize.multi_objective import MultiObjectiveOptimizer, MultiObservation, Objective
from materials_synthesis_agent.optimize.single_objective import LiteratureAnchor, Observation, SingleObjectiveOptimizer
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec

SPACE = ParameterSpace(
    [
        ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
        ParameterSpec("time_hours", "continuous", bounds=(1, 96)),
        ParameterSpec("solvent", "categorical", categories=("dioxane", "mesitylene", "DMAc")),
    ]
)
SOLVENT_BONUS = {"dioxane": 0.5, "mesitylene": 0.0, "DMAc": -0.3}
TRUE_OPT = {"temperature_c": 75.0, "time_hours": 48.0, "solvent": "dioxane"}


def true_fn(params: dict) -> float:
    x, t, s = params["temperature_c"], params["time_hours"], params["solvent"]
    return -((x - 75) ** 2) / 3000 - ((t - 48) ** 2) / 2000 + SOLVENT_BONUS[s]


def test_gp_posterior_is_accurate_near_a_training_point():
    """Regression test for the input-normalization bug: without Normalize(bounds=space.bounds),
    the fitted lengthscale collapses and the posterior reverts to the training mean a fraction of
    a percent away from any training point. This must stay close to the true value nearby."""
    torch.manual_seed(0)
    seed_params = [
        {"temperature_c": 10, "time_hours": 12, "solvent": "DMAc"},
        {"temperature_c": 90, "time_hours": 80, "solvent": "mesitylene"},
        {"temperature_c": 50, "time_hours": 24, "solvent": "dioxane"},
        {"temperature_c": 20, "time_hours": 15, "solvent": "dioxane"},
    ]
    observations = [Observation(params=p, value=true_fn(p), uncertainty=0.02) for p in seed_params]
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    model, train_x, _ = optimizer._build_model(observations)

    near_point = SPACE.encode({"temperature_c": 20.5, "time_hours": 15.2, "solvent": "dioxane"}).unsqueeze(0)
    with torch.no_grad():
        predicted = model.posterior(near_point).mean.item()
    actual_at_training_point = true_fn(seed_params[3])

    assert abs(predicted - actual_at_training_point) < 0.05, (
        f"GP posterior {predicted:.4f} near a training point (true value {actual_at_training_point:.4f}) "
        "diverged too much -- likely the input-normalization regression this test guards against."
    )


def test_single_objective_converges_near_known_optimum():
    torch.manual_seed(1)
    seed_params = [
        {"temperature_c": 30, "time_hours": 12, "solvent": "DMAc"},
        {"temperature_c": 100, "time_hours": 80, "solvent": "mesitylene"},
        {"temperature_c": 60, "time_hours": 24, "solvent": "dioxane"},
        {"temperature_c": 140, "time_hours": 5, "solvent": "DMAc"},
    ]
    observations = [Observation(params=p, value=true_fn(p), uncertainty=0.05) for p in seed_params]
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    anchors = [LiteratureAnchor(params={"temperature_c": 80, "time_hours": 40, "solvent": "dioxane"})]

    for _ in range(8):
        suggestion = optimizer.suggest_next(observations, literature_anchors=anchors)
        actual = true_fn(suggestion.params)
        observations.append(Observation(params=suggestion.params, value=actual, uncertainty=0.05))

    best = max(observations, key=lambda o: o.value)
    assert best.params["solvent"] == "dioxane"  # correctly identifies the best categorical choice
    assert true_fn(TRUE_OPT) - best.value < 0.05  # within 10% of the true optimum's value range


def test_single_objective_requires_at_least_two_observations():
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    import pytest

    with pytest.raises(ValueError):
        optimizer.suggest_next([Observation(params=TRUE_OPT, value=0.5, uncertainty=0.05)])


# --- Regression: all-continuous parameter space (no categorical dimension) ---
# MixedSingleTaskGP requires >=1 categorical dim; a researcher who fixes their solvent and
# optimizes only temperature/time/concentration must not crash. See single_objective._build_model.

CONTINUOUS_SPACE = ParameterSpace(
    [
        ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
        ParameterSpec("time_hours", "continuous", bounds=(1, 96)),
    ]
)


def _continuous_max_fn(p):  # optimum at (85, 60)
    return -((p["temperature_c"] - 85) ** 2) / 4000 - ((p["time_hours"] - 60) ** 2) / 3000


def test_all_continuous_space_does_not_crash_and_converges_maximize():
    torch.manual_seed(3)
    seeds = [
        {"temperature_c": 40, "time_hours": 20},
        {"temperature_c": 120, "time_hours": 80},
        {"temperature_c": 70, "time_hours": 50},
    ]
    obs = [Observation(params=p, value=_continuous_max_fn(p), uncertainty=0.02) for p in seeds]
    optimizer = SingleObjectiveOptimizer(CONTINUOUS_SPACE, maximize=True)
    for _ in range(6):
        s = optimizer.suggest_next(obs)
        obs.append(Observation(params=s.params, value=_continuous_max_fn(s.params), uncertainty=0.02))
    best = max(obs, key=lambda o: o.value)
    assert best.value > -0.03  # close to the optimum value of 0


# --- Regression: minimize direction ---
# The optimizer must actually move toward LOWER values when maximize=False; many materials metrics
# (particle size, defect density, reaction time, cost) are minimize goals.


def _defect_fn(p):  # MINIMIZE: lowest (best) at temperature 30
    return abs(p["temperature_c"] - 30) / 100.0


def test_minimize_direction_moves_toward_lower_values():
    torch.manual_seed(4)
    space = ParameterSpace(
        [ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
         ParameterSpec("atmosphere", "categorical", categories=("N2", "Ar"))]
    )
    seeds = [
        {"temperature_c": 60, "atmosphere": "N2"},
        {"temperature_c": 100, "atmosphere": "Ar"},
        {"temperature_c": 140, "atmosphere": "N2"},
    ]
    obs = [Observation(params=p, value=_defect_fn(p), uncertainty=0.01) for p in seeds]
    optimizer = SingleObjectiveOptimizer(space, maximize=False)
    for _ in range(6):
        s = optimizer.suggest_next(obs)
        obs.append(Observation(params=s.params, value=_defect_fn(s.params), uncertainty=0.01))
    best = min(obs, key=lambda o: o.value)
    # Best found should be near the true minimum (temp 30), NOT chasing the high-temperature end.
    assert best.params["temperature_c"] < 55
    assert best.value < 0.25


def test_multi_objective_returns_nondominated_set_matching_predictions():
    torch.manual_seed(2)
    space = ParameterSpace(
        [ParameterSpec("temp", "continuous", bounds=(0, 100)), ParameterSpec("solvent", "categorical", categories=("A", "B"))]
    )

    def obj1(p):
        return p["temp"] / 100 + (0.2 if p["solvent"] == "A" else 0.0)

    def obj2(p):
        return (100 - p["temp"]) / 100 + (0.2 if p["solvent"] == "B" else 0.0)

    objectives = [Objective("yield_frac", maximize=True), Objective("crystallinity", maximize=True)]
    seed_params = [
        {"temp": 10, "solvent": "A"}, {"temp": 90, "solvent": "B"},
        {"temp": 50, "solvent": "A"}, {"temp": 50, "solvent": "B"},
        {"temp": 20, "solvent": "B"}, {"temp": 80, "solvent": "A"},
    ]
    observations = [
        MultiObservation(p, {"yield_frac": obj1(p), "crystallinity": obj2(p)}, {"yield_frac": 0.02, "crystallinity": 0.02})
        for p in seed_params
    ]
    optimizer = MultiObjectiveOptimizer(space, objectives)
    suggestions = optimizer.suggest_next(observations, n_suggestions=4)

    assert len(suggestions) >= 2  # a real spread, not collapsed to a single point
    for s in suggestions:
        actual = {"yield_frac": obj1(s.params), "crystallinity": obj2(s.params)}
        for name in ("yield_frac", "crystallinity"):
            assert abs(s.predicted_values[name] - actual[name]) < 0.15  # prediction tracks reality

    # no suggestion should be dominated by another in the same returned set
    for i, a in enumerate(suggestions):
        for j, b in enumerate(suggestions):
            if i == j:
                continue
            dominates = all(b.predicted_values[o.name] >= a.predicted_values[o.name] for o in objectives) and any(
                b.predicted_values[o.name] > a.predicted_values[o.name] for o in objectives
            )
            assert not dominates, f"suggestion {i} is dominated by {j} -- Pareto filtering failed"


def test_multi_objective_requires_at_least_two_objectives():
    import pytest

    space = ParameterSpace([ParameterSpec("x", "continuous", bounds=(0, 1))])
    with pytest.raises(ValueError):
        MultiObjectiveOptimizer(space, [Objective("only_one")])
