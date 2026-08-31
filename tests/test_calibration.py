"""Calibration diagnostics for the BO surrogates -- the "no number without uncertainty" guardrail
made enforceable. Tests the leave-one-out calibration report, the overconfidence-detection path
(the failure mode the gate exists to catch), the variance-scale correction, and the blocking gate.

Per TESTING.md this is exactly the highest-cost-of-bug area (scoring/statistics), so the sharp
assertions are on DETECTION -- a surrogate that under-reports uncertainty must be caught, because
the whole product value is telling a chemist a suggestion is trustworthy.
"""

import math
import random

import pytest
import torch

from materials_synthesis_agent.optimize.multi_objective import (
    MultiObjectiveOptimizer,
    MultiObservation,
    Objective,
)
from materials_synthesis_agent.optimize.single_objective import (
    CalibrationError,
    Observation,
    SingleObjectiveOptimizer,
)
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec

# A large-magnitude response (hundreds), the case where output standardization actually matters:
# a raw-scale train_Y here would mis-scale BoTorch's unit-variance hyperparameter priors.
SPACE = ParameterSpace(
    [
        ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
        ParameterSpec("time_hours", "continuous", bounds=(1, 96)),
        ParameterSpec("solvent", "categorical", categories=("dioxane", "mesitylene")),
    ]
)


def _true_surface(params: dict) -> float:
    # BET-area-like response in the hundreds, smooth quadratic with a categorical bonus.
    x, t, s = params["temperature_c"], params["time_hours"], params["solvent"]
    base = 800.0 - (x - 90) ** 2 * 0.03 - (t - 40) ** 2 * 0.02
    return base + (40.0 if s == "dioxane" else 0.0)


def _grid_observations(n: int, noise_std: float, seed: int, stated_uncertainty):
    """n observations on a pseudo-random grid, y = true + Gaussian(noise_std). `stated_uncertainty`
    is what we TELL the GP each point's uncertainty is -- set it equal to noise_std for an honest
    model, or far smaller to simulate an overconfident one."""
    rng = random.Random(seed)
    solvents = ("dioxane", "mesitylene")
    obs = []
    for _ in range(n):
        p = {
            "temperature_c": rng.uniform(30, 140),
            "time_hours": rng.uniform(4, 90),
            "solvent": rng.choice(solvents),
        }
        y = _true_surface(p) + rng.gauss(0.0, noise_std)
        obs.append(Observation(params=p, value=y, uncertainty=stated_uncertainty))
    return obs


def test_standardize_keeps_posterior_in_original_units_for_large_responses():
    """Standardize(m=1) must un-transform the posterior back to real units: a posterior mean near a
    training point of an ~800-magnitude response must land near ~800, not near 0 (standardized) or
    wildly off. Guards against passing the outcome transform but reading the transformed scale."""
    torch.manual_seed(0)
    obs = _grid_observations(n=8, noise_std=2.0, seed=1, stated_uncertainty=2.0)
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    model, _, _ = optimizer._build_model(obs)
    x = SPACE.encode(obs[0].params).unsqueeze(0)
    with torch.no_grad():
        mu = model.posterior(x).mean.item()
    # Within a few noise-widths of the actual measured value, in ORIGINAL units.
    assert abs(mu - obs[0].value) < 20.0, f"posterior {mu:.1f} not in original units near {obs[0].value:.1f}"


def test_calibration_insufficient_data_does_not_assert_or_block():
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    obs = _grid_observations(n=3, noise_std=2.0, seed=2, stated_uncertainty=2.0)
    report = optimizer.calibration_report(obs)
    assert report.sufficient is False
    assert report.passes is None
    # The gate must NOT block early experiments just because calibration can't be assessed yet.
    returned = optimizer.check_calibration_or_raise(obs)  # must not raise
    assert returned.passes is None


def test_calibration_detects_overconfidence_and_gate_blocks():
    """The core test: real scatter is large (noise_std=25) but the model is TOLD the uncertainty is
    tiny (1e-2). The posterior intervals will be far too tight -> standardized residuals blow up,
    80% coverage collapses, passes is False, and the gate raises. This is the 13%-coverage failure
    mode from the sibling repo, caught here as a blocking condition."""
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    obs = _grid_observations(n=14, noise_std=25.0, seed=3, stated_uncertainty=1e-2)

    report = optimizer.calibration_report(obs)
    assert report.sufficient is True
    assert report.passes is False, report.summary()
    assert report.z_std > 1.3, f"expected overconfident z_std>1.3, got {report.z_std:.2f}"
    assert report.coverage_80 < 0.8, f"expected under-coverage, got {report.coverage_80:.0%}"
    assert report.variance_scale > 1.0  # recommends inflating the variance

    with pytest.raises(CalibrationError):
        optimizer.check_calibration_or_raise(obs)


def test_well_specified_model_passes_the_gate():
    """A model whose stated uncertainty matches the true noise must PASS -- the gate must not
    false-positive and block a genuinely calibrated surrogate. (seed 0 empirically lands
    z_std~1.1, coverage~81%.)"""
    torch.manual_seed(0)
    obs = _grid_observations(n=16, noise_std=12.0, seed=0, stated_uncertainty=12.0)
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    report = optimizer.calibration_report(obs)
    assert report.sufficient is True
    assert report.passes is True, report.summary()
    optimizer.check_calibration_or_raise(obs)  # must not raise


def test_conservative_underconfidence_is_not_blocked():
    """Over-coverage (z_std < 1, intervals wider than needed) is safe and must NOT fail the gate --
    blocking it would halt the loop on noise. (seed 2 empirically lands z_std~0.75, coverage~94%.)"""
    torch.manual_seed(2)
    obs = _grid_observations(n=16, noise_std=12.0, seed=2, stated_uncertainty=12.0)
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    report = optimizer.calibration_report(obs)
    assert report.sufficient is True
    assert report.z_std < 1.0  # underconfident / over-covering
    assert report.passes is True, report.summary()  # safe direction is not blocked


def test_variance_scale_inflates_reported_uncertainty_deterministically():
    """variance_scale must scale the reported predicted_uncertainty by exactly sqrt(scale), without
    changing the chosen candidate (the acquisition uses the raw model, so the suggestion is
    identical -- only the reported uncertainty is recalibrated)."""
    obs = _grid_observations(n=8, noise_std=10.0, seed=4, stated_uncertainty=10.0)

    torch.manual_seed(7)
    base = SingleObjectiveOptimizer(SPACE, maximize=True, variance_scale=1.0)
    s_base = base.suggest_next(obs)

    torch.manual_seed(7)
    scaled = SingleObjectiveOptimizer(SPACE, maximize=True, variance_scale=4.0)
    s_scaled = scaled.suggest_next(obs)

    assert s_base.params == s_scaled.params  # same suggestion
    assert s_scaled.predicted_uncertainty == pytest.approx(
        s_base.predicted_uncertainty * math.sqrt(4.0), rel=1e-6
    )


def test_variance_scale_rejects_nonpositive():
    with pytest.raises(ValueError):
        SingleObjectiveOptimizer(SPACE, maximize=True, variance_scale=0.0)


def test_conformal_multiplier_present_and_ordered():
    """The distribution-free correction is reported alongside the parametric one, and the 90%
    multiplier is at least the 80% one (wider interval for higher coverage)."""
    optimizer = SingleObjectiveOptimizer(SPACE, maximize=True)
    obs = _grid_observations(n=12, noise_std=15.0, seed=5, stated_uncertainty=1e-2)
    report = optimizer.calibration_report(obs)
    assert report.conformal_multiplier_80 is not None
    assert report.conformal_multiplier_90 >= report.conformal_multiplier_80


def test_multi_objective_calibration_is_per_objective():
    space = ParameterSpace(
        [
            ParameterSpec("temp", "continuous", bounds=(0, 100)),
            ParameterSpec("solvent", "categorical", categories=("A", "B")),
        ]
    )

    def o1(p):
        return p["temp"] / 100 + (0.2 if p["solvent"] == "A" else 0.0)

    def o2(p):
        return (100 - p["temp"]) / 100 + (0.2 if p["solvent"] == "B" else 0.0)

    rng = random.Random(6)
    obs = []
    for _ in range(8):
        p = {"temp": rng.uniform(0, 100), "solvent": rng.choice(("A", "B"))}
        # Tell the model the noise is tiny while adding real scatter -> both objectives overconfident.
        obs.append(
            MultiObservation(
                p,
                {"yield_frac": o1(p) + rng.gauss(0, 0.15), "crystallinity": o2(p) + rng.gauss(0, 0.15)},
                {"yield_frac": 1e-3, "crystallinity": 1e-3},
            )
        )

    objectives = [Objective("yield_frac", maximize=True), Objective("crystallinity", maximize=True)]
    optimizer = MultiObjectiveOptimizer(space, objectives)

    reports = optimizer.calibration_report(obs)
    assert set(reports.keys()) == {"yield_frac", "crystallinity"}
    for r in reports.values():
        assert r.sufficient is True

    # Overconfident on both axes -> the gate raises for the first failing objective.
    with pytest.raises(CalibrationError):
        optimizer.check_calibration_or_raise(obs)
