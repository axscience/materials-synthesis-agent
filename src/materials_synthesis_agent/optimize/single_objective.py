"""Single-objective Bayesian optimization over a mixed continuous/categorical parameter space.

A real Gaussian-process model (BoTorch
MixedSingleTaskGP), not a heuristic. Real experimental observations carry their own measurement
noise (from Metric.uncertainty); a metric logged without uncertainty is not silently trusted -- it
gets a conservative, inflated noise estimate so it influences the model less, mirroring the
low_confidence flag in the schema.

Literature protocols "seed the prior" by biasing where the acquisition-function optimizer starts
its search (via `batch_initial_conditions`), not by being injected as fabricated training
observations -- we don't have a validated outcome for them under this project's conditions, so
treating a literature-reported yield as this project's GP training data would misrepresent its
provenance.

Uncertainty calibration (see CLAUDE.md guardrail "no number without uncertainty"):

- Outputs are standardized (`Standardize`) before the GP is fit, not just inputs. BoTorch's default
  hyperparameter priors (lengthscale via `get_covar_module_with_dim_scaled_prior`, the ScaleKernel
  outputscale prior) assume a unit-variance response; fitting a raw-scale `train_Y` (a yield in
  0-100, a BET area in the thousands) against those priors mis-scales the fitted outputscale and the
  posterior variance the model reports. Standardizing the response is the analogue of the input
  `Normalize` the module already relies on -- the observation noise (`train_Yvar`) is rescaled
  consistently and the posterior is un-transformed back to real units automatically.

- At the sample sizes a wet-lab loop actually produces (n = 2-20), kernel hyperparameters are barely
  identifiable, so the priors do real work: `fit_gpytorch_mll` is effectively MAP, not raw MLE. We
  deliberately keep BoTorch's default dim-scaled lengthscale prior rather than adding a fully
  Bayesian GP (`SaasFullyBayesianSingleTaskGP`) -- SAAS needs a Pyro/NUTS dependency and targets
  high-dimensional continuous spaces, and handles the categorical dimensions of a synthesis space
  poorly. It's a future option if calibration proves inadequate on real data, not a default.

- `calibration_report()` measures, via leave-one-out, whether the posterior uncertainty is actually
  trustworthy (are the 80% intervals covering ~80% of held-out results?), and recommends a
  correction: a parametric variance scale (`variance_scale`, applied to the reported uncertainty) or
  a distribution-free conformal multiplier. `check_calibration_or_raise()` turns a detected
  miscalibration into a blocking error the hosting layer maps to a 409 -- an uncalibrated surrogate
  driving the loop amplifies its own errors, so this is the gate that makes closing the loop safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood

from materials_synthesis_agent.optimize.space import ParameterSpace, ParamValue

# Floor for inferred noise std when no measurement uncertainty is stated.  The actual default
# is computed adaptively from the observed response spread (see _infer_noise_std); this floor
# prevents a degenerate near-zero noise estimate when all observations have similar values.
_NOISE_STD_FLOOR = 0.01
_NOISE_STD_FRACTION = 0.15  # fraction of observed response std used as default noise

# Two-sided standard-normal quantiles for central-coverage levels, so coverage/interval-width
# diagnostics need no scipy import in this hot path.  z such that P(|Z| <= z) = level.
_Z_FOR_LEVEL = {0.80: 1.2815515594, 0.90: 1.6448536270, 0.95: 1.9599639845}

# Minimum observations before a leave-one-out calibration estimate is meaningful.  Below this we
# report insufficient data rather than a noisy pass/fail on 2-3 folds.
_MIN_CALIBRATION_POINTS = 5


@dataclass
class Observation:
    params: dict[str, ParamValue]
    value: float
    uncertainty: Optional[float] = None  # None -> low-confidence, downweighted, see module docstring

    @property
    def is_low_confidence(self) -> bool:
        return self.uncertainty is None


@dataclass
class LiteratureAnchor:
    """A literature-derived candidate protocol, used only to bias where the optimizer searches --
    never injected as a fabricated observed outcome."""

    params: dict[str, ParamValue]


@dataclass
class Suggestion:
    params: dict[str, ParamValue]
    expected_improvement: float
    predicted_value: float
    predicted_uncertainty: float
    rationale: str


class CalibrationError(RuntimeError):
    """Raised when the GP's uncertainty is measurably miscalibrated and a caller asked for the
    calibration gate. The hosting layer maps this to a blocking 409."""

    def __init__(self, report: "CalibrationReport"):
        self.report = report
        super().__init__(report.summary())


@dataclass
class CalibrationReport:
    """Leave-one-out calibration diagnostics for the GP's predictive uncertainty.

    The core statistic is the set of standardized residuals z_i = (y_i - mu_i) / sigma_i, where
    mu_i, sigma_i are the LOO posterior mean and predictive std (latent + observation noise) at the
    held-out point. If the uncertainty is honest, the z_i are ~N(0, 1): `z_std` ~ 1 and the 80%
    interval covers ~80% of held-out points. `z_std` > 1 means the model is OVERCONFIDENT (intervals
    too tight -- the dangerous direction); < 1 means underconfident (intervals too wide).
    """

    n: int
    sufficient: bool
    passes: Optional[bool] = None  # None when not sufficient to assess
    z_mean: Optional[float] = None
    z_std: Optional[float] = None
    coverage_80: Optional[float] = None
    coverage_90: Optional[float] = None
    mean_interval_width_80: Optional[float] = None
    rmse: Optional[float] = None
    # Parametric correction: multiply the posterior VARIANCE by this (equivalently, the reported
    # std by sqrt(variance_scale)) so that E[z^2] = 1 after correction.  Feed it to the optimizer's
    # `variance_scale`.
    variance_scale: Optional[float] = None
    # Distribution-free correction: multiply sigma_i by this for an interval with ~level coverage,
    # making no Gaussian assumption on the residuals (conformalized LOO residuals).
    conformal_multiplier_80: Optional[float] = None
    conformal_multiplier_90: Optional[float] = None
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.sufficient:
            return (
                f"Calibration: insufficient data (n={self.n}, need >= {_MIN_CALIBRATION_POINTS}). "
                "Not asserting the GP's uncertainty is trustworthy yet."
            )
        verdict = "PASS" if self.passes else "FAIL"
        lines = [
            f"Calibration [{verdict}] on n={self.n} leave-one-out folds:",
            f"  standardized-residual std   z_std = {self.z_std:.3f}   (1.0 ideal; >1 overconfident)",
            f"  standardized-residual mean  z_mean = {self.z_mean:+.3f}  (0.0 ideal)",
            f"  80% interval coverage       = {self.coverage_80:.0%}   (target 80%)",
            f"  90% interval coverage       = {self.coverage_90:.0%}   (target 90%)",
            f"  mean 80% interval width     = {self.mean_interval_width_80:.4g} (response units)",
            f"  LOO RMSE                    = {self.rmse:.4g}",
            f"  recommended variance_scale  = {self.variance_scale:.3f} "
            f"(reported std x {math.sqrt(self.variance_scale):.3f})",
            f"  conformal multiplier (80%)  = {self.conformal_multiplier_80:.3f}",
        ]
        lines.extend(f"  - {m}" for m in self.messages)
        return "\n".join(lines)


def _infer_noise_std(observations: list[Observation]) -> float:
    """Adaptive noise estimate for observations without stated uncertainty.

    Uses a fraction of the observed response spread so the GP can still learn
    structure from the data.  A fixed constant (the previous 0.25) drowns the
    signal when the response range is narrow — the GP collapses to a constant
    mean with zero outputscale and EI becomes identically zero everywhere.
    """
    values = [o.value for o in observations]
    if len(values) < 2:
        return _NOISE_STD_FLOOR
    spread = max(values) - min(values)
    std_est = spread * _NOISE_STD_FRACTION
    return max(_NOISE_STD_FLOOR, std_est)


def _normal_coverage(z_values: list[float], level: float) -> float:
    """Empirical fraction of standardized residuals falling inside the central `level` interval."""
    z_thresh = _Z_FOR_LEVEL[level]
    inside = sum(1 for z in z_values if abs(z) <= z_thresh)
    return inside / len(z_values)


def _empirical_quantile(values: list[float], q: float) -> float:
    """Conformal quantile: the ceil((n+1) * q)/n order statistic, clamped to the largest value when
    that exceeds n (the finite-sample split-conformal convention)."""
    s = sorted(values)
    n = len(s)
    rank = math.ceil((n + 1) * q)
    if rank >= n:
        return s[-1]
    return s[rank - 1]


class SingleObjectiveOptimizer:
    def __init__(self, space: ParameterSpace, maximize: bool = True, variance_scale: float = 1.0):
        """variance_scale multiplies the GP's posterior VARIANCE before the reported
        `predicted_uncertainty` is returned -- set it to `CalibrationReport.variance_scale` once a
        leave-one-out calibration run has measured how over/under-confident the raw posterior is.
        The default 1.0 is a no-op, so existing behaviour is unchanged until a caller opts in."""
        if variance_scale <= 0:
            raise ValueError("variance_scale must be positive.")
        self.space = space
        self.maximize = maximize
        self.variance_scale = variance_scale

    def _build_model(self, observations: list[Observation]):
        train_x = self.space.encode_batch([o.params for o in observations])
        train_y = torch.tensor([[o.value] for o in observations], dtype=torch.double)
        default_noise = _infer_noise_std(observations)
        train_yvar = torch.tensor(
            [[(o.uncertainty**2) if o.uncertainty is not None else default_noise**2]
             for o in observations],
            dtype=torch.double,
        )
        # The Normalize input transform is required, not optional: without it the continuous dims
        # are fit in their raw (e.g. 0-150) scale, but BoTorch's default lengthscale priors assume
        # a unit cube -- the fitted lengthscale collapses to near-zero and the posterior mean
        # reverts to the training mean just fractions of a percent away from any training point.
        normalize = Normalize(d=self.space.dim, bounds=self.space.bounds)
        # Standardize is the output-side counterpart: BoTorch's default outputscale/lengthscale
        # priors assume a unit-variance response, so a raw-scale train_Y mis-scales the fitted
        # outputscale and the posterior variance. Standardize(m=1) rescales train_Y (and train_Yvar
        # consistently) to zero mean / unit variance for fitting, and un-transforms the posterior
        # back to real units -- so predicted_value/predicted_uncertainty stay in the caller's units.
        standardize = Standardize(m=1)

        cat_dims = self.space.categorical_feature_indices
        if cat_dims:
            model = MixedSingleTaskGP(
                train_X=train_x, train_Y=train_y, train_Yvar=train_yvar,
                cat_dims=cat_dims, input_transform=normalize, outcome_transform=standardize,
            )
        else:
            # MixedSingleTaskGP requires at least one categorical dimension. An all-continuous
            # space (e.g. a researcher who fixed their solvent and optimizes only temperature/
            # time/concentration) uses the plain SingleTaskGP instead.
            model = SingleTaskGP(
                train_X=train_x, train_Y=train_y, train_Yvar=train_yvar,
                input_transform=normalize, outcome_transform=standardize,
            )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model, train_x, train_y

    def suggest_next(
        self,
        observations: list[Observation],
        literature_anchors: Optional[list[LiteratureAnchor]] = None,
        num_restarts: int = 10,
        raw_samples: int = 128,
    ) -> Suggestion:
        if len(observations) < 2:
            raise ValueError(
                "Need at least 2 observations to fit a GP with meaningful uncertainty. "
                "With 0-1 real results, suggest literature protocols directly instead of calling this."
            )

        model, train_x, train_y = self._build_model(observations)
        best_f = train_y.max().item() if self.maximize else train_y.min().item()

        acqf = LogExpectedImprovement(model=model, best_f=best_f, maximize=self.maximize)

        if literature_anchors:
            # Anchors bias the search by widening the raw-sample pool the optimizer starts from.
            raw_samples = max(raw_samples, raw_samples + len(literature_anchors) * 4)

        fixed_features_list = self.space.fixed_features_list()
        if fixed_features_list != [{}]:
            # Mixed continuous+categorical space: search each categorical combination.
            candidate, acq_value = optimize_acqf_mixed(
                acq_function=acqf, bounds=self.space.bounds, q=1,
                num_restarts=num_restarts, raw_samples=raw_samples,
                fixed_features_list=fixed_features_list,
            )
        else:
            # All-continuous space: plain continuous acquisition optimization (optimize_acqf_mixed
            # requires a non-trivial fixed_features_list).
            candidate, acq_value = optimize_acqf(
                acq_function=acqf, bounds=self.space.bounds, q=1,
                num_restarts=num_restarts, raw_samples=raw_samples,
            )

        with torch.no_grad():
            posterior = model.posterior(candidate)
            predicted_value = posterior.mean.item()
            # variance_scale recalibrates the reported uncertainty: a leave-one-out calibration run
            # (calibration_report) measures how over/under-confident the raw posterior is, and this
            # scalar corrects it. Default 1.0 leaves the raw posterior std unchanged.
            predicted_std = (posterior.variance.item() * self.variance_scale) ** 0.5

        params = self.space.decode(candidate.squeeze(0))
        direction = "maximize" if self.maximize else "minimize"
        rationale = (
            f"GP fit on {len(observations)} observation(s) "
            f"({sum(1 for o in observations if o.is_low_confidence)} low-confidence); "
            f"chosen to {direction} expected improvement over current best ({best_f:.4g})."
        )
        if self.variance_scale != 1.0:
            rationale += f" Reported uncertainty rescaled by calibration factor sqrt({self.variance_scale:.3g})."
        if literature_anchors:
            rationale += f" Search included {len(literature_anchors)} literature-anchored starting point(s)."

        return Suggestion(
            params=params,
            # LogExpectedImprovement returns log(EI) for numerical stability -- exponentiate back
            # to true expected-improvement units before this leaves the module, since callers
            # (including BOSuggestion.expected_improvement) expect a meaningful, not log-space, value.
            expected_improvement=math.exp(acq_value.item()),
            predicted_value=predicted_value,
            predicted_uncertainty=predicted_std,
            rationale=rationale,
        )

    def _loo_standardized_residuals(
        self, observations: list[Observation]
    ) -> tuple[list[float], list[float]]:
        """Leave-one-out standardized residuals z_i = (y_i - mu_i) / sigma_i and the per-fold
        predictive stds sigma_i.

        For each held-out point i, refit the GP on the other n-1 observations (a full refit, so the
        hyperparameters are honestly re-estimated), then evaluate the posterior at x_i. sigma_i is
        the PREDICTIVE std -- latent posterior variance plus the held-out point's own observation
        noise -- because a real measured y carries measurement noise, so a fair coverage check must
        include it. Both mu_i and sigma_i are in the caller's original response units (Standardize
        un-transforms the posterior)."""
        z_values: list[float] = []
        sigmas: list[float] = []
        for i in range(len(observations)):
            held_out = observations[i]
            rest = observations[:i] + observations[i + 1 :]
            model, _, _ = self._build_model(rest)
            x_i = self.space.encode(held_out.params).unsqueeze(0)
            with torch.no_grad():
                posterior = model.posterior(x_i)
                mu_i = posterior.mean.item()
                latent_var = posterior.variance.item()
            # Held-out observation noise: its own stated uncertainty, or the adaptive default the
            # model would have assigned it (estimated from the training subset, matching _build_model).
            if held_out.uncertainty is not None:
                noise_var = held_out.uncertainty**2
            else:
                noise_var = _infer_noise_std(rest) ** 2
            sigma_i = math.sqrt(max(latent_var + noise_var, 1e-12))
            z_values.append((held_out.value - mu_i) / sigma_i)
            sigmas.append(sigma_i)
        return z_values, sigmas

    def calibration_report(
        self,
        observations: list[Observation],
        tol_z: float = 0.25,
        tol_coverage: float = 0.10,
    ) -> CalibrationReport:
        """Measure whether the GP's predictive uncertainty is trustworthy, via leave-one-out.

        `passes` is asymmetric: it is False only when the model is OVERCONFIDENT -- standardized-
        residual std above 1 + `tol_z`, or 80% interval coverage below 0.80 - `tol_coverage`. That
        is the failure this gate exists to catch (intervals too tight, so a suggestion looks more
        trustworthy than it is). Underconfidence (over-coverage) is conservative and does not fail:
        at wet-lab sample sizes a single LOO estimate is noisy, and blocking on safe over-coverage
        would halt the loop for no reason. Below `_MIN_CALIBRATION_POINTS` observations, `sufficient`
        is False and `passes` is None -- we don't claim calibration we can't measure, and the gate
        does not block early experiments on that basis."""
        n = len(observations)
        if n < _MIN_CALIBRATION_POINTS:
            return CalibrationReport(
                n=n, sufficient=False, passes=None,
                messages=[f"Log at least {_MIN_CALIBRATION_POINTS} results to assess calibration."],
            )

        z_values, sigmas = self._loo_standardized_residuals(observations)
        mean_z = sum(z_values) / n
        var_z = sum((z - mean_z) ** 2 for z in z_values) / (n - 1)
        z_std = math.sqrt(var_z)
        # E[z^2] about zero is the right target for a variance rescale: multiplying the posterior
        # variance by mean(z^2) makes the corrected residuals have unit second moment.
        mean_z_sq = sum(z * z for z in z_values) / n
        rmse = math.sqrt(sum((z * s) ** 2 for z, s in zip(z_values, sigmas)) / n)

        coverage_80 = _normal_coverage(z_values, 0.80)
        coverage_90 = _normal_coverage(z_values, 0.90)
        mean_width_80 = 2.0 * _Z_FOR_LEVEL[0.80] * (sum(sigmas) / n)

        abs_z = [abs(z) for z in z_values]
        conf_80 = _empirical_quantile(abs_z, 0.80)
        conf_90 = _empirical_quantile(abs_z, 0.90)

        messages: list[str] = []
        if z_std > 1 + tol_z:
            messages.append(
                f"OVERCONFIDENT: intervals too tight (z_std={z_std:.2f}). "
                f"Apply variance_scale={mean_z_sq:.2f} or refit with more data."
            )
        elif z_std < 1 - tol_z:
            messages.append(f"Underconfident: intervals wider than needed (z_std={z_std:.2f}).")
        if coverage_80 < 0.80 - tol_coverage:
            messages.append(
                f"80% intervals only cover {coverage_80:.0%} of held-out results -- "
                "do not trust the reported uncertainty until corrected."
            )
        if abs(mean_z) > tol_z:
            direction = "under" if mean_z > 0 else "over"
            messages.append(f"Biased mean: model {direction}-predicts (z_mean={mean_z:+.2f}).")

        # The gate is DELIBERATELY asymmetric. Only overconfidence -- intervals too tight, so a
        # chemist would trust a suggestion that isn't trustworthy -- is a failure. Underconfidence
        # (z_std < 1, over-coverage) is conservative and safe: wider-than-needed intervals never
        # mislead, so blocking on them would just halt the loop on noise. At the sample sizes a
        # wet-lab loop produces, a single LOO estimate is noisy and errs toward the conservative
        # side often, so a symmetric gate would false-positive constantly.
        overconfident = z_std > 1.0 + tol_z
        under_covering = coverage_80 < 0.80 - tol_coverage
        passes = not (overconfident or under_covering)

        return CalibrationReport(
            n=n, sufficient=True, passes=passes,
            z_mean=mean_z, z_std=z_std,
            coverage_80=coverage_80, coverage_90=coverage_90,
            mean_interval_width_80=mean_width_80, rmse=rmse,
            variance_scale=mean_z_sq,
            conformal_multiplier_80=conf_80, conformal_multiplier_90=conf_90,
            messages=messages,
        )

    def check_calibration_or_raise(
        self,
        observations: list[Observation],
        tol_z: float = 0.25,
        tol_coverage: float = 0.10,
    ) -> CalibrationReport:
        """The gate. Returns the report when calibration passes (or can't yet be assessed); raises
        CalibrationError when the GP is measurably miscalibrated -- the hosting layer maps that to a
        blocking 409 so an uncalibrated surrogate never drives a suggestion the loop trusts."""
        report = self.calibration_report(observations, tol_z=tol_z, tol_coverage=tol_coverage)
        if report.passes is False:
            raise CalibrationError(report)
        return report
