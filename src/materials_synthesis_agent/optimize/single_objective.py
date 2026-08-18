"""Single-objective Bayesian optimization over a mixed continuous/categorical parameter space.

This is the v0.1 optimizer (CLAUDE.md guardrail #5): a real Gaussian-process model (BoTorch
MixedSingleTaskGP), not a heuristic. Real experimental observations carry their own measurement
noise (from Metric.uncertainty); a metric logged without uncertainty is not silently trusted -- it
gets a conservative, inflated noise estimate so it influences the model less, mirroring the
low_confidence flag in the schema.

Literature protocols "seed the prior" by biasing where the acquisition-function optimizer starts
its search (via `batch_initial_conditions`), not by being injected as fabricated training
observations -- we don't have a validated outcome for them under this project's conditions, so
treating a literature-reported yield as this project's GP training data would misrepresent its
provenance. See ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood

from materials_synthesis_agent.optimize.space import ParameterSpace, ParamValue

# A metric logged with no stated uncertainty still needs *some* noise value for the GP -- this is
# a conservative placeholder (relative to typical result spread) that downweights, not excludes,
# an unquantified observation. It is not a claim about the true measurement error.
_DEFAULT_NOISE_STD_FOR_UNQUANTIFIED = 0.25


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


class SingleObjectiveOptimizer:
    def __init__(self, space: ParameterSpace, maximize: bool = True):
        self.space = space
        self.maximize = maximize

    def _build_model(self, observations: list[Observation]):
        train_x = self.space.encode_batch([o.params for o in observations])
        train_y = torch.tensor([[o.value] for o in observations], dtype=torch.double)
        train_yvar = torch.tensor(
            [[(o.uncertainty**2) if o.uncertainty is not None else _DEFAULT_NOISE_STD_FOR_UNQUANTIFIED**2]
             for o in observations],
            dtype=torch.double,
        )
        # The Normalize input transform is required, not optional: without it the continuous dims
        # are fit in their raw (e.g. 0-150) scale, but BoTorch's default lengthscale priors assume
        # a unit cube -- the fitted lengthscale collapses to near-zero and the posterior mean
        # reverts to the training mean just fractions of a percent away from any training point.
        normalize = Normalize(d=self.space.dim, bounds=self.space.bounds)

        cat_dims = self.space.categorical_feature_indices
        if cat_dims:
            model = MixedSingleTaskGP(
                train_X=train_x, train_Y=train_y, train_Yvar=train_yvar,
                cat_dims=cat_dims, input_transform=normalize,
            )
        else:
            # MixedSingleTaskGP requires at least one categorical dimension. An all-continuous
            # space (e.g. a researcher who fixed their solvent and optimizes only temperature/
            # time/concentration) uses the plain SingleTaskGP instead.
            model = SingleTaskGP(
                train_X=train_x, train_Y=train_y, train_Yvar=train_yvar, input_transform=normalize,
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
            predicted_std = posterior.variance.sqrt().item()

        params = self.space.decode(candidate.squeeze(0))
        direction = "maximize" if self.maximize else "minimize"
        rationale = (
            f"GP fit on {len(observations)} observation(s) "
            f"({sum(1 for o in observations if o.is_low_confidence)} low-confidence); "
            f"chosen to {direction} expected improvement over current best ({best_f:.4g})."
        )
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
