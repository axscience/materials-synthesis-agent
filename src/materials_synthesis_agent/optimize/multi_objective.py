"""Multi-objective (Pareto) Bayesian optimization -- v0.2, ROADMAP.md.

Returns a set of Pareto-optimal suggestions, never a single scalarized number: the user picks the
tradeoff (CLAUDE.md convention, mirrored from the platform-level guardrail against silently
collapsing yield/crystallinity/cost into one ranked score).

Uses one independent MixedSingleTaskGP per objective (ModelListGP) and
qLogNoisyExpectedHypervolumeImprovement -- the standard BoTorch pattern for small-n multi-objective
optimization, more robust than a joint multi-task model at the sample sizes a wet-lab loop
actually produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed
from botorch.utils.multi_objective import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

from materials_synthesis_agent.optimize.single_objective import _DEFAULT_NOISE_STD_FOR_UNQUANTIFIED
from materials_synthesis_agent.optimize.space import ParameterSpace, ParamValue


@dataclass
class Objective:
    name: str
    maximize: bool = True


@dataclass
class MultiObservation:
    params: dict[str, ParamValue]
    values: dict[str, float]  # objective name -> value
    uncertainties: dict[str, Optional[float]] = None  # objective name -> uncertainty or None

    def __post_init__(self):
        if self.uncertainties is None:
            self.uncertainties = {}


@dataclass
class ParetoSuggestion:
    params: dict[str, ParamValue]
    predicted_values: dict[str, float]
    rationale: str


class MultiObjectiveOptimizer:
    def __init__(self, space: ParameterSpace, objectives: list[Objective]):
        if len(objectives) < 2:
            raise ValueError("MultiObjectiveOptimizer needs at least 2 objectives -- use SingleObjectiveOptimizer otherwise.")
        self.space = space
        self.objectives = objectives

    def _sign(self, objective: Objective) -> float:
        return 1.0 if objective.maximize else -1.0

    def _build_models(self, observations: list[MultiObservation]) -> ModelListGP:
        train_x = self.space.encode_batch([o.params for o in observations])
        models = []
        for objective in self.objectives:
            y = torch.tensor(
                [[self._sign(objective) * o.values[objective.name]] for o in observations], dtype=torch.double
            )
            yvar = torch.tensor(
                [[(o.uncertainties.get(objective.name) or _DEFAULT_NOISE_STD_FOR_UNQUANTIFIED) ** 2]
                 for o in observations],
                dtype=torch.double,
            )
            model = MixedSingleTaskGP(
                train_X=train_x,
                train_Y=y,
                train_Yvar=yvar,
                cat_dims=self.space.categorical_feature_indices,
                # See single_objective.py's _build_model for why this is required, not optional --
                # without it the fitted lengthscale collapses and the posterior reverts to the
                # training mean a fraction of a percent away from any training point.
                input_transform=Normalize(d=self.space.dim, bounds=self.space.bounds),
            )
            models.append(model)
        model_list = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        fit_gpytorch_mll(mll)
        return model_list

    def suggest_next(
        self,
        observations: list[MultiObservation],
        n_suggestions: int = 3,
        ref_point: Optional[dict[str, float]] = None,
        num_restarts: int = 8,
        raw_samples: int = 96,
    ) -> list[ParetoSuggestion]:
        if len(observations) < 3:
            raise ValueError("Need at least 3 observations to fit a multi-objective GP with meaningful uncertainty.")

        train_x = self.space.encode_batch([o.params for o in observations])
        model_list = self._build_models(observations)

        # All objectives internally standardized to "maximize" (sign-flipped in _build_models),
        # so the reference point must be flipped the same way. Default: worst observed value per
        # objective minus a small margin -- a documented heuristic, not a principled bound; pass
        # ref_point explicitly when you have a real one (e.g. "0 yield" is a natural true minimum).
        ref_point_signed = []
        for objective in self.objectives:
            if ref_point and objective.name in ref_point:
                ref_point_signed.append(self._sign(objective) * ref_point[objective.name])
            else:
                worst = min(self._sign(objective) * o.values[objective.name] for o in observations)
                ref_point_signed.append(worst - abs(worst) * 0.1 - 1e-3)

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model_list,
            ref_point=ref_point_signed,
            X_baseline=train_x,
            prune_baseline=True,
        )

        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=self.space.bounds,
            q=n_suggestions,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features_list=self.space.fixed_features_list(),
        )

        with torch.no_grad():
            posterior = model_list.posterior(candidates)
            means = posterior.mean  # n_suggestions x n_objectives, sign-flipped

        # Keep only the non-dominated subset of what was proposed -- optimize_acqf_mixed's batch
        # can include a dominated point if the joint hypervolume objective still favors it as part
        # of the batch; the *returned* suggestions should be a clean Pareto set.
        pareto_mask = is_non_dominated(means)

        suggestions = []
        for i in range(candidates.shape[0]):
            if not bool(pareto_mask[i]):
                continue
            params = self.space.decode(candidates[i])
            predicted = {
                obj.name: self._sign(obj) * means[i, j].item()  # un-flip back to real units
                for j, obj in enumerate(self.objectives)
            }
            rationale = (
                f"Pareto-optimal candidate from a {len(observations)}-observation multi-objective GP "
                f"over {[o.name for o in self.objectives]}; you choose the tradeoff among the returned set."
            )
            suggestions.append(ParetoSuggestion(params=params, predicted_values=predicted, rationale=rationale))
        return suggestions
