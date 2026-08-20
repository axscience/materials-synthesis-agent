"""Convert literature-extracted measured outcomes into GP-compatible Observations.

A ProtocolCandidate's `measured_outcomes` are real, published measurements — a BET surface area
reported in a paper alongside the synthesis conditions that produced it. When those conditions
can be mapped to a ParameterSpace, the measurement becomes a genuine Observation the GP can
train on, not just a LiteratureAnchor that biases the search.

The mapping is best-effort: a paper that reports temperature and solvent but not monomer
concentration leaves that dimension unknown. `protocol_to_observation` returns None when
too many required parameters are missing to produce a meaningful data point.
"""

from __future__ import annotations

import logging
from typing import Optional

from materials_synthesis_agent.optimize.space import ParameterSpace, ParamValue
from materials_synthesis_agent.optimize.single_objective import Observation
from materials_synthesis_agent.schema import MeasuredOutcome, ProtocolCandidate

logger = logging.getLogger(__name__)


def _extract_param_value(candidate: ProtocolCandidate, param_name: str) -> Optional[ParamValue]:
    """Try to read a named parameter from a ProtocolCandidate's fields. Returns None if the
    field is absent or its value can't be interpreted as the expected type."""
    field_map = {
        "temperature_c": candidate.temperature_c,
        "time_hours": candidate.time_hours,
        "concentration_molar": candidate.concentration_molar,
        "monomer_concentration_M": candidate.concentration_molar,
    }
    fv = field_map.get(param_name)
    if fv is not None:
        try:
            return float(fv.value.split()[0].rstrip("°CcMm%"))
        except (ValueError, IndexError):
            return None

    if param_name == "acoh_concentration_M" and candidate.modulator:
        val = candidate.modulator.value
        for token in val.replace(",", " ").split():
            try:
                num = float(token.rstrip("Mm"))
                if 0.1 <= num <= 20:
                    return num
            except ValueError:
                continue
        return None

    for field_name in ["solvent", "catalyst", "modulator", "synthesis_method",
                       "atmosphere", "activation_method"]:
        fv = getattr(candidate, field_name, None)
        if fv is not None and param_name.startswith(field_name):
            return fv.value

    return None


def protocol_to_params(
    candidate: ProtocolCandidate,
    space: ParameterSpace,
) -> Optional[dict[str, ParamValue]]:
    """Try to map a ProtocolCandidate's extracted fields to a ParameterSpace's parameters.
    Returns None if more than half the parameters can't be resolved — the data point would
    carry too much imputed noise to be useful."""
    params: dict[str, ParamValue] = {}
    missing = []

    for spec in space.specs:
        value = _extract_param_value(candidate, spec.name)

        if value is None:
            missing.append(spec.name)
            continue

        if spec.kind == "continuous":
            try:
                num = float(value)
            except (ValueError, TypeError):
                missing.append(spec.name)
                continue
            lo, hi = spec.bounds
            params[spec.name] = max(lo, min(hi, num))

        elif spec.kind == "categorical":
            str_val = str(value).lower().replace(" ", "_").replace("/", "/")
            matched = None
            for cat in spec.categories:
                if cat.lower() == str_val or str_val in cat.lower() or cat.lower() in str_val:
                    matched = cat
                    break
            if matched:
                params[spec.name] = matched
            else:
                missing.append(spec.name)

    if len(missing) > len(space.specs) / 2:
        logger.debug(
            "Skipping literature observation: too many missing params (%s of %s): %s",
            len(missing), len(space.specs), missing,
        )
        return None

    for spec in space.specs:
        if spec.name in params:
            continue
        if spec.kind == "continuous":
            params[spec.name] = (spec.bounds[0] + spec.bounds[1]) / 2
        elif spec.kind == "categorical":
            params[spec.name] = spec.categories[0]

    return params


def _coerce_to_spec(spec, value) -> Optional[ParamValue]:
    """Map a single free-text value onto one ParameterSpec (continuous clamp / categorical match),
    or None if it doesn't fit -- the same logic protocol_to_params applies per parameter."""
    if spec.kind == "continuous":
        try:
            num = float(value)
        except (ValueError, TypeError):
            return None
        lo, hi = spec.bounds
        return max(lo, min(hi, num))
    str_val = str(value).lower()
    for cat in spec.categories:
        if cat.lower() == str_val or str_val in cat.lower() or cat.lower() in str_val:
            return cat
    return None


def _metric_matches(metric_name: str, target_metric: str) -> bool:
    def norm(s: str) -> str:
        return s.lower().replace("_", "").replace(" ", "")
    return norm(metric_name) == norm(target_metric)


def experiment_to_params(candidate, experiment, space: ParameterSpace) -> dict[str, ParamValue]:
    """Map one LiteratureExperiment to the parameter space: start from the protocol's base
    conditions, then override with the conditions this specific experiment actually varied, then
    fill any still-missing dimension with the space midpoint / first category. An optimization
    table's rows differ only in the parameters they varied, so this recovers the real per-row point."""
    base = protocol_to_params(candidate, space)
    params: dict[str, ParamValue] = dict(base) if base else {}

    specs_by_name = {s.name: s for s in space.specs}
    for name, fv in experiment.conditions.items():
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        coerced = _coerce_to_spec(spec, fv.value)
        if coerced is not None:
            params[name] = coerced

    for spec in space.specs:  # fill anything still unknown
        if spec.name in params:
            continue
        params[spec.name] = (spec.bounds[0] + spec.bounds[1]) / 2 if spec.kind == "continuous" else spec.categories[0]
    return params


def selected_experiments_to_observations(
    candidate: ProtocolCandidate, space: ParameterSpace, target_metric: str
) -> list[Observation]:
    """Observations from a candidate's literature experiments the user selected for seeding, whose
    outcome metric matches the target objective. Each experiment's own (varied) conditions become a
    distinct data point -- this is the payload that lets the GP start from the paper's real data."""
    observations = []
    for exp in candidate.literature_experiments:
        if not exp.selected_for_seeding or not _metric_matches(exp.outcome.metric_name, target_metric):
            continue
        observations.append(Observation(
            params=experiment_to_params(candidate, exp, space),
            value=exp.outcome.value,
            uncertainty=exp.outcome.uncertainty,
        ))
    return observations


def outcomes_to_observations(
    candidate: ProtocolCandidate,
    space: ParameterSpace,
    target_metric: str,
) -> list[Observation]:
    """Convert a ProtocolCandidate's measured_outcomes into Observation objects for the GP.

    Only outcomes whose `metric_name` matches `target_metric` (case-insensitive, underscore-
    insensitive) produce observations. Each observation carries the paper's reported uncertainty
    when available; when not, the optimizer's conservative default noise kicks in (see
    single_objective._DEFAULT_NOISE_STD_FOR_UNQUANTIFIED).

    Returns an empty list if the protocol's conditions can't be mapped to the parameter space
    or no matching outcomes exist."""
    params = protocol_to_params(candidate, space)
    if params is None:
        return []

    norm = target_metric.lower().replace("_", "").replace(" ", "")
    observations = []
    for outcome in candidate.measured_outcomes:
        outcome_norm = outcome.metric_name.lower().replace("_", "").replace(" ", "")
        if outcome_norm != norm:
            continue
        observations.append(Observation(
            params=params,
            value=outcome.value,
            uncertainty=outcome.uncertainty,
        ))

    return observations
