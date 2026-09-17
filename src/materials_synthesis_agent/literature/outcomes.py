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

from materials_synthesis_agent.literature import normalize as _nz
from materials_synthesis_agent.materials import MaterialPack, default_pack
from materials_synthesis_agent.optimize.space import ParameterSpace, ParamValue
from materials_synthesis_agent.optimize.single_objective import Observation
from materials_synthesis_agent.schema import MeasuredOutcome, ProtocolCandidate

logger = logging.getLogger(__name__)


def _extract_param_value(
    candidate: ProtocolCandidate, param_name: str, pack: Optional[MaterialPack] = None
) -> Optional[ParamValue]:
    """Read a named parameter off a ProtocolCandidate and normalize it: a float for continuous
    params (unit-aware), a canonical token for categorical params, or None when the field is absent,
    not stated, or an unresolved sweep. Parsing goes through the campaign's `pack`, so what this
    returns matches the categories `derive_parameter_space` produces (which route through the same
    pack). Falls back to the shared `normalize` mechanism for a hand-specified space whose parameter
    isn't one the pack declares."""
    if pack is None:
        pack = default_pack()
    # Continuous, with the monomer_concentration_M alias for concentration_molar.
    canonical = "concentration_molar" if "concentration" in param_name else param_name
    if pack.continuous_field(canonical) is not None:
        fv = getattr(candidate, canonical, None)
        if fv is None:
            return None
        return pack.parse_quantity(canonical, fv.value)

    # Categorical, matched against the pack's declared fields first.
    for f in pack.categorical_fields:
        if param_name.startswith(f.name):
            fv = getattr(candidate, f.name, None)
            return pack.canonicalize(f.name, fv.value) if fv is not None else None

    # Fallback for a space parameter the pack doesn't declare (hand-specified opt.setup_space).
    for field_name in ("solvent", "catalyst", "modulator", "synthesis_method",
                       "atmosphere", "activation_method"):
        if param_name.startswith(field_name):
            fv = getattr(candidate, field_name, None)
            return _nz.canonicalize_category(field_name, fv.value) if fv is not None else None
    return None


def protocol_to_params(
    candidate: ProtocolCandidate,
    space: ParameterSpace,
    pack: Optional[MaterialPack] = None,
) -> Optional[dict[str, ParamValue]]:
    """Try to map a ProtocolCandidate's extracted fields to a ParameterSpace's parameters.
    Returns None if more than half the parameters can't be resolved — the data point would
    carry too much imputed noise to be useful."""
    if pack is None:
        pack = default_pack()
    params: dict[str, ParamValue] = {}
    missing = []

    for spec in space.specs:
        value = _extract_param_value(candidate, spec.name, pack)

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
            # `value` is already a canonical token (from _extract_param_value); match it exactly
            # against the space's canonical categories.
            matched = next((c for c in spec.categories if c.lower() == str(value).lower()), None)
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


def _coerce_to_spec(spec, value, pack: Optional[MaterialPack] = None) -> Optional[ParamValue]:
    """Map a single free-text value (from an experiment's conditions) onto one ParameterSpec, via
    the campaign's pack so a swept/not-stated value becomes None rather than a fabricated point.
    Falls back to the shared normalizer for a spec the pack doesn't declare."""
    if pack is None:
        pack = default_pack()
    if spec.kind == "continuous":
        if pack.continuous_field(spec.name) is not None:
            num = pack.parse_quantity(spec.name, str(value))
        else:
            kind = spec.name if spec.name in ("temperature_c", "time_hours", "concentration_molar") else "generic"
            num = _nz.parse_quantity(str(value), kind)
        if num is None:
            return None
        lo, hi = spec.bounds
        return max(lo, min(hi, num))
    canon = (pack.canonicalize(spec.name, str(value))
             if pack.categorical_field(spec.name) is not None
             else _nz.canonicalize_category(spec.name, str(value)))
    if canon is None:
        return None
    return next((c for c in spec.categories if c.lower() == canon.lower()), None)


def _metric_matches(metric_name: str, target_metric: str) -> bool:
    def norm(s: str) -> str:
        return s.lower().replace("_", "").replace(" ", "")
    return norm(metric_name) == norm(target_metric)


def experiment_to_params(
    candidate, experiment, space: ParameterSpace, pack: Optional[MaterialPack] = None
) -> dict[str, ParamValue]:
    """Map one LiteratureExperiment to the parameter space: start from the protocol's base
    conditions, then override with the conditions this specific experiment actually varied, then
    fill any still-missing dimension with the space midpoint / first category. An optimization
    table's rows differ only in the parameters they varied, so this recovers the real per-row point."""
    if pack is None:
        pack = default_pack()
    base = protocol_to_params(candidate, space, pack)
    params: dict[str, ParamValue] = dict(base) if base else {}

    specs_by_name = {s.name: s for s in space.specs}
    for name, fv in experiment.conditions.items():
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        coerced = _coerce_to_spec(spec, fv.value, pack)
        if coerced is not None:
            params[name] = coerced

    for spec in space.specs:  # fill anything still unknown
        if spec.name in params:
            continue
        params[spec.name] = (spec.bounds[0] + spec.bounds[1]) / 2 if spec.kind == "continuous" else spec.categories[0]
    return params


def selected_experiments_to_observations(
    candidate: ProtocolCandidate, space: ParameterSpace, target_metric: str,
    pack: Optional[MaterialPack] = None,
) -> list[Observation]:
    """Observations from a candidate's literature experiments the user selected for seeding, whose
    outcome metric matches the target objective. Each experiment's own (varied) conditions become a
    distinct data point -- this is the payload that lets the GP start from the paper's real data."""
    if pack is None:
        pack = default_pack()
    observations = []
    for exp in candidate.literature_experiments:
        if not exp.selected_for_seeding or not _metric_matches(exp.outcome.metric_name, target_metric):
            continue
        observations.append(Observation(
            params=experiment_to_params(candidate, exp, space, pack),
            value=exp.outcome.value,
            uncertainty=exp.outcome.uncertainty,
        ))
    return observations


def outcomes_to_observations(
    candidate: ProtocolCandidate,
    space: ParameterSpace,
    target_metric: str,
    pack: Optional[MaterialPack] = None,
) -> list[Observation]:
    """Convert a ProtocolCandidate's measured_outcomes into Observation objects for the GP.

    Only outcomes whose `metric_name` matches `target_metric` (case-insensitive, underscore-
    insensitive) produce observations. Each observation carries the paper's reported uncertainty
    when available; when not, the optimizer's conservative default noise kicks in (see
    single_objective._DEFAULT_NOISE_STD_FOR_UNQUANTIFIED).

    Returns an empty list if the protocol's conditions can't be mapped to the parameter space
    or no matching outcomes exist."""
    if pack is None:
        pack = default_pack()
    params = protocol_to_params(candidate, space, pack)
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
