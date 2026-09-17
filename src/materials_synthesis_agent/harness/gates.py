"""Deterministic gates -- what replaces the Critic Agent.

Every check the critic was meant to make is deterministic and already coded elsewhere; the gates
just run it inline on a tool's output and return a `Gate` the planner reads. Two kinds:

  * blocking  -> the tool_result comes back with is_error=True; the planner must handle it (log more
    results, tell the user, route around). Calibration (raised inside the optimizer handler) and
    invalid extractions and missing-uncertainty numbers are blocking -- "no number without
    uncertainty" is the worst failure to ship silently.
  * advisory  -> surfaced in the result but not an error. Feasibility is advisory, matching the
    synthesis package's "flag, don't drop" philosophy: a non-purchasable block might still be
    something the user can make.

No LLM is involved. A judgment-based critic (e.g. "this contradicts observation X") is deliberately
out of v1; when added it annotates, it never triggers an automatic replan.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

Severity = Literal["ok", "info", "warning", "error"]


class Gate(BaseModel):
    blocked: bool = False
    severity: Severity = "ok"
    message: str = ""
    detail: dict[str, Any] = {}

    @classmethod
    def ok(cls) -> "Gate":
        return cls()

    @classmethod
    def block(cls, message: str, **detail: Any) -> "Gate":
        return cls(blocked=True, severity="error", message=message, detail=detail)

    @classmethod
    def warn(cls, message: str, **detail: Any) -> "Gate":
        return cls(blocked=False, severity="warning", message=message, detail=detail)


# Plausible ranges for common COF outcome metrics, used to drop extraction errors (a "not
# reported" mis-parsed as 0.0, a negative surface area, an absurd value) before they poison the GP.
# Versioned data, not magic constants: widen/add entries as new metrics appear. Keys are normalized
# (lowercase, spaces/hyphens -> underscore).
_PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "bet_surface_area": (1.0, 8000.0),      # m2/g; COFs top out ~5000-7000
    "surface_area": (1.0, 8000.0),
    "pore_volume": (0.0, 5.0),              # cm3/g
    "total_pore_volume": (0.0, 5.0),
    "pore_size": (0.0, 10.0),               # nm
    "co2_uptake": (0.0, 2000.0),            # mg/g
    "yield": (0.0, 100.0),                  # %
    "yield_percent": (0.0, 100.0),
    "crystallinity": (0.0, 100.0),          # fractional or %, both < 100
}


def plausible_outcome(metric_name: str, value, pack=None) -> bool:
    """True if `value` is a physically plausible measurement of `metric_name`. Rejects non-finite
    values, and anything at or below the metric's floor (so a 0.0 surface area -- almost always a
    mis-extraction of 'not reported' -- is dropped) or above its ceiling. Metrics with no known
    bounds must simply be positive and finite: a measured property that comes out <= 0 is far more
    likely an extraction error than a real datum.

    When a `pack` is given, its plausibility bounds are used (a MOF's h2/ch4/water uptake, etc.);
    otherwise the legacy COF bounds below apply, so every existing caller is unchanged."""
    if pack is not None:
        return pack.plausible_outcome(metric_name, value)

    import math

    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    key = metric_name.lower().replace(" ", "_").replace("-", "_")
    if key in _PLAUSIBLE_BOUNDS:
        lo, hi = _PLAUSIBLE_BOUNDS[key]
        return lo < v <= hi
    return v > 0


def validate_smiles(smiles: str) -> bool:
    """True if RDKit can parse the SMILES. If RDKit is unavailable, we do not claim validity or
    invalidity -- the caller treats `None` as "not checked" (see _gate_extraction)."""
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return True  # RDKit absent: don't block on a check we can't run
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _as_dict(output: Any) -> dict:
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    return {}


def _gate_suggestion(output: Any) -> Gate:
    """A next-experiment suggestion must carry a usable uncertainty. Calibration itself is enforced
    upstream (the optimizer raises CalibrationError before we get here); this catches a suggestion
    that somehow arrives with no interval."""
    d = _as_dict(output)
    if not d:
        return Gate.ok()
    u = d.get("predicted_uncertainty")
    if u is None or (isinstance(u, (int, float)) and u <= 0):
        return Gate.block(
            "Suggestion has no usable uncertainty -- refusing to present a number without an "
            "interval.",
            predicted_uncertainty=u,
        )
    return Gate.ok()


def _gate_extraction(output: Any, pack=None) -> Gate:
    """Extracted building-block SMILES must be chemically valid. The citation-or-inferred invariant
    is already guaranteed by the FieldValue schema validator, so it needs no gate here.

    A block is SMILES-validated only when its monomer_role is one the pack marks as organic
    (`smiles_roles`). This is what lets a MOF pass: its metal node / SBU is not a small-molecule
    SMILES, so it is skipped rather than blocking the whole extraction. With no pack (or an empty
    smiles_roles, the COF default) every block is validated, as before."""
    d = _as_dict(output)
    blocks = d.get("building_blocks") or {}
    if not isinstance(blocks, dict):
        return Gate.ok()
    roles = d.get("monomer_roles") or {}
    invalid = []
    for name, fv in blocks.items():
        if pack is not None:
            role_fv = roles.get(name) if isinstance(roles, dict) else None
            role = role_fv.get("value") if isinstance(role_fv, dict) else getattr(role_fv, "value", None)
            if not pack.validates_smiles_for_role(role):
                continue
        smiles = fv.get("value") if isinstance(fv, dict) else getattr(fv, "value", None)
        if smiles and not validate_smiles(str(smiles)):
            invalid.append(name)
    if invalid:
        return Gate.block(
            f"Extraction produced invalid SMILES for: {', '.join(invalid)}.",
            invalid_blocks=invalid,
        )
    return Gate.ok()


def _gate_feasibility(output: Any) -> Gate:
    """Advisory: surface non-purchasable / non-synthesizable blocks without blocking -- the user may
    still be able to make them. Expects a feasibility report dict with a 'flags' or
    'non_purchasable' list."""
    d = _as_dict(output)
    flags = d.get("flags") or d.get("non_purchasable") or []
    if flags:
        return Gate.warn(
            f"Feasibility flags on: {', '.join(map(str, flags))} (not blocking).",
            flags=list(flags),
        )
    return Gate.ok()


# tool name -> gate function. Tools not listed here pass through ungated.
_GATES = {
    "opt.suggest_next": _gate_suggestion,
    "lit.extract_protocol": _gate_extraction,
    "feas.check": _gate_feasibility,
    "design.handoff": _gate_feasibility,
}


def run_gates(tool_name: str, output: Any, pack=None) -> Gate:
    gate = _GATES.get(tool_name)
    if gate is None:
        return Gate.ok()
    # Only the extraction gate is pack-aware (SMILES-role selection); the others ignore it.
    if gate is _gate_extraction:
        return gate(output, pack)
    return gate(output)
