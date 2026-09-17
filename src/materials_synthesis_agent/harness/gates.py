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

from typing import Any, Literal

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


def _gate_extraction(output: Any) -> Gate:
    """Extracted building-block SMILES must be chemically valid. The citation-or-inferred invariant
    is already guaranteed by the FieldValue schema validator, so it needs no gate here."""
    d = _as_dict(output)
    blocks = d.get("building_blocks") or {}
    if not isinstance(blocks, dict):
        return Gate.ok()
    invalid = []
    for name, fv in blocks.items():
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


def run_gates(tool_name: str, output: Any) -> Gate:
    gate = _GATES.get(tool_name)
    return gate(output) if gate else Gate.ok()
