"""Feasibility checking: RDKit validity + commercial-availability lookup.

v0.1 scope only (CLAUDE.md guardrail #8): validity + purchasability. Retrosynthesis is a v0.2
optional adapter (see `feasibility.retrosynthesis`) with a separate, heavier dependency.

Infeasible building blocks are flagged on the ProtocolCandidate, never silently dropped -- the user
decides what to do with an infeasible suggestion, the checker's job is only to surface the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests
from rdkit import Chem

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


@dataclass
class BuildingBlockCheck:
    name: str
    smiles: str
    is_valid_structure: bool
    canonical_smiles: Optional[str] = None
    is_purchasable: Optional[bool] = None  # None = lookup not attempted / lookup failed
    pubchem_cid: Optional[int] = None
    flags: list[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        """Conservative: only "feasible" if structurally valid AND known purchasable. Anything
        else is a flag for the user to look at, not a silent pass."""
        return self.is_valid_structure and bool(self.is_purchasable)


def check_structure(name: str, smiles: str) -> BuildingBlockCheck:
    """Pure-RDKit validity check -- no network call, always available."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return BuildingBlockCheck(
            name=name,
            smiles=smiles,
            is_valid_structure=False,
            flags=[f"'{smiles}' is not a parseable SMILES string (RDKit could not build a molecule)."],
        )
    canonical = Chem.MolToSmiles(mol)
    return BuildingBlockCheck(name=name, smiles=smiles, is_valid_structure=True, canonical_smiles=canonical)


def check_purchasability(check: BuildingBlockCheck, timeout_s: float = 10.0) -> BuildingBlockCheck:
    """Look up commercial availability via PubChem. Network call -- callers should treat failure
    as "unknown," not "infeasible": a lookup failure is not evidence the compound is unavailable."""
    if not check.is_valid_structure:
        return check
    try:
        resp = requests.get(
            f"{PUBCHEM_BASE}/compound/smiles/{requests.utils.quote(check.canonical_smiles)}/cids/JSON",
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            check.flags.append(
                f"PubChem lookup returned {resp.status_code} -- purchasability unknown, not assumed unavailable."
            )
            return check
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            check.is_purchasable = False
            check.flags.append("No PubChem CID found -- likely not a commercially catalogued compound.")
            return check
        check.pubchem_cid = cids[0]
        check.is_purchasable = True
        return check
    except (requests.RequestException, ValueError) as exc:
        check.flags.append(f"PubChem lookup failed ({exc}) -- purchasability unknown, not assumed unavailable.")
        return check


def check_building_block(name: str, smiles: str, verify_purchasability: bool = True) -> BuildingBlockCheck:
    check = check_structure(name, smiles)
    if verify_purchasability and check.is_valid_structure:
        check = check_purchasability(check)
    return check


def check_protocol_candidate(candidate, retrosynthesis_config: Optional[str] = None) -> list[str]:
    """Run feasibility checks over every building block on a ProtocolCandidate and return the
    flags to attach to `candidate.feasibility_flags`. Does not mutate the candidate -- the caller
    decides how to apply the result, matching the append-only/versioned schema discipline.

    If `retrosynthesis_config` is given (a real config.yml path from `setup-retrosynthesis`), a
    non-purchasable block gets a retrosynthesis check instead of just a dead-end flag -- see
    `feasibility.retrosynthesis.check_building_block_with_retrosynthesis`. Left None (the
    default), behavior is unchanged from before retrosynthesis existed.
    """
    flags: list[str] = []
    for bb_name, field_value in candidate.building_blocks.items():
        if retrosynthesis_config is not None:
            from materials_synthesis_agent.feasibility.retrosynthesis import check_building_block_with_retrosynthesis

            result = check_building_block_with_retrosynthesis(bb_name, field_value.value, retrosynthesis_config)
        else:
            result = check_building_block(bb_name, field_value.value)

        if not result.is_valid_structure:
            flags.append(f"{bb_name}: invalid structure -- {'; '.join(result.flags)}")
        elif not result.is_purchasable:
            flags.append(f"{bb_name}: not confirmed purchasable -- {'; '.join(result.flags) or 'no CID found'}")
    return flags
