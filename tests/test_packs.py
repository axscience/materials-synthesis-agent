"""Material packs -- the material-class config the general pipeline consumes.

Two things are under test: (1) the COF pack reproduces the behavior that used to be hardcoded in
normalize/design_space/gates (a migration must not change results), and (2) the MOF pack adds the
new class purely as data -- new fields, new vocabulary, and metal-node SMILES skipping -- with no
engine change.
"""

import pytest

from materials_synthesis_agent.harness.gates import _gate_extraction, plausible_outcome
from materials_synthesis_agent.literature import normalize as nz
from materials_synthesis_agent.literature.design_space import derive_parameter_space
from materials_synthesis_agent.materials import COF_PACK, MOF_PACK, default_pack, get_pack
from materials_synthesis_agent.schema import FieldValue, ProtocolCandidate, ProtocolSource


# --- registry ----------------------------------------------------------------
def test_registry_and_default():
    assert get_pack("cof") is COF_PACK
    assert get_pack("mof") is MOF_PACK
    assert default_pack() is COF_PACK
    with pytest.raises(KeyError):
        get_pack("perovskite")


# --- (1) COF pack == today's hardcoded behavior ------------------------------
def test_cof_pack_matches_legacy_normalize():
    """Every COF value must bucket/parse identically through the pack and the standalone normalize
    functions the pack migrated -- otherwise the refactor silently changed the science."""
    cases_cat = [
        ("solvent", "1,4-dioxane/mesitylene (v/v 9:1)"),
        ("solvent", "anhydrous 1,4-dioxane, 4 mL"),
        ("catalyst", "pyrrolidine, 80 uL"),
        ("modulator", "Acetic acid (modulator/catalyst); exact loading not stated"),
        ("synthesis_method", "solvothermal (sealed vial)"),
        ("atmosphere", "under nitrogen atmosphere"),
    ]
    for field, raw in cases_cat:
        assert COF_PACK.canonicalize(field, raw) == nz.canonicalize_category(field, raw)

    for field, raw in [("temperature_c", "Room temperature (RT); also 120C"),
                       ("time_hours", "72 hours (3 days)"),
                       ("concentration_molar", "~203 mM")]:
        assert COF_PACK.parse_quantity(field, raw) == nz.parse_quantity(raw, field)


def test_cof_pack_plausible_matches_legacy_gate():
    for metric, val in [("bet_surface_area", 3000), ("bet_surface_area", 0.0),
                        ("yield", 85), ("yield", 140), ("crystallinity", 50)]:
        assert COF_PACK.plausible_outcome(metric, val) == plausible_outcome(metric, val)


# --- (2) MOF pack: new class, data only --------------------------------------
def test_mof_new_dimensions():
    assert "metal_source" in MOF_PACK.categorical_names
    assert "metal_linker_ratio" in MOF_PACK.continuous_names
    # metal node source buckets to a canonical salt
    assert MOF_PACK.canonicalize("metal_source", "ZrCl4 (zirconium tetrachloride), 0.5 mmol") == "ZrCl4"
    assert MOF_PACK.canonicalize("metal_source", "zinc nitrate hexahydrate") == "Zn(NO3)2"
    # MOF-specific modulators COF didn't carry
    assert MOF_PACK.canonicalize("modulator", "formic acid, 30 equiv") == "formic acid"
    assert MOF_PACK.canonicalize("modulator", "benzoic acid") == "benzoic acid"
    # MOF-specific methods
    assert MOF_PACK.canonicalize("synthesis_method", "ultrasonication for 1 h") == "sonochemical"
    # DEF is a MOF solvent COF's vocab lacked
    assert MOF_PACK.canonicalize("solvent", "N,N-diethylformamide (DEF)") == "DEF"


def test_mof_new_metrics_plausible():
    assert MOF_PACK.plausible_outcome("h2_uptake", 50) is True
    assert MOF_PACK.plausible_outcome("ch4_uptake", 300) is True
    assert MOF_PACK.plausible_outcome("h2_uptake", 0.0) is False   # floor drops a mis-extraction


# --- (2) SMILES-role gating: MOF metal node must not block extraction ---------
def test_smiles_role_gating():
    # COF: empty smiles_roles -> validate every block (unchanged behavior)
    assert COF_PACK.validates_smiles_for_role("node") is True
    # MOF: only organic roles validated; the metal node is skipped
    assert MOF_PACK.validates_smiles_for_role("linker") is True
    assert MOF_PACK.validates_smiles_for_role("node") is False

    # A candidate whose metal node has a non-SMILES value passes under the MOF pack (node skipped)
    # but would be flagged if we validated it as a COF block.
    extraction = {
        "building_blocks": {
            "BDC": {"value": "OC(=O)c1ccc(C(=O)O)cc1"},   # organic linker, valid SMILES
            "Zr6node": {"value": "Zr6O4(OH)4"},           # metal SBU, not a SMILES
        },
        "monomer_roles": {
            "BDC": {"value": "linker"},
            "Zr6node": {"value": "node"},
        },
    }
    assert _gate_extraction(extraction, MOF_PACK).blocked is False
    # Under COF rules (validate everything) the metal node's non-SMILES is caught.
    cof_gate = _gate_extraction(extraction, COF_PACK)
    assert cof_gate.blocked is True and "Zr6node" in cof_gate.detail.get("invalid_blocks", [])


# --- derive_parameter_space respects the pack --------------------------------
def _mof_candidate(metal, solvent, temp, ratio):
    return ProtocolCandidate(
        target_id="t", source=ProtocolSource.MANUAL,
        solvent=FieldValue(value=solvent, inferred=True),
        temperature_c=FieldValue(value=temp, inferred=True),
        modulator=FieldValue(value="formic acid", inferred=True),
    )


def test_derive_space_uses_pack_vocabulary():
    # Two MOF protocols with distinct solvents -> a solvent categorical axis in the MOF vocabulary.
    protos = [
        _mof_candidate("ZrCl4", "DMF", "120", "1:1"),
        _mof_candidate("ZrCl4", "N,N-diethylformamide (DEF)", "100", "2:1"),
    ]
    space = derive_parameter_space(protos, pack=MOF_PACK)
    assert space is not None
    solvent_spec = next(s for s in space.specs if s.name == "solvent")
    assert set(solvent_spec.categories) == {"DMF", "DEF"}
    # temperature axis derived and clamped within the MOF pack's abs bounds (25, 250)
    temp_spec = next(s for s in space.specs if s.name == "temperature_c")
    assert 25.0 <= temp_spec.bounds[0] and temp_spec.bounds[1] <= 250.0
