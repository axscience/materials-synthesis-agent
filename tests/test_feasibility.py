from materials_synthesis_agent.feasibility import check_protocol_candidate, check_structure
from materials_synthesis_agent.schema import FieldValue, ProtocolCandidate, ProtocolSource


def test_check_structure_valid_smiles():
    result = check_structure("benzene", "c1ccccc1")
    assert result.is_valid_structure
    assert result.canonical_smiles == "c1ccccc1"
    assert result.flags == []


def test_check_structure_invalid_smiles():
    result = check_structure("garbage", "not_a_smiles!!!")
    assert not result.is_valid_structure
    assert result.flags


def test_check_protocol_candidate_flags_invalid_building_block():
    candidate = ProtocolCandidate(
        target_id="t1",
        source=ProtocolSource.MANUAL,
        building_blocks={"bad": FieldValue(value="not_a_smiles!!!", inferred=True)},
    )
    flags = check_protocol_candidate(candidate)
    assert len(flags) == 1
    assert "invalid structure" in flags[0]


def test_check_protocol_candidate_flags_unknown_purchasability_conservatively(monkeypatch):
    # A valid structure whose purchasability was never actually checked (e.g. lookup skipped or
    # unavailable) must still be flagged -- "unknown" is not silently treated as "feasible."
    # BuildingBlockCheck.is_feasible documents this: valid structure + None purchasability != feasible.
    import materials_synthesis_agent.feasibility.checker as checker_module

    def fake_check_building_block(name, smiles, verify_purchasability=True):
        return checker_module.check_structure(name, smiles)  # never runs the purchasability lookup

    monkeypatch.setattr(checker_module, "check_building_block", fake_check_building_block)
    candidate = ProtocolCandidate(
        target_id="t1",
        source=ProtocolSource.MANUAL,
        building_blocks={"benzene": FieldValue(value="c1ccccc1", inferred=True)},
    )
    flags = check_protocol_candidate(candidate)
    assert len(flags) == 1
    assert "not confirmed purchasable" in flags[0]
