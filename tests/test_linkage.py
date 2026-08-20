from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi", reason="requires the [structure] extra")

from materials_synthesis_agent.structure.linkage import classify_linkage

FIXTURES = Path(__file__).parent / "fixtures" / "cif"


def test_classifies_cof5_as_boronate():
    result = classify_linkage(str(FIXTURES / "COF-5_05001N2.cif"))
    assert result.linkage_chemistry == "boronate ester"
    assert result.confidence > 0
    assert result.evidence


def test_classifies_cof1_as_boronate():
    # COF-1's boroxine ring presents the same B(O,O,C) coordination environment as COF-5's
    # dioxaborole ring -- the classifier doesn't distinguish the two ring types (see
    # structure/linkage.py's module docstring), so both should classify as "boronate ester".
    result = classify_linkage(str(FIXTURES / "COF-1_05000N2.cif"))
    assert result.linkage_chemistry == "boronate ester"


def test_classifies_cof300_as_imine():
    result = classify_linkage(str(FIXTURES / "COF-300_09000N3.cif"))
    assert result.linkage_chemistry == "imine condensation"
    assert result.confidence > 0


def test_classifies_lzu1_as_imine():
    result = classify_linkage(str(FIXTURES / "COF-LZU1_11020N2.cif"))
    assert result.linkage_chemistry == "imine condensation"


def test_imine_evidence_cites_the_measured_bond_length():
    result = classify_linkage(str(FIXTURES / "COF-300_09000N3.cif"))
    assert any("C=N bond" in e for e in result.evidence)


def test_boronate_evidence_cites_the_measured_bonds():
    result = classify_linkage(str(FIXTURES / "COF-5_05001N2.cif"))
    assert any("B-O bonds" in e for e in result.evidence)
