from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi", reason="requires the [structure] extra")

from materials_synthesis_agent.structure.cif import parse_cif

FIXTURES = Path(__file__).parent / "fixtures" / "cif"


def test_parse_cif_reads_real_cof5_composition():
    # COF-5 = HHTP + BDBA boronate-ester condensation. Composition confirmed directly against the
    # source CIF (see NOTICE.md), not assumed from the formula alone.
    parsed = parse_cif(str(FIXTURES / "COF-5_05001N2.cif"))
    assert parsed.composition == {"B": 12, "O": 24, "C": 108, "H": 48}


def test_parse_cif_reads_real_cell_parameters():
    parsed = parse_cif(str(FIXTURES / "COF-5_05001N2.cif"))
    a, b, c, alpha, beta, gamma = parsed.cell
    assert a == pytest.approx(29.701, abs=1e-3)
    assert c == pytest.approx(6.9204, abs=1e-3)
    assert gamma == pytest.approx(120.0, abs=1e-3)


def test_parse_cif_space_group():
    parsed = parse_cif(str(FIXTURES / "COF-5_05001N2.cif"))
    assert parsed.space_group == "P 1"


def test_reduced_composition_divides_out_the_gcd():
    parsed = parse_cif(str(FIXTURES / "COF-5_05001N2.cif"))
    # B12 O24 C108 H48 -> GCD 12 -> B1 O2 C9 H4
    assert dict(parsed.reduced_composition) == {"B": 1, "O": 2, "C": 9, "H": 4}


def test_cell_volume_is_positive_and_reasonable():
    parsed = parse_cif(str(FIXTURES / "COF-5_05001N2.cif"))
    # a=b=29.701, c=6.9204, gamma=120 -- hexagonal cell volume = a*b*c*sin(gamma)
    assert parsed.cell_volume == pytest.approx(29.701 * 29.701 * 6.9204 * 0.8660254, rel=1e-3)


def test_parse_cif_on_a_pure_imine_cof_has_no_boron_or_oxygen():
    # COF-300 elements are H,C,N per CURATED-COFs' own metadata -- confirms the fixture really is
    # a non-boronate structure, not an assumption.
    parsed = parse_cif(str(FIXTURES / "COF-300_09000N3.cif"))
    assert "B" not in parsed.composition
    assert "O" not in parsed.composition
    assert parsed.composition.get("N", 0) > 0
