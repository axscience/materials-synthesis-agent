"""build_index/match_structure tested fully offline against a small local mirror of real
CURATED-COFs metadata (see tests/fixtures/cif/NOTICE.md for the source CIFs and their real IDs) --
no network needed for these. download_curated_cofs itself (the live network fetch) is exercised
separately in TestRealDownload, gated the same way test_retrosynthesis.py gates its real search:
opt-in, not part of the default fast suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi", reason="requires the [structure] extra")

from materials_synthesis_agent.cli.project import default_structure_database_data_dir
from materials_synthesis_agent.structure.cif import parse_cif
from materials_synthesis_agent.structure.database import (
    StructureDatabaseUnavailable,
    build_index,
    download_curated_cofs,
    load_index,
    match_cif,
    match_structure,
    save_index,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cif"
_DEFAULT_INDEX = default_structure_database_data_dir() / "index.json"

# Real rows from CURATED-COFs' own cof-frameworks.csv / cof-papers.csv (fetched directly, not
# invented) for exactly the 4 fixtures this test suite ships.
_FRAMEWORKS_ROWS = [
    {"CURATED-COFs ID": "05000N2", "Source": "tong-v2, 035", "Name": "COF-1", "Elements": "H,B,C,O", "Modifications": "none"},
    {"CURATED-COFs ID": "05001N2", "Source": "tong-v2, 053", "Name": "COF-5", "Elements": "H,B,C,O", "Modifications": "replicated 2x in C direction"},
    {"CURATED-COFs ID": "09000N3", "Source": "tong-v2, 046", "Name": "COF-300", "Elements": "H,C,N", "Modifications": "none"},
    {"CURATED-COFs ID": "11020N2", "Source": "tong-v2, 063", "Name": "COF-LZU1", "Elements": "H,C,N", "Modifications": "replicated 2x in C direction"},
]
_PAPERS_ROWS = [
    {"CURATED-COFs paper ID": "p0500", "Reference": "Science, 2005, 310, 1166-1170", "DOI": "10.1126/science.1120411", "Title": "Porous, crystalline, covalent organic frameworks"},
    {"CURATED-COFs paper ID": "p0900", "Reference": "J. Am. Chem. Soc., 2009, 131, 4570-4571", "DOI": "10.1021/ja8096256", "Title": "A crystalline imine-linked 3-D porous covalent organic framework"},
    {"CURATED-COFs paper ID": "p1102", "Reference": "J. Am. Chem. Soc., 2011, 133, 19816-19822", "DOI": "10.1021/ja206846p", "Title": "Construction of covalent organic framework for catalysis: Pd/COF-LZU1 in Suzuki-Miyaura coupling reaction"},
]


@pytest.fixture
def mini_repo(tmp_path):
    repo_dir = tmp_path / "CURATED-COFs-mini"
    cifs_dir = repo_dir / "cifs"
    cifs_dir.mkdir(parents=True)
    for src_name, curated_id in [
        ("COF-1_05000N2.cif", "05000N2"),
        ("COF-5_05001N2.cif", "05001N2"),
        ("COF-300_09000N3.cif", "09000N3"),
        ("COF-LZU1_11020N2.cif", "11020N2"),
    ]:
        (cifs_dir / f"{curated_id}.cif").write_text((FIXTURES / src_name).read_text())

    with open(repo_dir / "cof-frameworks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_FRAMEWORKS_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(_FRAMEWORKS_ROWS)

    with open(repo_dir / "cof-papers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_PAPERS_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(_PAPERS_ROWS)

    return repo_dir


def test_build_index_indexes_every_real_cif(mini_repo):
    entries = build_index(mini_repo)
    assert {e.name for e in entries} == {"COF-1", "COF-5", "COF-300", "COF-LZU1"}


def test_build_index_resolves_the_real_paper_via_the_id_prefix_convention(mini_repo):
    entries = build_index(mini_repo)
    cof5 = next(e for e in entries if e.name == "COF-5")
    assert cof5.paper_id == "p0500"
    assert cof5.paper_doi == "10.1126/science.1120411"
    assert cof5.paper_title == "Porous, crystalline, covalent organic frameworks"

    lzu1 = next(e for e in entries if e.name == "COF-LZU1")
    assert lzu1.paper_id == "p1102"
    assert lzu1.paper_doi == "10.1021/ja206846p"


def test_match_structure_finds_the_exact_match(mini_repo):
    index = build_index(mini_repo)
    parsed = parse_cif(str(FIXTURES / "COF-5_05001N2.cif"))
    matches = match_structure(parsed, index)
    assert len(matches) == 1
    assert matches[0].matched_name == "COF-5"
    assert matches[0].matched_id == "05001N2"
    assert matches[0].source_database == "CURATED-COFs"
    assert matches[0].paper_doi == "10.1126/science.1120411"


def test_match_structure_does_not_cross_match_different_materials(mini_repo):
    index = build_index(mini_repo)
    parsed = parse_cif(str(FIXTURES / "COF-300_09000N3.cif"))
    matches = match_structure(parsed, index)
    assert len(matches) == 1
    assert matches[0].matched_name == "COF-300"


def test_match_structure_returns_no_matches_for_an_unindexed_structure(mini_repo, tmp_path):
    # Same CIF format, deliberately altered composition -- not one of the 4 indexed structures.
    fake_cif = tmp_path / "not_in_database.cif"
    real = (FIXTURES / "COF-5_05001N2.cif").read_text()
    # Add an extra, chemically nonsensical element line to change the fingerprint composition.
    fake_cif.write_text(real.replace("B          B       0.10820", "Xe         Xe      0.10820"))
    index = build_index(mini_repo)
    parsed = parse_cif(str(fake_cif))
    matches = match_structure(parsed, index)
    assert matches == []


def test_save_and_load_index_round_trips(mini_repo, tmp_path):
    entries = build_index(mini_repo)
    index_path = tmp_path / "index.json"
    save_index(entries, index_path)
    reloaded = load_index(str(index_path))
    assert {e.name for e in reloaded} == {e.name for e in entries}


def test_match_cif_end_to_end_via_a_saved_index(mini_repo, tmp_path):
    entries = build_index(mini_repo)
    index_path = tmp_path / "index.json"
    save_index(entries, index_path)

    matches = match_cif(str(FIXTURES / "COF-LZU1_11020N2.cif"), str(index_path))
    assert len(matches) == 1
    assert matches[0].matched_name == "COF-LZU1"


def test_match_cif_raises_when_no_index_configured():
    with pytest.raises(StructureDatabaseUnavailable):
        match_cif(str(FIXTURES / "COF-5_05001N2.cif"), None)


@pytest.mark.skipif(
    not _DEFAULT_INDEX.exists(),
    reason="Real database not downloaded -- run `materials-agent setup-structure-database` first. "
    "Opt-in like test_retrosynthesis.py's TestRealSearch, not part of the default fast suite.",
)
class TestRealDownload:
    """Confirms the real network fetch (github.com/danieleongari/CURATED-COFs) against the actual
    live archive, not just the offline mini_repo fixture above -- same 'verified against the real
    thing, not just installed' bar feasibility/retrosynthesis.py holds itself to."""

    def test_downloaded_repo_has_the_real_cifs_directory_and_metadata(self, tmp_path):
        repo_dir = download_curated_cofs(tmp_path)
        assert (repo_dir / "cifs").is_dir()
        assert (repo_dir / "cof-frameworks.csv").exists()
        assert (repo_dir / "cof-papers.csv").exists()
        assert len(list((repo_dir / "cifs").glob("*.cif"))) > 900  # ~984 at time of writing
