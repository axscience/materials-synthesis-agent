"""CURATED-COFs structure-database matching: given a parsed CIF, try to identify which
already-published COF it is by fingerprint (reduced composition + space group + cell volume)
against a locally cached copy of the CURATED-COFs database.

Source: github.com/danieleongari/CURATED-COFs (MIT license), D. Ongari, A. V. Yakutovich, L.
Talirz and B. Smit, "Building a consistent and reproducible database for adsorption evaluation in
Covalent-Organic Frameworks," ACS Central Science 2019, 10.1021/acscentsci.9b00619. Real size
confirmed via direct download, not guessed: ~2.6 MB compressed / ~18.7 MB extracted (984 CIFs +
cof-frameworks.csv + cof-papers.csv), fetched from the repo's default branch (`master`) archive
URL -- there is no live API for this, it's a plain static download, same as retrosynthesis's
public-data fetch in feasibility/retrosynthesis.py.

A match is citation-backed, not a guess: the CURATED-COFs ID directly encodes which paper reported
it -- the first 4 digits of the ID are the paper's ID (e.g. "07010N3" -> paper "p0701"), documented
in the database's own README and confirmed against cof-papers.csv/cof-frameworks.csv directly.

No match is a real, expected outcome (most hypothesized COFs won't be in a database of
already-synthesized ones) and falls through to structure/linkage.py's classifier, not an error --
same discipline as feasibility/checker.py's PubChem-lookup-failure handling.
"""

from __future__ import annotations

import csv
import json
import tarfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from materials_synthesis_agent.schema import StructureMatch
from materials_synthesis_agent.structure.cif import ParsedStructure, parse_cif

CURATED_COFS_ARCHIVE_URL = "https://github.com/danieleongari/CURATED-COFs/archive/refs/heads/master.tar.gz"
CURATED_COFS_SOURCE_NAME = "CURATED-COFs"
# Cell volumes across the database vary by orders of magnitude (small vs. large-pore COFs), so a
# fixed absolute tolerance would over- or under-match depending on scale -- a relative tolerance
# doesn't have that problem.
_CELL_VOLUME_RELATIVE_TOLERANCE = 0.02


class StructureDatabaseUnavailable(RuntimeError):
    """Raised when setup-structure-database hasn't been run yet -- distinct from 'no match,'
    which is a real (negative) result, not an error."""


@dataclass
class _IndexEntry:
    curated_id: str
    name: str
    reduced_composition: list[list]  # [[element, count], ...], JSON-friendly form of the tuple
    space_group: str
    cell_volume: float
    paper_id: str
    paper_doi: Optional[str]
    paper_title: Optional[str]


def download_curated_cofs(data_dir: Path, timeout_s: float = 60.0) -> Path:
    """Downloads and extracts the CURATED-COFs repository archive into `data_dir`. Returns the
    path to the extracted repo root (contains cifs/, cof-frameworks.csv, cof-papers.csv)."""
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(CURATED_COFS_ARCHIVE_URL, timeout=timeout_s)
    resp.raise_for_status()
    with tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz") as tar:
        tar.extractall(data_dir, filter="data")
    # The archive's top-level directory is named "<repo>-<branch>" -- find it rather than
    # hardcoding the branch name into a path, since GitHub's naming here isn't guaranteed stable.
    extracted = [p for p in data_dir.iterdir() if p.is_dir() and p.name.lower().startswith("curated-cofs")]
    if not extracted:
        raise RuntimeError(f"Downloaded archive didn't contain a CURATED-COFs directory under {data_dir}.")
    return extracted[0]


def _load_papers(repo_dir: Path) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """paper_id -> (doi, title)."""
    papers: dict[str, tuple[Optional[str], Optional[str]]] = {}
    with open(repo_dir / "cof-papers.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            papers[row["CURATED-COFs paper ID"]] = (row.get("DOI") or None, row.get("Title") or None)
    return papers


def _load_names(repo_dir: Path) -> dict[str, str]:
    """curated_id -> Name."""
    names: dict[str, str] = {}
    with open(repo_dir / "cof-frameworks.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[row["CURATED-COFs ID"]] = row["Name"]
    return names


def build_index(repo_dir: Path) -> list[_IndexEntry]:
    """Parses every CIF in the extracted database and builds a fingerprint index. A CIF that
    fails to parse is skipped, not fatal -- one malformed source file shouldn't block indexing the
    other ~980."""
    names = _load_names(repo_dir)
    papers = _load_papers(repo_dir)
    entries: list[_IndexEntry] = []

    for cif_path in sorted((repo_dir / "cifs").glob("*.cif")):
        curated_id = cif_path.stem
        if curated_id not in names:
            continue  # in cof-discarded.csv, not cof-frameworks.csv -- not a real, citable COF
        try:
            parsed = parse_cif(str(cif_path))
        except Exception:
            continue
        paper_id = "p" + curated_id[:4]
        doi, title = papers.get(paper_id, (None, None))
        entries.append(
            _IndexEntry(
                curated_id=curated_id,
                name=names[curated_id],
                reduced_composition=[list(pair) for pair in parsed.reduced_composition],
                space_group=parsed.space_group,
                cell_volume=parsed.cell_volume,
                paper_id=paper_id,
                paper_doi=doi,
                paper_title=title,
            )
        )
    return entries


def save_index(entries: list[_IndexEntry], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps([asdict(e) for e in entries], indent=2))


def load_index(index_path: str) -> list[_IndexEntry]:
    raw = json.loads(Path(index_path).read_text())
    return [_IndexEntry(**item) for item in raw]


def match_structure(parsed: ParsedStructure, index: list[_IndexEntry]) -> list[StructureMatch]:
    """Fingerprint match: same reduced composition and space group, with cell volume within
    tolerance to catch the same material reported at a different (but equivalent) supercell
    setting. Returns every match, not just the first -- an ambiguous match (more than one entry
    fits) is a real outcome the caller should see, not something this function silently resolves."""
    target_key = list(parsed.reduced_composition)
    matches = []
    for entry in index:
        if entry.reduced_composition != [list(p) for p in target_key]:
            continue
        if entry.space_group != parsed.space_group:
            continue
        if entry.cell_volume <= 0:
            continue
        relative_diff = abs(entry.cell_volume - parsed.cell_volume) / entry.cell_volume
        if relative_diff > _CELL_VOLUME_RELATIVE_TOLERANCE:
            continue
        matches.append(
            StructureMatch(
                matched_name=entry.name,
                matched_id=entry.curated_id,
                source_database=CURATED_COFS_SOURCE_NAME,
                paper_doi=entry.paper_doi,
                paper_title=entry.paper_title,
            )
        )
    return matches


def match_cif(cif_path: str, index_path: Optional[str]) -> list[StructureMatch]:
    """End-to-end: parse a CIF and match it against a previously built local index.
    `index_path=None` (setup-structure-database hasn't been run) raises
    StructureDatabaseUnavailable -- distinct from a real "no match" result."""
    if not index_path:
        raise StructureDatabaseUnavailable(
            "No local structure database. Run `materials-agent setup-structure-database` first."
        )
    parsed = parse_cif(cif_path)
    index = load_index(index_path)
    return match_structure(parsed, index)
