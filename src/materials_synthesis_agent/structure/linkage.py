"""Linkage-chemistry classification from a CIF's periodic bond graph, for the fallback path when
a structure doesn't match anything in the CURATED-COFs database (structure/database.py) -- an
unnamed, hypothesized COF still needs *some* way to scope the literature search.

Scope, stated plainly rather than implied: this classifies exactly the two linkage chemistries
verified against real, labeled structures from CURATED-COFs (github.com/danieleongari/
CURATED-COFs) --

  - imine: an N atom bonded to two C atoms, one of them a short C=N bond. Measured on COF-300
    (1.267 A) and COF-LZU1 (1.281 A), both real imine-linked COFs in the database (elements
    H,C,N -- no O, no B, ruling out any other N-forming linkage in these two structures).
  - boronate (dioxaborole or boroxine ring): a B atom bonded to two O atoms and one C atom.
    Measured on COF-5 (B-O 1.449 A, B-C 1.537 A) and consistent with COF-1's boroxine-ring boron
    environment. The two ring types aren't distinguished -- both present the same immediate B
    coordination environment, and distinguishing them isn't needed to scope a literature search
    by linkage chemistry.

Hydrazone, imide, and triazine linkages are NOT implemented -- adding them without a labeled
structure to verify bond-length thresholds against would be exactly the kind of unfounded
inference the citation-grounding discipline elsewhere in this package exists to prevent. A
structure that doesn't match either rule returns a LinkageClassification with confidence=0.0 and
an empty `linkage_chemistry`, not a guess.

Periodic-boundary-aware neighbor search via `gemmi.NeighborSearch` -- a COF's bonds routinely
cross the unit cell edge, so a naive single-cell distance check would miss real bonds.
"""

from __future__ import annotations

from materials_synthesis_agent.schema import LinkageClassification
from materials_synthesis_agent.structure.cif import load_small_structure

# Bond-length windows (Angstrom), each anchored to a real measurement above, with a tolerance wide
# enough to cover the "unnatural bond length and geometry" CURATED-COFs' own README warns some of
# its as-reported (pre-DFT-optimization) CIFs carry, but narrow enough to not catch unrelated bonds.
_IMINE_CN_DOUBLE = (1.20, 1.32)  # measured 1.267, 1.281
_BORONATE_BO = (1.30, 1.50)  # measured 1.449
_BORONATE_BC = (1.48, 1.62)  # measured 1.537
_MAX_SEARCH_RADIUS = 2.0


def _neighbor_elements(ns, ss, site, max_dist: float) -> list[tuple[str, float]]:
    marks = ns.find_site_neighbors(site, min_dist=0.1, max_dist=max_dist)
    site_pos = site.orth(ss.cell)
    return [(m.element.name, site_pos.dist(m.pos)) for m in marks]


def classify_linkage(cif_path: str) -> LinkageClassification:
    import gemmi

    ss = load_small_structure(cif_path)
    ns = gemmi.NeighborSearch(ss, _MAX_SEARCH_RADIUS).populate()
    sites = ss.get_all_unit_cell_sites()

    imine_evidence: list[str] = []
    for site in sites:
        if site.element.name != "N":
            continue
        neighbors = _neighbor_elements(ns, ss, site, _MAX_SEARCH_RADIUS)
        carbons = [d for el, d in neighbors if el == "C"]
        if len(carbons) >= 2 and any(_IMINE_CN_DOUBLE[0] <= d <= _IMINE_CN_DOUBLE[1] for d in carbons):
            imine_evidence.append(f"N site with a {min(carbons):.3f} A C=N bond (imine range {_IMINE_CN_DOUBLE})")

    boronate_evidence: list[str] = []
    for site in sites:
        if site.element.name != "B":
            continue
        neighbors = _neighbor_elements(ns, ss, site, _MAX_SEARCH_RADIUS)
        o_bonds = [d for el, d in neighbors if el == "O" and _BORONATE_BO[0] <= d <= _BORONATE_BO[1]]
        c_bonds = [d for el, d in neighbors if el == "C" and _BORONATE_BC[0] <= d <= _BORONATE_BC[1]]
        if len(o_bonds) >= 2 and len(c_bonds) >= 1:
            boronate_evidence.append(f"B site with {len(o_bonds)} B-O bonds (~{sum(o_bonds)/len(o_bonds):.3f} A) and a B-C bond (~{c_bonds[0]:.3f} A)")

    has_imine = len(imine_evidence) > 0
    has_boronate = len(boronate_evidence) > 0

    if has_imine and not has_boronate:
        # Confidence scales with how many independent N sites show the pattern, capped well below
        # 1.0 -- this is a bond-length heuristic over two verified examples, not a validated model.
        confidence = min(0.5 + 0.05 * len(imine_evidence), 0.85)
        return LinkageClassification(linkage_chemistry="imine condensation", confidence=confidence, evidence=imine_evidence[:5])

    if has_boronate and not has_imine:
        confidence = min(0.5 + 0.05 * len(boronate_evidence), 0.85)
        return LinkageClassification(linkage_chemistry="boronate ester", confidence=confidence, evidence=boronate_evidence[:5])

    if has_imine and has_boronate:
        # Both patterns present -- a real possibility (mixed-linkage COFs exist) but ambiguous for
        # this rule set. Report low confidence with both pieces of evidence rather than picking one.
        return LinkageClassification(
            linkage_chemistry="",
            confidence=0.2,
            evidence=["Both imine and boronate signatures found -- ambiguous for this rule set."] + imine_evidence[:2] + boronate_evidence[:2],
        )

    return LinkageClassification(
        linkage_chemistry="",
        confidence=0.0,
        evidence=["No imine or boronate bonding pattern found -- this build only classifies those two linkage chemistries."],
    )
