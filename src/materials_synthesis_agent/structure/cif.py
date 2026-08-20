"""CIF parsing: a file path in, a minimal internal structure representation out.

Uses `gemmi` (zero runtime dependencies -- confirmed via `pip install --dry-run` against this
repo's `numpy`/`scipy`/`torch` pins, which ruled out `pymatgen` for this: its scipy floor moved to
`>=1.13.0`, colliding with the `scipy<1.13` pin `pyproject.toml` carries for torch 2.2's NumPy-1.x
ABI). Verified against real CIFs from the CURATED-COFs database (github.com/danieleongari/
CURATED-COFs, MIT) -- see tests/test_cif.py and structure/database.py's module docstring.

`ParsedStructure` is the minimal, JSON-serializable fingerprint surface `structure/database.py`
needs. Bond-graph analysis (`structure/linkage.py`) works from the raw `gemmi.SmallStructure`
directly via `load_small_structure`, since periodic-boundary-aware neighbor search
(`gemmi.NeighborSearch`) needs the real gemmi object, not this module's flattened view of it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedStructure:
    name: str
    cell: tuple[float, float, float, float, float, float]  # a, b, c, alpha, beta, gamma
    space_group: str
    composition: dict[str, int]  # element symbol -> count, full unit cell (symmetry-expanded)

    @property
    def cell_volume(self) -> float:
        a, b, c, alpha, beta, gamma = self.cell
        al, be, ga = math.radians(alpha), math.radians(beta), math.radians(gamma)
        return a * b * c * math.sqrt(
            1 - math.cos(al) ** 2 - math.cos(be) ** 2 - math.cos(ga) ** 2
            + 2 * math.cos(al) * math.cos(be) * math.cos(ga)
        )

    @property
    def reduced_composition(self) -> tuple[tuple[str, int], ...]:
        """Composition reduced by the GCD of its counts, so a structure and a supercell/subcell
        expansion of the same material fingerprint the same way. Sorted for a stable, hashable
        key."""
        counts = list(self.composition.values())
        divisor = counts[0]
        for c in counts[1:]:
            divisor = math.gcd(divisor, c)
        divisor = divisor or 1
        return tuple(sorted((el, n // divisor) for el, n in self.composition.items()))


def load_small_structure(path: str):
    """Returns the raw `gemmi.SmallStructure` for a CIF file -- used directly by
    structure/linkage.py for periodic-boundary-aware bond analysis. Imports gemmi lazily so the
    base package works without the optional `structure` extra installed."""
    import gemmi

    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()
    return gemmi.make_small_structure_from_block(block)


def parse_cif(path: str) -> ParsedStructure:
    """Parse a CIF into a minimal, JSON-serializable structure representation. Raises whatever
    gemmi raises on a malformed file -- a bad CIF is a real error, not a "no match" outcome."""
    ss = load_small_structure(path)
    sites = ss.get_all_unit_cell_sites()
    composition = dict(Counter(s.element.name for s in sites))
    cell = ss.cell
    return ParsedStructure(
        name=ss.name,
        cell=(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma),
        space_group=ss.spacegroup_hm,
        composition=composition,
    )
