"""Material packs -- the one place a material class's chemistry is declared as data.

Everything that is *specific to a material class* (which synthesis conditions are parameters, the
controlled vocabulary each categorical parameter buckets into, the physical ranges a measured
metric may plausibly fall in, which extracted blocks carry a validatable SMILES, and the prompt
framing) lives in a `MaterialPack`. Everything *general* (the GP/BO engine in `optimize`, the
harness loop, the calibration gate, the retrieve->extract->normalize->derive pipeline structure)
stays untouched and consumes a pack through the small surface below.

Adding a material class = author one `MaterialPack` and register it. No edits to `design_space`,
`normalize`, `gates`, or the optimizer.

This module owns only the *vocabulary and config*. The parsing *mechanism* (number/sweep/not-stated
regexes) stays in `literature.normalize`, which a pack calls -- so a pack is pure data plus a couple
of thin dispatch methods, never its own parser.

Versioned data, per the project convention (prices/rubrics/prompts/configs are versioned data, never
constants in code): every pack carries a `version`, and a run records which pack+version produced a
space so a later reader knows the chemistry assumptions in force.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from materials_synthesis_agent.literature import normalize as _nz

# --- field declarations -------------------------------------------------------


@dataclass(frozen=True)
class ContinuousField:
    """A continuous synthesis parameter.

    `parse_kind` is the key handed to `normalize.parse_quantity` (unit-aware parsing); `abs_bounds`
    is the physical clamp the derived range is intersected with, so a mis-extracted 900 h never
    widens the space past what the chemistry allows.
    """
    name: str
    parse_kind: str                     # "temperature_c" | "time_hours" | "concentration_molar" | "ratio" | "generic"
    abs_bounds: tuple[float, float]


@dataclass(frozen=True)
class CategoricalField:
    """A categorical synthesis parameter and its controlled vocabulary.

    `vocab` is precedence-ordered `(canonical_token, keywords)` -- first keyword hit wins, exactly
    the shape `normalize`'s buckets use today, so migrating a COF vocab in is a copy. `mixture_rule`
    is an optional pre-pass for the handful of "both X and Y present -> the mixture" cases
    (dioxane/mesitylene); it returns a canonical token or None.
    """
    name: str
    vocab: tuple[tuple[str, tuple[str, ...]], ...]
    mixture_rule: Optional[Callable[[str], Optional[str]]] = None


@dataclass(frozen=True)
class PackPrompts:
    """The material-class-specific *copy* the extract / expand / reason prompts interpolate.

    The prompt *builders* in `design_space` stay one implementation; only these strings change per
    class. Nothing here is a full prompt -- they are the slots that were hardcoded to COFs.
    """
    persona: str                        # "a materials scientist specializing in COF synthesis"
    bond_noun: str                      # "imine/linkage bond formation" | "metal-carboxylate coordination"
    method_reference: str               # "LFAST (Yaghi group, JACS 2026)" | "modulated synthesis (Behrens)"
    dimension_hints: tuple[str, ...]    # the per-dimension "what to explore" bullets


# --- the pack ----------------------------------------------------------------


@dataclass(frozen=True)
class MaterialPack:
    key: str                            # "cof", "mof"
    display_name: str
    version: str
    continuous_fields: tuple[ContinuousField, ...]
    categorical_fields: tuple[CategoricalField, ...]
    plausible_bounds: dict[str, tuple[float, float]]
    # monomer_roles (schema.ProtocolCandidate.monomer_roles) whose blocks are organic and MUST parse
    # as SMILES. Roles NOT in this set (a MOF metal node / SBU) are skipped by the SMILES gate rather
    # than blocking the whole extraction. Empty set = validate every block (COF default today).
    smiles_roles: frozenset[str]
    prompts: PackPrompts

    # ---- the surface the general pipeline calls -----------------------------

    def continuous_field(self, name: str) -> Optional[ContinuousField]:
        return next((f for f in self.continuous_fields if f.name == name), None)

    def categorical_field(self, name: str) -> Optional[CategoricalField]:
        return next((f for f in self.categorical_fields if f.name == name), None)

    @property
    def continuous_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.continuous_fields)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.categorical_fields)

    def parse_quantity(self, field_name: str, value) -> Optional[float]:
        """Unit-aware parse of a continuous field, via the shared normalize mechanism. None if the
        field is unknown to this pack, not stated, or an unresolved sweep."""
        f = self.continuous_field(field_name)
        if f is None:
            return None
        return _nz.parse_quantity(value, f.parse_kind)

    def canonicalize(self, field_name: str, value) -> Optional[str]:
        """Bucket a categorical value to this pack's vocabulary, or None (not stated / unrecognized).
        Unrecognized is dropped, never used as a raw category -- same discipline as today."""
        f = self.categorical_field(field_name)
        if f is None or _nz.is_not_stated(value):
            return None
        low = str(value).lower()
        if f.mixture_rule is not None:
            mixed = f.mixture_rule(low)
            if mixed is not None:
                return mixed
        for canon, keywords in f.vocab:
            if any(k in low for k in keywords):
                return canon
        return None

    def plausible_outcome(self, metric_name: str, value) -> bool:
        """Physical-plausibility check for a measured metric (drops mis-extractions before the GP).
        Unknown metrics must simply be positive and finite -- an unbounded metric that comes out <=0
        is far likelier an extraction error than a real datum."""
        import math

        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(v):
            return False
        key = metric_name.lower().replace(" ", "_").replace("-", "_")
        if key in self.plausible_bounds:
            lo, hi = self.plausible_bounds[key]
            return lo < v <= hi
        return v > 0

    def validates_smiles_for_role(self, role: Optional[str]) -> bool:
        """True if a block with this monomer_role should be SMILES-validated. Empty smiles_roles
        means validate everything (COF behavior); a role outside a non-empty set (a MOF metal node)
        is skipped."""
        if not self.smiles_roles:
            return True
        return role in self.smiles_roles


# --- registry ----------------------------------------------------------------

_REGISTRY: dict[str, MaterialPack] = {}


def register(pack: MaterialPack) -> MaterialPack:
    _REGISTRY[pack.key] = pack
    return pack


def get_pack(key: str) -> MaterialPack:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown material pack '{key}'. Registered: {sorted(_REGISTRY)}")


def default_pack() -> MaterialPack:
    return _REGISTRY[DEFAULT_PACK_KEY]


DEFAULT_PACK_KEY = "cof"


# =============================================================================
# Pack #1 -- COF (migrates today's hardcoded constants; behavior-preserving)
# =============================================================================

def _cof_solvent_mixture(low: str) -> Optional[str]:
    if "dioxane" in low and "mesitylene" in low:
        return "dioxane/mesitylene"
    return None


_NONE_KW = ("none", "no catalyst", "no modulator", "without", "not used",
            "catalyst-free", "modulator-free")

COF_PACK = register(MaterialPack(
    key="cof",
    display_name="Covalent Organic Framework",
    version="2026-09-17",
    continuous_fields=(
        ContinuousField("temperature_c", "temperature_c", (25.0, 300.0)),
        ContinuousField("time_hours", "time_hours", (0.5, 168.0)),
        ContinuousField("concentration_molar", "concentration_molar", (0.001, 1.0)),
    ),
    categorical_fields=(
        CategoricalField("solvent", (
            ("dioxane", ("dioxane",)),
            ("mesitylene", ("mesitylene",)),
            ("o-dichlorobenzene", ("o-dichlorobenzene", "o-dcb", "odcb", "dichlorobenzene")),
            ("n-butanol", ("n-butanol", "butanol", "buoh")),
            ("NMP", ("nmp", "n-methylpyrrolidone", "methylpyrrolidone")),
            ("DMAc", ("dmac", "dimethylacetamide")),
            ("DMF", ("dmf", "dimethylformamide")),
            ("DMSO", ("dmso", "dimethyl sulfoxide")),
            ("methanol", ("methanol", "meoh")),   # before ethanol -- substring collision
            ("ethanol", ("ethanol", "etoh")),
            ("acetonitrile", ("acetonitrile", "mecn")),
            ("chloroform", ("chloroform", "chcl3")),
            ("toluene", ("toluene",)),
            ("water", ("water", "aqueous", "h2o")),
            ("THF", ("thf", "tetrahydrofuran")),
        ), mixture_rule=_cof_solvent_mixture),
        CategoricalField("catalyst", (
            ("none", _NONE_KW),
            ("acetic acid", ("acetic acid", "hoac", "acoh")),
            ("pyrrolidine", ("pyrrolidine",)),
            ("Sc(OTf)3", ("sc(otf)", "scandium triflate")),
            ("BF3", ("bf3",)),
        )),
        CategoricalField("modulator", (
            ("none", _NONE_KW),
            ("acetic acid", ("acetic acid", "hoac", "acoh", "aa(aq)")),
            ("aniline", ("aniline",)),
            ("trifluoroacetic acid", ("trifluoroacetic", "tfa")),
            ("nitrile", ("benzonitrile", "nitrile")),
        )),
        CategoricalField("synthesis_method", (
            ("mechanochemical", ("mechanochem", "ball mill", "grinding", "grind")),
            ("interfacial", ("interfacial", "liquid-liquid", "liquid-air")),
            ("microwave", ("microwave",)),
            ("vapor-assisted", ("vapor", "vapour")),
            ("solvothermal", ("solvothermal", "sealed tube", "autoclave", "sealed vial", "sealed pyrex")),
            ("room-temperature", ("room-temperature", "room temperature", "rt ", "ambient")),
        )),
        CategoricalField("atmosphere", (
            ("N2", ("n2", "nitrogen", "dinitrogen")),
            ("Ar", ("argon", " ar ", "ar)", "(ar", "ar,", "ar.")),
            ("vacuum", ("vacuum", "evacuat", "degass", "sealed under reduced")),
            ("ambient", ("ambient", "air", "open vessel", "not inert")),
        )),
    ),
    plausible_bounds={
        "bet_surface_area": (1.0, 8000.0),
        "surface_area": (1.0, 8000.0),
        "pore_volume": (0.0, 5.0),
        "total_pore_volume": (0.0, 5.0),
        "pore_size": (0.0, 10.0),
        "co2_uptake": (0.0, 2000.0),
        "yield": (0.0, 100.0),
        "yield_percent": (0.0, 100.0),
        "crystallinity": (0.0, 100.0),
    },
    smiles_roles=frozenset(),  # COFs are all-organic: validate every block, as today
    prompts=PackPrompts(
        persona="a materials scientist specializing in COF synthesis optimization",
        bond_noun="imine/linkage bond formation",
        method_reference="LFAST (Yaghi group, JACS 2026)",
        dimension_hints=(
            "Solvents: what solvent systems haven't been tried? Consider polarity, boiling point, "
            "and compatibility with the linkage chemistry. Mixed solvents at different ratios count.",
            "Temperature: unexplored regimes? Low-temp crystallization (RT-60C) vs high-temp (>150C)?",
            "Modulators: different acid strengths, concentrations, or altogether different modulators?",
            "Time: short reactions (<6h) or very long ones (>7d) not explored?",
            "Concentration: dilute vs concentrated conditions?",
            "Atmosphere: vacuum or specific gas atmospheres?",
        ),
    ),
))


# =============================================================================
# Pack #2 -- MOF (the new class; only data + prompt copy, no engine changes)
# =============================================================================

_MOF_MODULATOR = (
    ("none", _NONE_KW),
    ("formic acid", ("formic acid", "hcooh", "formate")),
    ("acetic acid", ("acetic acid", "hoac", "acoh")),
    ("benzoic acid", ("benzoic acid", "phcooh", "benzoate")),
    ("trifluoroacetic acid", ("trifluoroacetic", "tfa")),
    ("L-proline", ("proline",)),
    ("HCl", ("hydrochloric", "hcl")),
)

MOF_PACK = register(MaterialPack(
    key="mof",
    display_name="Metal-Organic Framework",
    version="2026-09-17",
    continuous_fields=(
        ContinuousField("temperature_c", "temperature_c", (25.0, 250.0)),
        ContinuousField("time_hours", "time_hours", (0.25, 336.0)),
        ContinuousField("concentration_molar", "concentration_molar", (0.001, 2.0)),
        # NEW vs COF: the central MOF lever, and pH (matters for carboxylate deprotonation).
        ContinuousField("metal_linker_ratio", "ratio", (0.1, 20.0)),
        ContinuousField("ph", "generic", (0.0, 14.0)),
    ),
    categorical_fields=(
        CategoricalField("solvent", (
            ("DMF", ("dmf", "dimethylformamide")),
            ("DEF", ("def", "diethylformamide")),
            ("DMA", ("dma", "dimethylacetamide", "dmac")),
            ("water", ("water", "aqueous", "h2o")),
            ("methanol", ("methanol", "meoh")),
            ("ethanol", ("ethanol", "etoh")),
            ("acetonitrile", ("acetonitrile", "mecn")),
            ("DMSO", ("dmso", "dimethyl sulfoxide")),
        )),
        # NEW vs COF: the metal node as a first-class categorical parameter.
        CategoricalField("metal_source", (
            ("ZrCl4", ("zrcl4", "zirconium chloride", "zirconium tetrachloride")),
            ("ZrOCl2", ("zrocl2", "zirconyl chloride")),
            ("Zn(NO3)2", ("zinc nitrate", "zn(no3)")),
            ("Cu(NO3)2", ("copper nitrate", "cu(no3)")),
            ("FeCl3", ("iron chloride", "ferric chloride", "fecl3")),
            ("Al(NO3)3", ("aluminum nitrate", "aluminium nitrate", "al(no3)")),
            ("Cr(NO3)3", ("chromium nitrate", "cr(no3)")),
            ("Co(NO3)2", ("cobalt nitrate", "co(no3)")),
        )),
        CategoricalField("modulator", _MOF_MODULATOR),
        CategoricalField("synthesis_method", (
            ("solvothermal", ("solvothermal", "autoclave", "sealed vial", "sealed vessel", "teflon")),
            ("mechanochemical", ("mechanochem", "ball mill", "grinding", "grind")),
            ("microwave", ("microwave",)),
            ("sonochemical", ("sonochem", "ultrasoni", "sonicat")),
            ("electrochemical", ("electrochem", "anodic", "electrodeposit")),
            ("diffusion", ("diffusion", "layering", "slow evaporation")),
            ("room-temperature", ("room-temperature", "room temperature", "rt ")),
        )),
        CategoricalField("atmosphere", (
            ("N2", ("n2", "nitrogen", "dinitrogen")),
            ("Ar", ("argon", " ar ", "ar)", "(ar", "ar,", "ar.")),
            ("vacuum", ("vacuum", "evacuat", "degass")),
            ("ambient", ("ambient", "air", "open vessel", "not inert")),
        )),
    ),
    plausible_bounds={
        "bet_surface_area": (1.0, 8000.0),   # NU-1501 ~7300; ceiling still covers records
        "surface_area": (1.0, 8000.0),
        "pore_volume": (0.0, 6.0),
        "total_pore_volume": (0.0, 6.0),
        "pore_size": (0.0, 100.0),           # mesoporous MOFs reach tens of nm
        "co2_uptake": (0.0, 2000.0),
        "h2_uptake": (0.0, 200.0),           # NEW: mg/g (or wt% depending on report; widen as needed)
        "ch4_uptake": (0.0, 500.0),          # NEW
        "water_uptake": (0.0, 2000.0),       # NEW
        "yield": (0.0, 100.0),
        "yield_percent": (0.0, 100.0),
        "crystallinity": (0.0, 100.0),
    },
    # MOFs have a metal node that is NOT a small-molecule SMILES. Validate SMILES only on the
    # organic linker roles; skip the node so it doesn't block extraction.
    smiles_roles=frozenset({"linker", "organic_linker", "core"}),
    prompts=PackPrompts(
        persona="a materials scientist specializing in metal-organic framework (MOF) synthesis",
        bond_noun="metal-carboxylate/azolate coordination and SBU formation",
        method_reference="modulated synthesis (Behrens/Schaate coordination-modulation approach)",
        dimension_hints=(
            "Metal source: which metal salts/counterions haven't been tried (nitrate vs chloride vs "
            "acetate)? Different oxidation states or SBU-directing salts?",
            "Metal:linker ratio: stoichiometry strongly controls phase and defectivity -- is the "
            "ratio undersampled? Metal-rich vs linker-rich regimes?",
            "Modulator: monocarboxylic-acid modulators (formic/acetic/benzoic/TFA) at what "
            "equivalents? Modulator concentration is the key defect/crystallinity lever for Zr-MOFs.",
            "Solvent: DMF/DEF/water/alcohol systems and their water content?",
            "Temperature/time: solvothermal ramp and dwell not yet explored?",
            "pH: for aqueous/carboxylate systems, is the deprotonation window covered?",
        ),
    ),
))
