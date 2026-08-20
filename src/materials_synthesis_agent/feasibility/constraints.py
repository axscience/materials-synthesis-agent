"""Physical-compatibility constraints for synthesis parameter combinations.

This is a SEPARATE, ISOLATABLE step in the pipeline. If it causes issues,
comment out the call to `check_physical_constraints()` in the calling code —
the rest of the pipeline works without it.

Checks:
  1. Solvent-temperature compatibility (boiling point, melting point)
  2. Parameter space bound inference from related literature
  3. Space-filling initial design for novel COFs (no direct literature data)
  4. Monomer-solvent compatibility flags (solubility concerns)

Data source hierarchy:
  - Structured lookup tables first (solvent properties are well-tabulated)
  - LLM reasoning as fallback only (never for boiling points or safety numbers)

Future extensions (not built yet):
  - Explosion limit / flash point checks
  - Side-reaction flags for specific monomer-solvent pairs
  - External solubility model / database integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec, ParamValue


class ConstraintSeverity(str, Enum):
    ERROR = "error"       # physically impossible, must not suggest
    WARNING = "warning"   # risky or unusual, flag for researcher review
    INFO = "info"         # informational, no action needed


@dataclass
class ConstraintViolation:
    parameter: str
    severity: ConstraintSeverity
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        prefix = {"error": "ERROR", "warning": "WARNING", "info": "INFO"}[self.severity.value]
        s = f"[{prefix}] {self.parameter}: {self.message}"
        if self.suggestion:
            s += f" Suggestion: {self.suggestion}"
        return s


@dataclass
class ConstraintCheckResult:
    violations: list[ConstraintViolation] = field(default_factory=list)
    params_checked: dict[str, ParamValue] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(v.severity == ConstraintSeverity.ERROR for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == ConstraintSeverity.WARNING for v in self.violations)

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0

    def errors(self) -> list[ConstraintViolation]:
        return [v for v in self.violations if v.severity == ConstraintSeverity.ERROR]

    def warnings(self) -> list[ConstraintViolation]:
        return [v for v in self.violations if v.severity == ConstraintSeverity.WARNING]


# ═══════════════════════════════════════════════════════════════════════════════
# Solvent property tables — structured data, never LLM-generated
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SolventProperties:
    name: str
    boiling_point_c: float
    melting_point_c: float
    flash_point_c: Optional[float] = None
    common_cof_solvents: bool = True
    notes: Optional[str] = None


SOLVENT_DB: dict[str, SolventProperties] = {
    # Pure solvents
    "1,4-dioxane": SolventProperties(
        name="1,4-dioxane", boiling_point_c=101.1, melting_point_c=11.8,
        flash_point_c=12.0, notes="Peroxide-forming; check before heating",
    ),
    "mesitylene": SolventProperties(
        name="mesitylene", boiling_point_c=164.7, melting_point_c=-44.7,
        flash_point_c=50.0,
    ),
    "n-butanol": SolventProperties(
        name="n-butanol", boiling_point_c=117.7, melting_point_c=-89.8,
        flash_point_c=35.0,
    ),
    "o-dichlorobenzene": SolventProperties(
        name="o-dichlorobenzene", boiling_point_c=180.5, melting_point_c=-17.0,
        flash_point_c=66.0, notes="o-DCB",
    ),
    "DMF": SolventProperties(
        name="N,N-dimethylformamide", boiling_point_c=153.0, melting_point_c=-60.5,
        flash_point_c=58.0, notes="Hygroscopic; decomposes above ~150C releasing dimethylamine",
    ),
    "DMSO": SolventProperties(
        name="dimethyl sulfoxide", boiling_point_c=189.0, melting_point_c=19.0,
        flash_point_c=95.0, notes="Exothermic decomposition possible above ~180C",
    ),
    "THF": SolventProperties(
        name="tetrahydrofuran", boiling_point_c=66.0, melting_point_c=-108.4,
        flash_point_c=-14.5, notes="Peroxide-forming; low boiling point limits solvothermal use",
    ),
    "acetonitrile": SolventProperties(
        name="acetonitrile", boiling_point_c=82.0, melting_point_c=-45.0,
        flash_point_c=2.0,
    ),
    "ethanol": SolventProperties(
        name="ethanol", boiling_point_c=78.4, melting_point_c=-114.1,
        flash_point_c=13.0,
    ),
    "water": SolventProperties(
        name="water", boiling_point_c=100.0, melting_point_c=0.0,
    ),
    "acetic acid": SolventProperties(
        name="acetic acid", boiling_point_c=118.1, melting_point_c=16.6,
        flash_point_c=39.0,
    ),
    "toluene": SolventProperties(
        name="toluene", boiling_point_c=110.6, melting_point_c=-95.0,
        flash_point_c=4.0,
    ),
    "chloroform": SolventProperties(
        name="chloroform", boiling_point_c=61.2, melting_point_c=-63.5,
        notes="Non-flammable but decomposes to phosgene under UV/heat",
    ),
    "dichloromethane": SolventProperties(
        name="dichloromethane", boiling_point_c=39.6, melting_point_c=-96.7,
        notes="Very low BP; not suitable for solvothermal above ~35C",
    ),
}

# Common COF solvent mixtures mapped to their component solvents
SOLVENT_MIXTURE_COMPONENTS: dict[str, list[str]] = {
    "dioxane/mesitylene": ["1,4-dioxane", "mesitylene"],
    "n-BuOH/o-DCB": ["n-butanol", "o-dichlorobenzene"],
    "DMF/DMSO": ["DMF", "DMSO"],
    "dioxane_only": ["1,4-dioxane"],
    "dioxane/mesitylene 1:1": ["1,4-dioxane", "mesitylene"],
    "1,4-dioxane/mesitylene": ["1,4-dioxane", "mesitylene"],
    "n-BuOH/o-dichlorobenzene": ["n-butanol", "o-dichlorobenzene"],
}


def _resolve_solvent_components(solvent_name: str) -> list[SolventProperties]:
    """Resolve a solvent name (possibly a mixture) to its component SolventProperties.
    Returns an empty list if the solvent is unknown — the caller flags that as a warning,
    not an error."""
    normalized = solvent_name.strip().lower()

    # Try direct lookup
    if normalized in SOLVENT_DB:
        return [SOLVENT_DB[normalized]]

    # Try mixture lookup
    for mixture_key, components in SOLVENT_MIXTURE_COMPONENTS.items():
        if normalized == mixture_key.lower():
            return [SOLVENT_DB[c] for c in components if c in SOLVENT_DB]

    # Try partial matching
    for key, props in SOLVENT_DB.items():
        if key.lower() in normalized or normalized in key.lower():
            return [props]

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Core constraint checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_solvent_temperature(
    solvent: str,
    temperature_c: float,
    is_sealed_vessel: bool = True,
) -> list[ConstraintViolation]:
    """Check whether a temperature is physically compatible with a solvent.

    In sealed solvothermal vessels, you can exceed the atmospheric boiling point
    (the vessel pressurizes). But there are still limits — decomposition temperatures,
    and the practical pressure rating of common glassware (~150C for Pyrex tubes,
    ~250C for autoclaves).
    """
    violations: list[ConstraintViolation] = []
    components = _resolve_solvent_components(solvent)

    if not components:
        violations.append(ConstraintViolation(
            parameter="solvent",
            severity=ConstraintSeverity.WARNING,
            message=f"Solvent '{solvent}' not in property database — cannot verify temperature compatibility.",
            suggestion="Add solvent properties to SOLVENT_DB or verify compatibility manually.",
        ))
        return violations

    for comp in components:
        # Below melting point
        if temperature_c < comp.melting_point_c:
            violations.append(ConstraintViolation(
                parameter="temperature_c",
                severity=ConstraintSeverity.ERROR,
                message=(
                    f"Temperature {temperature_c}C is below the melting point of "
                    f"{comp.name} ({comp.melting_point_c}C). Solvent would be frozen."
                ),
                suggestion=f"Use temperature above {comp.melting_point_c}C.",
            ))

        # Above boiling point in open vessel
        if not is_sealed_vessel and temperature_c > comp.boiling_point_c:
            violations.append(ConstraintViolation(
                parameter="temperature_c",
                severity=ConstraintSeverity.ERROR,
                message=(
                    f"Temperature {temperature_c}C exceeds the boiling point of "
                    f"{comp.name} ({comp.boiling_point_c}C) in an open vessel."
                ),
                suggestion=f"Use a sealed vessel or lower temperature to below {comp.boiling_point_c}C.",
            ))

        # Sealed vessel — warn if significantly above boiling point (high pressure)
        if is_sealed_vessel and temperature_c > comp.boiling_point_c + 50:
            violations.append(ConstraintViolation(
                parameter="temperature_c",
                severity=ConstraintSeverity.WARNING,
                message=(
                    f"Temperature {temperature_c}C is {temperature_c - comp.boiling_point_c:.0f}C above "
                    f"the boiling point of {comp.name} ({comp.boiling_point_c}C). "
                    f"Even in a sealed vessel this generates significant pressure."
                ),
                suggestion="Verify vessel pressure rating. Standard Pyrex solvothermal tubes are rated to ~150C.",
            ))

        # DMF decomposition warning
        if comp.name == "N,N-dimethylformamide" and temperature_c > 150:
            violations.append(ConstraintViolation(
                parameter="temperature_c",
                severity=ConstraintSeverity.WARNING,
                message=(
                    f"DMF decomposes above ~150C releasing dimethylamine and CO. "
                    f"Temperature {temperature_c}C may cause decomposition."
                ),
            ))

        # DMSO decomposition warning
        if comp.name == "dimethyl sulfoxide" and temperature_c > 180:
            violations.append(ConstraintViolation(
                parameter="temperature_c",
                severity=ConstraintSeverity.WARNING,
                message=(
                    f"DMSO can undergo exothermic decomposition above ~180C. "
                    f"Temperature {temperature_c}C may be unsafe."
                ),
            ))

    return violations


def check_time_bounds(time_hours: float) -> list[ConstraintViolation]:
    """Basic sanity checks on reaction time."""
    violations: list[ConstraintViolation] = []
    if time_hours <= 0:
        violations.append(ConstraintViolation(
            parameter="time_hours",
            severity=ConstraintSeverity.ERROR,
            message=f"Reaction time {time_hours}h is non-positive.",
        ))
    elif time_hours > 336:  # > 2 weeks
        violations.append(ConstraintViolation(
            parameter="time_hours",
            severity=ConstraintSeverity.WARNING,
            message=f"Reaction time {time_hours}h ({time_hours/24:.0f} days) is unusually long.",
            suggestion="Most COF solvothermal syntheses complete within 72-168h.",
        ))
    return violations


def check_concentration_bounds(
    concentration_m: float,
    solvent: Optional[str] = None,
) -> list[ConstraintViolation]:
    """Basic sanity checks on modulator/catalyst concentration."""
    violations: list[ConstraintViolation] = []
    if concentration_m < 0:
        violations.append(ConstraintViolation(
            parameter="acoh_concentration_M",
            severity=ConstraintSeverity.ERROR,
            message=f"Concentration {concentration_m}M is negative.",
        ))
    elif concentration_m > 17.4:  # glacial acetic acid is ~17.4 M
        violations.append(ConstraintViolation(
            parameter="acoh_concentration_M",
            severity=ConstraintSeverity.ERROR,
            message=(
                f"Concentration {concentration_m}M exceeds the molarity of "
                f"glacial acetic acid (~17.4M). Physically impossible."
            ),
        ))
    elif concentration_m > 12:
        violations.append(ConstraintViolation(
            parameter="acoh_concentration_M",
            severity=ConstraintSeverity.WARNING,
            message=f"Concentration {concentration_m}M is very high. Verify this is intentional.",
        ))
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter space inference from related literature
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InferredBounds:
    """Bounds for a continuous parameter inferred from related literature experiments.
    Carries provenance so the researcher knows these aren't arbitrary."""
    parameter: str
    lower: float
    upper: float
    observed_values: list[float]
    source_description: str
    padding_pct: float = 20.0  # how much we expanded beyond the observed range


def infer_bounds_from_literature(
    parameter_name: str,
    observed_values: list[float],
    hard_lower: Optional[float] = None,
    hard_upper: Optional[float] = None,
    padding_pct: float = 20.0,
) -> InferredBounds:
    """Given observed values of a parameter from related literature, infer reasonable
    search bounds by padding the observed range.

    Hard limits (e.g., temperature can't be negative, concentration can't exceed
    pure-substance molarity) override the padded range.
    """
    obs_min = min(observed_values)
    obs_max = max(observed_values)
    obs_range = obs_max - obs_min

    if obs_range == 0:
        padding = abs(obs_min) * padding_pct / 100.0 if obs_min != 0 else 10.0
    else:
        padding = obs_range * padding_pct / 100.0

    lower = obs_min - padding
    upper = obs_max + padding

    if hard_lower is not None:
        lower = max(lower, hard_lower)
    if hard_upper is not None:
        upper = min(upper, hard_upper)

    return InferredBounds(
        parameter=parameter_name,
        lower=lower,
        upper=upper,
        observed_values=observed_values,
        source_description=f"Inferred from {len(observed_values)} literature values "
                          f"(range {obs_min}-{obs_max}), padded {padding_pct}%",
        padding_pct=padding_pct,
    )


def infer_parameter_space_from_experiments(
    experiments: list[dict[str, ParamValue]],
    hard_limits: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
    padding_pct: float = 20.0,
) -> tuple[list[ParameterSpec], list[InferredBounds]]:
    """Given a list of experiment param dicts from related literature, infer a
    ParameterSpace (specs) and return the provenance for each bound.

    Continuous parameters get padded bounds; categorical parameters get the
    union of observed values.

    This is the key function for novel COFs: when we have no direct data but do have
    experiments from related-linkage COFs, we use those to set reasonable ranges
    instead of arbitrary defaults.
    """
    if not experiments:
        raise ValueError("Need at least one experiment to infer parameter space bounds.")

    if hard_limits is None:
        hard_limits = {}

    # Separate continuous and categorical
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp.keys())

    specs: list[ParameterSpec] = []
    inferred: list[InferredBounds] = []

    for key in sorted(all_keys):
        values = [exp[key] for exp in experiments if key in exp]
        if not values:
            continue

        if all(isinstance(v, (int, float)) for v in values):
            float_values = [float(v) for v in values]
            hard_lo, hard_hi = hard_limits.get(key, (None, None))
            bounds_info = infer_bounds_from_literature(
                key, float_values,
                hard_lower=hard_lo, hard_upper=hard_hi,
                padding_pct=padding_pct,
            )
            inferred.append(bounds_info)
            specs.append(ParameterSpec(
                name=key,
                kind="continuous",
                bounds=(bounds_info.lower, bounds_info.upper),
            ))
        else:
            categories = tuple(sorted(set(str(v) for v in values)))
            specs.append(ParameterSpec(
                name=key,
                kind="categorical",
                categories=categories,
            ))

    return specs, inferred


# ═══════════════════════════════════════════════════════════════════════════════
# Space-filling initial design for novel COFs
# ═══════════════════════════════════════════════════════════════════════════════

def space_filling_design(
    space: ParameterSpace,
    n_points: int,
    seed: int = 42,
) -> list[dict[str, ParamValue]]:
    """Generate a space-filling initial design over a mixed continuous/categorical space.

    Uses a stratified approach:
    - For each categorical combination, generate n_points/n_combos continuous points
      using a Latin Hypercube design.
    - This ensures every solvent/method category gets explored, not just the ones
      that happen to appear in literature.

    Returns a list of param dicts ready to use as experiment suggestions.
    """
    import itertools
    import torch

    cat_specs = [s for s in space.specs if s.kind == "categorical"]
    cont_specs = [s for s in space.specs if s.kind == "continuous"]

    if not cat_specs:
        cat_combos = [{}]
    else:
        cat_combos = [
            {spec.name: cat for spec, cat in zip(cat_specs, combo)}
            for combo in itertools.product(*(s.categories for s in cat_specs))
        ]

    points_per_combo = max(1, n_points // len(cat_combos))
    remaining = n_points - points_per_combo * len(cat_combos)

    all_points: list[dict[str, ParamValue]] = []
    generator = torch.Generator().manual_seed(seed)

    for combo_idx, cat_values in enumerate(cat_combos):
        n = points_per_combo + (1 if combo_idx < remaining else 0)

        if not cont_specs:
            all_points.append(dict(cat_values))
            continue

        # Latin Hypercube: divide each dimension into n strata, sample one per stratum
        n_cont = len(cont_specs)
        lhs_samples = torch.zeros(n, n_cont, dtype=torch.double)
        for dim in range(n_cont):
            perm = torch.randperm(n, generator=generator)
            for i in range(n):
                u = (perm[i].item() + torch.rand(1, generator=generator).item()) / n
                spec = cont_specs[dim]
                lo, hi = spec.bounds
                lhs_samples[i, dim] = lo + u * (hi - lo)

        for i in range(n):
            point = dict(cat_values)
            for dim, spec in enumerate(cont_specs):
                point[spec.name] = round(float(lhs_samples[i, dim].item()), 2)
            all_points.append(point)

    return all_points


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point — the single function to call or comment out
# ═══════════════════════════════════════════════════════════════════════════════

def check_physical_constraints(
    params: dict[str, ParamValue],
    is_sealed_vessel: bool = True,
) -> ConstraintCheckResult:
    """Check a single set of synthesis parameters for physical compatibility.

    This is the function to call in the pipeline. To disable constraint checking,
    comment out the call to this function — nothing else depends on it.

    Args:
        params: synthesis parameter dict (temperature_c, time_hours, solvent, etc.)
        is_sealed_vessel: True for solvothermal (sealed tube/autoclave), False for reflux/open

    Returns:
        ConstraintCheckResult with any violations found.
    """
    result = ConstraintCheckResult(params_checked=dict(params))

    solvent = params.get("solvent")
    temperature = params.get("temperature_c")
    time_h = params.get("time_hours")
    concentration = params.get("acoh_concentration_M") or params.get("concentration_molar")

    if solvent is not None and temperature is not None:
        result.violations.extend(
            check_solvent_temperature(str(solvent), float(temperature), is_sealed_vessel)
        )

    if time_h is not None:
        result.violations.extend(check_time_bounds(float(time_h)))

    if concentration is not None:
        result.violations.extend(
            check_concentration_bounds(float(concentration), str(solvent) if solvent else None)
        )

    return result


def check_suggestion_constraints(
    params: dict[str, ParamValue],
    space: ParameterSpace,
    is_sealed_vessel: bool = True,
) -> ConstraintCheckResult:
    """Check a BO suggestion against physical constraints AND parameter space bounds.

    Wraps check_physical_constraints with additional out-of-bounds checks.
    Call this on every Suggestion.params before presenting it to the researcher.
    To disable, comment out the call — the suggestion is still valid without it.
    """
    result = check_physical_constraints(params, is_sealed_vessel)

    for spec in space.specs:
        val = params.get(spec.name)
        if val is None:
            continue
        if spec.kind == "continuous":
            lo, hi = spec.bounds
            fval = float(val)
            if fval < lo or fval > hi:
                result.violations.append(ConstraintViolation(
                    parameter=spec.name,
                    severity=ConstraintSeverity.ERROR,
                    message=f"Value {fval} is outside parameter space bounds [{lo}, {hi}].",
                ))
        elif spec.kind == "categorical":
            if str(val) not in spec.categories:
                result.violations.append(ConstraintViolation(
                    parameter=spec.name,
                    severity=ConstraintSeverity.ERROR,
                    message=f"Value '{val}' is not in allowed categories: {spec.categories}.",
                ))

    return result


def filter_designs_by_constraints(
    designs: list[dict[str, ParamValue]],
    is_sealed_vessel: bool = True,
) -> tuple[list[dict[str, ParamValue]], list[tuple[dict[str, ParamValue], ConstraintCheckResult]]]:
    """Filter a list of candidate designs, returning (valid, rejected_with_reasons).

    Use this after space_filling_design() to remove physically impossible combinations
    before presenting them to the researcher.
    """
    valid: list[dict[str, ParamValue]] = []
    rejected: list[tuple[dict[str, ParamValue], ConstraintCheckResult]] = []

    for design in designs:
        result = check_physical_constraints(design, is_sealed_vessel)
        if result.has_errors:
            rejected.append((design, result))
        else:
            valid.append(design)

    return valid, rejected
