"""Tests for feasibility.constraints — physical compatibility checking.

This module is designed to be isolatable: if constraints.py is commented out,
nothing else breaks. These tests verify the constraint logic independently.
"""

import pytest

from materials_synthesis_agent.feasibility.constraints import (
    ConstraintCheckResult,
    ConstraintSeverity,
    ConstraintViolation,
    InferredBounds,
    SolventProperties,
    check_concentration_bounds,
    check_physical_constraints,
    check_solvent_temperature,
    check_suggestion_constraints,
    check_time_bounds,
    filter_designs_by_constraints,
    infer_bounds_from_literature,
    infer_parameter_space_from_experiments,
    space_filling_design,
    SOLVENT_DB,
)
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec


# ═══════════════════════════════════════════════════════════════════════════════
# Solvent-temperature compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolventTemperature:
    def test_valid_solvothermal_below_bp(self):
        violations = check_solvent_temperature("dioxane/mesitylene", 90.0, is_sealed_vessel=True)
        errors = [v for v in violations if v.severity == ConstraintSeverity.ERROR]
        assert len(errors) == 0

    def test_valid_solvothermal_above_bp_sealed(self):
        """120C is above dioxane's BP (101C) but fine in a sealed vessel."""
        violations = check_solvent_temperature("dioxane/mesitylene", 120.0, is_sealed_vessel=True)
        errors = [v for v in violations if v.severity == ConstraintSeverity.ERROR]
        assert len(errors) == 0

    def test_error_above_bp_open_vessel(self):
        """120C exceeds dioxane's BP (101C) — error in open vessel."""
        violations = check_solvent_temperature("dioxane/mesitylene", 120.0, is_sealed_vessel=False)
        errors = [v for v in violations if v.severity == ConstraintSeverity.ERROR]
        assert len(errors) > 0
        assert "boiling point" in errors[0].message.lower()

    def test_warning_high_pressure_sealed(self):
        """200C is way above dioxane's BP even sealed — should warn about pressure."""
        violations = check_solvent_temperature("dioxane/mesitylene", 200.0, is_sealed_vessel=True)
        warnings = [v for v in violations if v.severity == ConstraintSeverity.WARNING]
        assert len(warnings) > 0
        assert any("pressure" in w.message.lower() for w in warnings)

    def test_error_below_melting_point(self):
        """DMSO melts at 19C — 10C should be an error."""
        violations = check_solvent_temperature("DMSO", 10.0)
        errors = [v for v in violations if v.severity == ConstraintSeverity.ERROR]
        assert len(errors) > 0
        assert "melting point" in errors[0].message.lower()

    def test_dmf_decomposition_warning(self):
        violations = check_solvent_temperature("DMF", 160.0, is_sealed_vessel=True)
        warnings = [v for v in violations if v.severity == ConstraintSeverity.WARNING]
        assert any("decomp" in w.message.lower() for w in warnings)

    def test_dmso_decomposition_warning(self):
        violations = check_solvent_temperature("DMSO", 190.0, is_sealed_vessel=True)
        warnings = [v for v in violations if v.severity == ConstraintSeverity.WARNING]
        assert any("decomp" in w.message.lower() for w in warnings)

    def test_unknown_solvent_warns(self):
        violations = check_solvent_temperature("xylitol_melt", 100.0)
        warnings = [v for v in violations if v.severity == ConstraintSeverity.WARNING]
        assert len(warnings) == 1
        assert "not in property database" in warnings[0].message

    def test_mixture_checks_all_components(self):
        """n-BuOH/o-DCB — both components should be checked."""
        # 200C is fine for o-DCB (bp 180.5) but triggers a warning for being far above n-butanol (bp 117.7)
        violations = check_solvent_temperature("n-BuOH/o-DCB", 200.0, is_sealed_vessel=True)
        assert len(violations) > 0


class TestTimeBounds:
    def test_valid_time(self):
        assert check_time_bounds(72.0) == []

    def test_negative_time_error(self):
        violations = check_time_bounds(-5.0)
        assert len(violations) == 1
        assert violations[0].severity == ConstraintSeverity.ERROR

    def test_very_long_time_warning(self):
        violations = check_time_bounds(500.0)
        assert len(violations) == 1
        assert violations[0].severity == ConstraintSeverity.WARNING


class TestConcentrationBounds:
    def test_valid_concentration(self):
        assert check_concentration_bounds(6.0) == []

    def test_negative_concentration_error(self):
        violations = check_concentration_bounds(-1.0)
        assert len(violations) == 1
        assert violations[0].severity == ConstraintSeverity.ERROR

    def test_above_glacial_acoh_error(self):
        violations = check_concentration_bounds(20.0)
        assert len(violations) == 1
        assert violations[0].severity == ConstraintSeverity.ERROR
        assert "glacial" in violations[0].message.lower()

    def test_high_concentration_warning(self):
        violations = check_concentration_bounds(14.0)
        assert len(violations) == 1
        assert violations[0].severity == ConstraintSeverity.WARNING


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter space inference
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferBounds:
    def test_basic_padding(self):
        bounds = infer_bounds_from_literature("temperature_c", [90.0, 120.0, 150.0], padding_pct=20.0)
        assert bounds.lower < 90.0
        assert bounds.upper > 150.0
        assert bounds.parameter == "temperature_c"

    def test_hard_limits_respected(self):
        bounds = infer_bounds_from_literature(
            "temperature_c", [90.0, 120.0],
            hard_lower=0.0, hard_upper=200.0, padding_pct=100.0,
        )
        assert bounds.lower >= 0.0
        assert bounds.upper <= 200.0

    def test_single_value_still_works(self):
        bounds = infer_bounds_from_literature("time_hours", [72.0], padding_pct=20.0)
        assert bounds.lower < 72.0
        assert bounds.upper > 72.0


class TestInferParameterSpace:
    def test_mixed_continuous_categorical(self):
        experiments = [
            {"temperature_c": 120.0, "time_hours": 72.0, "solvent": "dioxane/mesitylene"},
            {"temperature_c": 90.0, "time_hours": 48.0, "solvent": "n-BuOH/o-DCB"},
            {"temperature_c": 100.0, "time_hours": 120.0, "solvent": "dioxane/mesitylene"},
        ]
        specs, inferred = infer_parameter_space_from_experiments(experiments)

        cont_specs = [s for s in specs if s.kind == "continuous"]
        cat_specs = [s for s in specs if s.kind == "categorical"]
        assert len(cont_specs) == 2  # temperature_c, time_hours
        assert len(cat_specs) == 1   # solvent
        assert set(cat_specs[0].categories) == {"dioxane/mesitylene", "n-BuOH/o-DCB"}

    def test_hard_limits_applied(self):
        experiments = [
            {"temperature_c": 90.0},
            {"temperature_c": 120.0},
        ]
        specs, inferred = infer_parameter_space_from_experiments(
            experiments,
            hard_limits={"temperature_c": (60.0, 200.0)},
            padding_pct=50.0,
        )
        assert specs[0].bounds[0] >= 60.0
        assert specs[0].bounds[1] <= 200.0

    def test_empty_experiments_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            infer_parameter_space_from_experiments([])


# ═══════════════════════════════════════════════════════════════════════════════
# Space-filling design
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpaceFillingDesign:
    @pytest.fixture
    def mixed_space(self):
        return ParameterSpace([
            ParameterSpec(name="temperature_c", kind="continuous", bounds=(60.0, 200.0)),
            ParameterSpec(name="time_hours", kind="continuous", bounds=(12.0, 168.0)),
            ParameterSpec(name="solvent", kind="categorical", categories=(
                "dioxane/mesitylene", "n-BuOH/o-DCB",
            )),
        ])

    def test_generates_correct_count(self, mixed_space):
        designs = space_filling_design(mixed_space, n_points=10)
        assert len(designs) == 10

    def test_covers_all_categories(self, mixed_space):
        designs = space_filling_design(mixed_space, n_points=10)
        solvents = {d["solvent"] for d in designs}
        assert "dioxane/mesitylene" in solvents
        assert "n-BuOH/o-DCB" in solvents

    def test_continuous_values_in_bounds(self, mixed_space):
        designs = space_filling_design(mixed_space, n_points=20)
        for d in designs:
            assert 60.0 <= d["temperature_c"] <= 200.0
            assert 12.0 <= d["time_hours"] <= 168.0

    def test_deterministic_with_seed(self, mixed_space):
        d1 = space_filling_design(mixed_space, n_points=10, seed=42)
        d2 = space_filling_design(mixed_space, n_points=10, seed=42)
        assert d1 == d2

    def test_different_seeds_different_designs(self, mixed_space):
        d1 = space_filling_design(mixed_space, n_points=10, seed=42)
        d2 = space_filling_design(mixed_space, n_points=10, seed=99)
        assert d1 != d2


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry points
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckPhysicalConstraints:
    def test_valid_cof_lzu1_conditions(self):
        result = check_physical_constraints({
            "temperature_c": 120.0,
            "time_hours": 72.0,
            "acoh_concentration_M": 6.0,
            "solvent": "dioxane/mesitylene",
        })
        assert not result.has_errors

    def test_impossible_conditions_flagged(self):
        result = check_physical_constraints({
            "temperature_c": 200.0,
            "time_hours": 72.0,
            "acoh_concentration_M": 20.0,
            "solvent": "DMF/DMSO",
        })
        assert result.has_errors  # concentration above glacial AcOH

    def test_missing_params_no_crash(self):
        """Partial params shouldn't crash — just check what's present."""
        result = check_physical_constraints({"temperature_c": 100.0})
        assert isinstance(result, ConstraintCheckResult)


class TestCheckSuggestionConstraints:
    def test_in_bounds_and_compatible(self):
        space = ParameterSpace([
            ParameterSpec(name="temperature_c", kind="continuous", bounds=(60.0, 200.0)),
            ParameterSpec(name="solvent", kind="categorical", categories=("dioxane/mesitylene",)),
        ])
        result = check_suggestion_constraints(
            {"temperature_c": 120.0, "solvent": "dioxane/mesitylene"}, space
        )
        assert not result.has_errors

    def test_out_of_bounds_error(self):
        space = ParameterSpace([
            ParameterSpec(name="temperature_c", kind="continuous", bounds=(60.0, 200.0)),
        ])
        result = check_suggestion_constraints({"temperature_c": 250.0}, space)
        assert result.has_errors
        assert "outside parameter space" in result.errors()[0].message

    def test_invalid_category_error(self):
        space = ParameterSpace([
            ParameterSpec(name="solvent", kind="categorical", categories=("dioxane/mesitylene",)),
        ])
        result = check_suggestion_constraints({"solvent": "hexane"}, space)
        assert result.has_errors


class TestFilterDesigns:
    def test_filters_out_impossible(self):
        designs = [
            {"temperature_c": 120.0, "solvent": "dioxane/mesitylene", "acoh_concentration_M": 6.0},
            {"temperature_c": 120.0, "solvent": "dioxane/mesitylene", "acoh_concentration_M": 20.0},
        ]
        valid, rejected = filter_designs_by_constraints(designs)
        assert len(valid) == 1
        assert len(rejected) == 1
        assert rejected[0][1].has_errors


# ═══════════════════════════════════════════════════════════════════════════════
# Solvent database coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolventDB:
    def test_common_cof_solvents_present(self):
        expected = ["1,4-dioxane", "mesitylene", "n-butanol", "o-dichlorobenzene", "DMF", "DMSO"]
        for solvent in expected:
            assert solvent in SOLVENT_DB, f"Missing common COF solvent: {solvent}"

    def test_all_entries_have_bp_mp(self):
        for name, props in SOLVENT_DB.items():
            assert props.boiling_point_c is not None, f"{name} missing boiling point"
            assert props.melting_point_c is not None, f"{name} missing melting point"
            assert props.boiling_point_c > props.melting_point_c, (
                f"{name}: BP ({props.boiling_point_c}) <= MP ({props.melting_point_c})"
            )
