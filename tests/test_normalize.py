"""Value normalization -- tested against the actual free-text the extractor produced on COF-LZU1."""

import pytest

from materials_synthesis_agent.literature.normalize import (
    canonicalize_category,
    is_not_stated,
    is_sweep,
    parse_quantity,
    sweep_values,
)


# --- (3) not-stated -> missing, but only when the value IS a not-stated statement ---
def test_not_stated_detection():
    assert is_not_stated("Not explicitly stated in the provided excerpt") is True
    assert is_not_stated("") is True
    assert is_not_stated("n/a") is True
    assert is_not_stated(None) is True
    # names a catalyst; only the loading is unknown -> NOT missing
    assert is_not_stated("Acetic acid (modulator/catalyst); exact loading not stated") is False
    assert is_not_stated("none reported") is False
    assert is_not_stated("120") is False


# --- (1) amount parsing, unit-aware ---
def test_parse_temperature():
    assert parse_quantity("120", "temperature_c") == 120.0
    assert parse_quantity("Room temperature (RT); also 120°C for reference", "temperature_c") == 120.0
    assert parse_quantity("Room temperature (RT)", "temperature_c") == 25.0
    assert parse_quantity("Not explicitly stated in the provided excerpt", "temperature_c") is None


def test_parse_time():
    assert parse_quantity("72 hours (3 days)", "time_hours") == 72.0
    assert parse_quantity("3 days", "time_hours") == 72.0
    assert parse_quantity("30 min", "time_hours") == pytest.approx(0.5)
    assert parse_quantity("a few minutes", "time_hours") is None  # no number


def test_parse_concentration():
    assert parse_quantity("~203 mM", "concentration_molar") == pytest.approx(0.203)
    assert parse_quantity("3M HOAc / 1.5M NaCl", "concentration_molar") == 3.0


# --- (4) sweeps -> not a single value ---
def test_sweep_detection_and_values():
    assert is_sweep("Aqueous acetic acid at various concentrations (1M, 3M, 6M)") is True
    assert is_sweep("1M, 3M, 6M") is True
    assert sweep_values("1M, 3M, 6M") == [1.0, 3.0, 6.0]
    # a compound description (different units) is NOT a sweep
    assert is_sweep("Tp: 0.34 mmol in 4 mL dioxane (~203 mM total)") is False
    assert is_sweep("3M HOAc / 1.5M NaCl") is False   # only 2 distinct M values
    # a sweep must not be coerced to a single number
    assert parse_quantity("acetic acid at 1M, 3M, 6M", "concentration_molar") is None


# --- (2) categorical canonicalization ---
def test_canonicalize_solvent():
    assert canonicalize_category("solvent", "1,4-dioxane/mesitylene (v/v 9:1) with aqueous catalyst") == "dioxane/mesitylene"
    assert canonicalize_category("solvent", "anhydrous 1,4-dioxane, 4 mL") == "dioxane"
    assert canonicalize_category("solvent", "water (aqueous solution)") == "water"
    assert canonicalize_category("solvent", "Not explicitly stated") is None


def test_canonicalize_catalyst_modulator():
    assert canonicalize_category("catalyst", "pyrrolidine, 80 µL") == "pyrrolidine"
    assert canonicalize_category("catalyst", "Acetic acid (modulator/catalyst); exact loading not stated") == "acetic acid"
    assert canonicalize_category("modulator", "none reported") == "none"
    assert canonicalize_category("modulator", "Aqueous acetic acid (HOAc) at various concentrations (1M, 3M, 6M)") == "acetic acid"


def test_canonicalize_method_and_atmosphere():
    assert canonicalize_category("synthesis_method", "solvothermal (hot plate, sealed vial)") == "solvothermal"
    assert canonicalize_category("synthesis_method", "room-temperature solution") == "room-temperature"
    assert canonicalize_category("atmosphere", "ambient (not explicitly stated as inert)") == "ambient"
    assert canonicalize_category("atmosphere", "under nitrogen atmosphere") == "N2"
    assert canonicalize_category("atmosphere", "Not explicitly stated in the provided excerpt") is None
