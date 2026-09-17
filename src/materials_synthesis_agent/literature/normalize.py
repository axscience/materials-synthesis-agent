"""Normalize free-text extracted condition values into GP-usable numbers and canonical categories.

Extraction returns conditions as prose -- "Room temperature (RT); also 120C for reference",
"3M HOAc / 1.5M NaCl", "1,4-dioxane/mesitylene (v/v 9:1)", "Not explicitly stated". Fed raw into a
GP those become unusable (every long string is its own category; "not stated" becomes a value).
This module turns a raw value into one of:

  * a float          -- for continuous params, parsed with unit awareness,
  * a canonical str  -- for categorical params, bucketed to a controlled vocabulary,
  * None             -- when the value is not stated, or is an unresolved *sweep* (a whole axis
                        crammed into one field, e.g. "acetic acid at 1M, 3M, 6M"), which must never
                        be coerced to a single fabricated value.

Both the space-derivation path (design_space.derive_parameter_space) and the candidate->observation
path (outcomes) route through here, so the space's categories and a candidate's values are
canonicalized identically and actually match.
"""

from __future__ import annotations

import re
from typing import Optional

# --- "not stated" -------------------------------------------------------------
# Anchored at the start so "Acetic acid ... exact loading not stated" is NOT treated as missing --
# it names a catalyst; only the loading is unknown.
_NOT_STATED_START = re.compile(
    r"(not\s+(?:explicitly\s+|fully\s+|clearly\s+)?"
    r"(?:stated|specified|reported|given|detailed|available|mentioned|provided|determined)"
    r"|unknown|not\s+applicable|no\s+data)", re.I)
_NOT_STATED_WHOLE = {"n/a", "na", "none", "-", "--", "unknown", "not applicable", ""}


def is_not_stated(s) -> bool:
    if not isinstance(s, str) or not s.strip():
        return True
    t = s.strip().lower()
    return t in _NOT_STATED_WHOLE or bool(_NOT_STATED_START.match(t))


# --- numbers & sweeps ---------------------------------------------------------
_NUM = re.compile(r"[-+]?\d*\.?\d+")
_VALUE_UNIT = re.compile(r"([-+]?\d*\.?\d+)\s*([a-zA-Zµ°%]+)")
_ROOM_TEMP = re.compile(r"\broom[\s-]?temperature\b|\brt\b|\bambient\b", re.I)
_SWEEP_WORD = re.compile(r"\b(various|varying|ranging|series of|screened|"
                         r"different\s+\w+\s+concentrations|each\s+of)\b", re.I)


def _distinct(nums, tol=1e-6):
    out = []
    for n in nums:
        if not any(abs(n - m) <= tol for m in out):
            out.append(n)
    return out


def _by_unit(s: str) -> dict[str, list[float]]:
    d: dict[str, list[float]] = {}
    for val, unit in _VALUE_UNIT.findall(s):
        d.setdefault(unit.lower(), []).append(float(val))
    return d


def is_sweep(s) -> bool:
    """True when a field encodes a whole swept axis rather than one value: an explicit sweep word,
    or 3+ distinct numbers sharing the same unit ('1M, 3M, 6M')."""
    if not isinstance(s, str):
        return False
    if _SWEEP_WORD.search(s):
        return True
    return any(len(_distinct(v)) >= 3 for v in _by_unit(s).values())


def sweep_values(s) -> list[float]:
    """The distinct numeric values of the most-populated swept unit (>=2), for splitting a swept
    field into multiple data points when outcomes can be paired to them."""
    if not isinstance(s, str):
        return []
    units = _by_unit(s)
    if not units:
        return []
    _, vals = max(units.items(), key=lambda kv: len(_distinct(kv[1])))
    d = _distinct(vals)
    return sorted(d) if len(d) >= 2 else []


_TEMP_C = re.compile(r"(-?\d*\.?\d+)\s*°?\s*[cC]\b")
_TIME_H = re.compile(r"(\d*\.?\d+)\s*(?:h\b|hr\b|hrs\b|hour)", re.I)
_TIME_D = re.compile(r"(\d*\.?\d+)\s*day", re.I)
_TIME_MIN = re.compile(r"(\d*\.?\d+)\s*min", re.I)


def parse_quantity(s, kind: str = "generic") -> Optional[float]:
    """Parse a numeric value from free text. None if not stated or an unresolved sweep."""
    if is_not_stated(s) or is_sweep(s):
        return None
    s = str(s)
    if kind == "temperature_c":
        m = _TEMP_C.search(s)
        if m:
            return float(m.group(1))
        if _ROOM_TEMP.search(s):
            return 25.0
    elif kind == "time_hours":
        m = _TIME_H.search(s)
        if m:
            return float(m.group(1))
        m = _TIME_D.search(s)
        if m:
            return float(m.group(1)) * 24.0
        m = _TIME_MIN.search(s)
        if m:
            return float(m.group(1)) / 60.0
    elif kind == "concentration_molar":
        for val, unit in _VALUE_UNIT.findall(s):
            u = unit.lower()
            if u == "m":
                return float(val)
            if u == "mm":
                return float(val) / 1000.0
    nums = _NUM.findall(s.replace(",", " "))
    return float(nums[0]) if nums else None


# --- categorical canonicalization --------------------------------------------
# (canonical_token, keywords) -- first match wins, so order encodes precedence.
_SOLVENT = [
    ("dioxane", ("dioxane",)),
    ("mesitylene", ("mesitylene",)),
    ("o-dichlorobenzene", ("o-dichlorobenzene", "o-dcb", "odcb", "dichlorobenzene")),
    ("n-butanol", ("n-butanol", "butanol", "buoh")),
    ("NMP", ("nmp", "n-methylpyrrolidone", "methylpyrrolidone")),
    ("DMAc", ("dmac", "dimethylacetamide")),
    ("DMF", ("dmf", "dimethylformamide")),
    ("DMSO", ("dmso", "dimethyl sulfoxide")),
    ("methanol", ("methanol", "meoh")),   # before ethanol -- "methanol" contains the substring "ethanol"
    ("ethanol", ("ethanol", "etoh")),
    ("acetonitrile", ("acetonitrile", "mecn")),
    ("chloroform", ("chloroform", "chcl3")),
    ("toluene", ("toluene",)),
    ("water", ("water", "aqueous", "h2o")),
    ("THF", ("thf", "tetrahydrofuran")),
]
_NONE = ("none", "no catalyst", "no modulator", "without", "not used", "catalyst-free", "modulator-free")
_CATALYST = [
    ("none", _NONE),
    ("acetic acid", ("acetic acid", "hoac", "acoh")),
    ("pyrrolidine", ("pyrrolidine",)),
    ("Sc(OTf)3", ("sc(otf)", "scandium triflate")),
    ("BF3", ("bf3",)),
]
_MODULATOR = [
    ("none", _NONE),
    ("acetic acid", ("acetic acid", "hoac", "acoh", "aa(aq)")),
    ("aniline", ("aniline",)),
    ("trifluoroacetic acid", ("trifluoroacetic", "tfa")),
    ("nitrile", ("benzonitrile", "nitrile")),
]
_ATMOSPHERE = [
    ("N2", ("n2", "nitrogen", "dinitrogen")),
    ("Ar", ("argon", " ar ", "ar)", "(ar", "ar,", "ar.")),
    ("vacuum", ("vacuum", "evacuat", "degass", "sealed under reduced")),
    ("ambient", ("ambient", "air", "open vessel", "not inert")),
]
_METHOD = [
    ("mechanochemical", ("mechanochem", "ball mill", "grinding", "grind")),
    ("interfacial", ("interfacial", "liquid-liquid", "liquid-air")),
    ("microwave", ("microwave",)),
    ("vapor-assisted", ("vapor", "vapour")),
    ("solvothermal", ("solvothermal", "sealed tube", "autoclave", "sealed vial", "sealed pyrex")),
    ("room-temperature", ("room-temperature", "room temperature", "rt ", "ambient")),
]
_CANON = {
    "solvent": _SOLVENT, "catalyst": _CATALYST, "modulator": _MODULATOR,
    "atmosphere": _ATMOSPHERE, "synthesis_method": _METHOD,
}


def canonicalize_category(field: str, s) -> Optional[str]:
    """Bucket a free-text categorical value to the field's controlled vocabulary, or None when it's
    not stated or matches nothing (an unrecognized value is dropped, never used as a raw category)."""
    if is_not_stated(s):
        return None
    low = str(s).lower()
    # Mixed solvent: both present -> the mixture (checked before single-solvent buckets).
    if field == "solvent" and "dioxane" in low and "mesitylene" in low:
        return "dioxane/mesitylene"
    for canon, keywords in _CANON.get(field, []):
        if any(k in low for k in keywords):
            return canon
    return None
