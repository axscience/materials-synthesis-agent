"""Post-extraction design space analysis: expand beyond what the literature reports, auto-derive
a parameter space, and synthesize cross-protocol reasoning.

Three distinct capabilities, each independently callable:

  1. `expand_design_space` -- LFAST-inspired: after extracting protocols, ask the LLM what
     conditions *should* be tried but *haven't* been reported. This is the gap between "extract
     what papers did" and "propose what papers haven't done."

  2. `derive_parameter_space` -- Build a ParameterSpace automatically from the union of conditions
     across all extracted protocols, with ±20% margins on continuous ranges and the full set of
     observed categories. No LLM call, pure data analysis.

  3. `cross_protocol_reasoning` -- Synthesize across all protocols: consensus conditions, variation
     hotspots, contradictions, coverage gaps, and a recommended starting point. Requires an LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from materials_synthesis_agent.llm import LLMClient
from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec
from materials_synthesis_agent.schema import ProtocolCandidate, Target

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 2: Auto-derive ParameterSpace from extracted protocols (no LLM needed)
# ---------------------------------------------------------------------------

_CONTINUOUS_FIELDS = {
    "temperature_c": (25.0, 300.0),
    "time_hours": (0.5, 168.0),
    "concentration_molar": (0.001, 1.0),
}

_CATEGORICAL_FIELDS = ["solvent", "catalyst", "modulator", "synthesis_method", "atmosphere"]


def _parse_numeric(value: str) -> Optional[float]:
    """Best-effort numeric extraction from a free-text field value."""
    cleaned = value.split()[0].rstrip("°CcMmHh%")
    try:
        return float(cleaned)
    except (ValueError, IndexError):
        return None


def _normalize_category(value: str) -> str:
    return value.strip().lower().replace("  ", " ")


def derive_parameter_space(
    protocols: list[ProtocolCandidate],
    margin_fraction: float = 0.2,
    min_margin: float = 5.0,
) -> Optional[ParameterSpace]:
    """Build a ParameterSpace from the union of conditions across all extracted protocols.

    Continuous parameters get bounds from [min - margin, max + margin] of observed values,
    clamped to physically reasonable floors/ceilings. Categorical parameters collect every
    distinct value seen. Parameters observed in fewer than 2 protocols are skipped (not enough
    signal to define a meaningful range).

    Returns None if fewer than 2 protocols have extractable conditions."""
    if len(protocols) < 2:
        return None

    continuous_values: dict[str, list[float]] = {k: [] for k in _CONTINUOUS_FIELDS}
    categorical_values: dict[str, set[str]] = {k: set() for k in _CATEGORICAL_FIELDS}

    from materials_synthesis_agent.literature.normalize import (
        canonicalize_category,
        parse_quantity,
    )

    for p in protocols:
        for field_name in _CONTINUOUS_FIELDS:
            fv = getattr(p, field_name, None)
            if fv is not None:
                num = parse_quantity(fv.value, field_name)
                if num is not None:
                    continuous_values[field_name].append(num)

        for exp in p.literature_experiments:
            for cond_name, cond_fv in exp.conditions.items():
                if cond_name in continuous_values:
                    num = parse_quantity(cond_fv.value, cond_name)
                    if num is not None:
                        continuous_values[cond_name].append(num)
                elif cond_name in categorical_values:
                    canon = canonicalize_category(cond_name, cond_fv.value)
                    if canon is not None:
                        categorical_values[cond_name].add(canon)

        for field_name in _CATEGORICAL_FIELDS:
            fv = getattr(p, field_name, None)
            if fv is not None:
                canon = canonicalize_category(field_name, fv.value)
                if canon is not None:  # unrecognized/not-stated -> not a category
                    categorical_values[field_name].add(canon)

    specs: list[ParameterSpec] = []
    for field_name, (abs_lo, abs_hi) in _CONTINUOUS_FIELDS.items():
        vals = continuous_values[field_name]
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        margin = max((hi - lo) * margin_fraction, min_margin)
        specs.append(ParameterSpec(
            name=field_name,
            kind="continuous",
            bounds=(max(abs_lo, lo - margin), min(abs_hi, hi + margin)),
        ))

    for field_name in _CATEGORICAL_FIELDS:
        cats = sorted(categorical_values[field_name])
        if len(cats) < 2:
            continue
        specs.append(ParameterSpec(
            name=field_name,
            kind="categorical",
            categories=tuple(cats),
        ))

    if not specs:
        return None

    logger.info(
        "Derived parameter space: %d continuous + %d categorical dimensions from %d protocols",
        sum(1 for s in specs if s.kind == "continuous"),
        sum(1 for s in specs if s.kind == "categorical"),
        len(protocols),
    )
    return ParameterSpace(specs)


# ---------------------------------------------------------------------------
# Step 1: Design space expansion (LLM-assisted, LFAST steps 4-5 equivalent)
# ---------------------------------------------------------------------------

@dataclass
class ExpansionSuggestion:
    """One untried condition the LLM proposes, with its reasoning."""
    dimension: str
    suggested_value: str
    rationale: str


@dataclass
class DesignSpaceExpansion:
    """The result of asking "what should we try that hasn't been reported?" """
    suggestions: list[ExpansionSuggestion] = field(default_factory=list)
    raw_text: str = ""


_EXPANSION_TOOL_NAME = "propose_untried_conditions"
_EXPANSION_TOOL_DESCRIPTION = (
    "Propose synthesis conditions that haven't been tried in the extracted literature but are "
    "chemically reasonable and worth exploring for this material."
)
_EXPANSION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "Which parameter dimension: 'solvent', 'temperature_c', "
                        "'time_hours', 'modulator', 'catalyst', 'concentration_molar', 'atmosphere'.",
                    },
                    "suggested_value": {
                        "type": "string",
                        "description": "The specific condition to try (e.g. 'DMF/DMSO 4:1 v/v', '180', '0.5 M TFA').",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this is worth trying -- chemical reasoning, analogy to related systems, "
                        "or a gap in the explored space.",
                    },
                },
                "required": ["dimension", "suggested_value", "rationale"],
            },
        },
    },
    "required": ["suggestions"],
}


def _summarize_protocols(protocols: list[ProtocolCandidate]) -> str:
    """Compact summary of extracted conditions for the expansion prompt."""
    lines = []
    for i, p in enumerate(protocols, 1):
        parts = [f"Protocol {i}:"]
        if p.building_blocks:
            parts.append(f"  Monomers: {', '.join(p.building_blocks.keys())}")
        if p.solvent:
            parts.append(f"  Solvent: {p.solvent.value}")
        if p.temperature_c:
            parts.append(f"  Temperature: {p.temperature_c.value}")
        if p.time_hours:
            parts.append(f"  Time: {p.time_hours.value}")
        if p.modulator:
            parts.append(f"  Modulator: {p.modulator.value}")
        if p.catalyst:
            parts.append(f"  Catalyst: {p.catalyst.value}")
        if p.concentration_molar:
            parts.append(f"  Concentration: {p.concentration_molar.value}")
        if p.atmosphere:
            parts.append(f"  Atmosphere: {p.atmosphere.value}")
        if p.synthesis_method:
            parts.append(f"  Method: {p.synthesis_method.value}")
        n_exp = len(p.literature_experiments)
        if n_exp:
            parts.append(f"  Experiments reported: {n_exp}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _build_expansion_prompt(target: Target, protocols: list[ProtocolCandidate]) -> str:
    target_parts = []
    if target.name:
        target_parts.append(f"Material: {target.name}")
    target_parts.append(f"Linkage: {target.linkage_chemistry}")
    if target.functional_groups:
        target_parts.append(f"Monomers: {', '.join(target.functional_groups)}")
    objectives = target.all_objectives
    if objectives:
        target_parts.append(f"Optimizing: {', '.join(o.name + ' (' + o.direction.value + ')' for o in objectives)}")

    return f"""You are a materials scientist specializing in COF synthesis optimization. You have
extracted {len(protocols)} synthesis protocols from the literature for the following target:

{chr(10).join(target_parts)}

HERE ARE THE CONDITIONS THAT HAVE BEEN TRIED:
{_summarize_protocols(protocols)}

YOUR TASK: Propose conditions that HAVE NOT been tried in these protocols but are chemically
reasonable and worth exploring. Think like LFAST (Yaghi group, JACS 2026) -- the goal is to
expand the design space beyond what papers report, not just interpolate within it.

For each suggestion, consider:
- Solvents: what solvent systems haven't been tried? Consider polarity, boiling point, and
  compatibility with {target.linkage_chemistry} bond formation. Mixed solvents with different
  ratios count as distinct conditions.
- Temperature: are there unexplored temperature regimes? Low-temperature crystallization
  (RT to 60°C) vs high-temperature (>150°C)?
- Modulators: what modulators/catalysts haven't been tested? Different acid strengths,
  concentrations, or altogether different modulators (Lewis acids, bases)?
- Time: short reactions (<6h) or very long ones (>7d) that haven't been explored?
- Concentration: dilute vs concentrated conditions?
- Atmosphere: has anyone tried vacuum or specific gas atmospheres?

Be specific -- "try a different solvent" is useless; "try DMF/mesitylene 4:1 v/v because DMF's
higher polarity may favor nucleation for imine COFs" is actionable.

Only propose conditions that are SAFE and chemically reasonable. Do not suggest conditions that
would decompose the monomers or are known to be incompatible with {target.linkage_chemistry} chemistry."""


def expand_design_space(
    target: Target,
    protocols: list[ProtocolCandidate],
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
) -> DesignSpaceExpansion:
    """Ask the LLM to propose untried conditions based on the extracted protocols.
    This is the LFAST steps 4-5 equivalent: "what should we try that the literature hasn't?"

    Requires an LLM client. Returns an empty expansion if client is None."""
    if not protocols:
        return DesignSpaceExpansion()

    if client is None:
        from materials_synthesis_agent.cli.llm_config import get_configured_llm_client
        client = get_configured_llm_client(model=model)

    prompt = _build_expansion_prompt(target, protocols)

    data = client.call_tool(
        prompt=prompt,
        tool_name=_EXPANSION_TOOL_NAME,
        tool_description=_EXPANSION_TOOL_DESCRIPTION,
        tool_schema=_EXPANSION_TOOL_SCHEMA,
        max_tokens=1500,
    )

    suggestions = [
        ExpansionSuggestion(
            dimension=s["dimension"],
            suggested_value=s["suggested_value"],
            rationale=s["rationale"],
        )
        for s in data.get("suggestions", [])
    ]

    logger.info("Design space expansion: %d untried conditions proposed", len(suggestions))
    return DesignSpaceExpansion(suggestions=suggestions, raw_text=prompt)


# ---------------------------------------------------------------------------
# Step 7: Cross-protocol condition reasoning (LLM-assisted)
# ---------------------------------------------------------------------------

@dataclass
class CrossProtocolAnalysis:
    consensus_conditions: list[str] = field(default_factory=list)
    variation_hotspots: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    recommended_starting_point: str = ""


_REASONING_TOOL_NAME = "cross_protocol_analysis"
_REASONING_TOOL_DESCRIPTION = (
    "Analyze patterns across multiple extracted synthesis protocols for the same material family."
)
_REASONING_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "consensus_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parameter values that appear consistently across protocols -- the community's "
            "established best practices. Each entry is one specific condition with the count of protocols "
            "that use it (e.g. 'Temperature 120°C (4/5 protocols)').",
        },
        "variation_hotspots": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parameters that vary most across protocols -- the dimensions where optimization "
            "has the most room. Each entry names the parameter and the range seen.",
        },
        "contradictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cases where successful protocols use opposing conditions. Each entry describes "
            "the contradiction and what it might mean.",
        },
        "coverage_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Conditions or parameter combinations NO protocol has tried that are worth "
            "exploring. Different from variation_hotspots: gaps are about absence, not spread.",
        },
        "recommended_starting_point": {
            "type": "string",
            "description": "Based on the literature consensus, the single best starting protocol for "
            "optimization: specific conditions (solvent, temperature, time, modulator, concentration) "
            "and why these represent the community's best current knowledge.",
        },
    },
    "required": [
        "consensus_conditions", "variation_hotspots", "contradictions",
        "coverage_gaps", "recommended_starting_point",
    ],
}


def _build_reasoning_prompt(target: Target, protocols: list[ProtocolCandidate]) -> str:
    target_parts = []
    if target.name:
        target_parts.append(f"Material: {target.name}")
    target_parts.append(f"Linkage: {target.linkage_chemistry}")
    objectives = target.all_objectives
    if objectives:
        target_parts.append(f"Optimizing: {', '.join(o.name + ' (' + o.direction.value + ')' for o in objectives)}")

    return f"""You are a materials scientist analyzing {len(protocols)} synthesis protocols extracted from
the literature for:

{chr(10).join(target_parts)}

{_summarize_protocols(protocols)}

Analyze these protocols TOGETHER and report:

1. CONSENSUS CONDITIONS: What parameter values appear most consistently? These are the community's
   established best practices for this material. Be specific with counts.

2. VARIATION HOTSPOTS: Which parameters vary most across protocols? These are the dimensions where
   optimization effort should focus -- wide variation means the community hasn't converged and there
   may be significant room for improvement.

3. CONTRADICTIONS: Any cases where protocols that both report good results use opposing conditions?
   These are scientifically interesting and suggest non-trivial interactions between parameters.

4. COVERAGE GAPS: What conditions or parameter combinations has NO protocol tried? What's the
   "dark matter" of this material's condition space?

5. RECOMMENDED STARTING POINT: Based on the literature consensus, what single set of conditions
   would you recommend as the starting point for Bayesian optimization? Be specific: state the
   exact solvent system, temperature, time, modulator (if any), and concentration. Explain why
   these conditions represent the best available prior knowledge."""


def cross_protocol_reasoning(
    target: Target,
    protocols: list[ProtocolCandidate],
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
) -> CrossProtocolAnalysis:
    """Synthesize patterns across all extracted protocols. Requires an LLM client."""
    if len(protocols) < 2:
        return CrossProtocolAnalysis(
            recommended_starting_point="Not enough protocols to synthesize patterns."
        )

    if client is None:
        from materials_synthesis_agent.cli.llm_config import get_configured_llm_client
        client = get_configured_llm_client(model=model)

    prompt = _build_reasoning_prompt(target, protocols)

    data = client.call_tool(
        prompt=prompt,
        tool_name=_REASONING_TOOL_NAME,
        tool_description=_REASONING_TOOL_DESCRIPTION,
        tool_schema=_REASONING_TOOL_SCHEMA,
        max_tokens=2000,
    )

    analysis = CrossProtocolAnalysis(
        consensus_conditions=data.get("consensus_conditions", []),
        variation_hotspots=data.get("variation_hotspots", []),
        contradictions=data.get("contradictions", []),
        coverage_gaps=data.get("coverage_gaps", []),
        recommended_starting_point=data.get("recommended_starting_point", ""),
    )

    logger.info(
        "Cross-protocol analysis: %d consensus, %d hotspots, %d contradictions, %d gaps",
        len(analysis.consensus_conditions), len(analysis.variation_hotspots),
        len(analysis.contradictions), len(analysis.coverage_gaps),
    )
    return analysis
