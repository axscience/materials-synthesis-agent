"""Citation-grounded protocol extraction via forced tool-use, provider-agnostic (Anthropic,
OpenAI, xAI/Grok, or Moonshot/Kimi -- see `llm/`).

CLAUDE.md guardrail #2: every extracted field is either citation-grounded (an `excerpt` from the
source text) or explicitly flagged `inferred=True`. The tool schema below makes that structural,
not just a prompt instruction -- the model must supply one or the other for every field it reports,
and `_field_value` refuses to build a FieldValue that has neither (matching the same validator on
`schema.models.FieldValue`).

Every call that costs money supports `dry_run=True` (CLAUDE.md convention) -- required because
materials-copilot shows this estimate to hosted users before running the job, and it's good
practice for local CLI use too.
"""

from __future__ import annotations

from typing import Optional

from materials_synthesis_agent.literature.retrieval import Paper
from materials_synthesis_agent.llm import LLMClient, PROVIDERS
from materials_synthesis_agent.llm.pricing import estimate_cost_usd
from materials_synthesis_agent.schema import Citation, FieldValue, MeasuredOutcome, ProtocolCandidate, ProtocolSource, Target

DEFAULT_MODEL = PROVIDERS["anthropic"].default_model  # kept for backwards-compatible callers

_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "description": "The extracted value, as plain text."},
        "excerpt": {
            "type": ["string", "null"],
            "description": "The exact text from the paper this value was read from. Null only if inferred=true.",
        },
        "inferred": {
            "type": "boolean",
            "description": "true if this value was reasoned/estimated rather than read directly from the text.",
        },
    },
    "required": ["value", "inferred"],
}

# Provider-agnostic: this is a plain JSON Schema, wrapped in each provider's own envelope by its
# LLMClient adapter (Anthropic's "input_schema" vs. OpenAI-style "parameters" -- see llm/).
EXTRACTION_TOOL_NAME = "record_protocol"
EXTRACTION_TOOL_DESCRIPTION = (
    "Record one candidate synthesis protocol extracted from the paper text. "
    "For each monomer, report its common abbreviation as the key (e.g. TAPB, PDA, HHTP) "
    "and its SMILES as the value. Include the synthesis method (solvothermal, mechanochemical, "
    "etc.), full solvent system, any catalyst or modulator with loading, reaction atmosphere, "
    "activation/workup, yield, and key characterization data that confirms success. "
    "Every field must carry either a direct excerpt from the text, or inferred=true if you "
    "reasoned to the value rather than reading it directly. Never provide a value with no "
    "excerpt and inferred=false."
)
EXTRACTION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "building_blocks": {
            "type": "object",
            "description": (
                "Monomer name -> SMILES string. Use the common abbreviation as the key "
                "(e.g. 'TAPB', 'PDA', 'HHTP', 'DAB'). For each, the value field is the SMILES "
                "representation. If only the name is stated and you know the corresponding "
                "SMILES, set inferred=true."
            ),
            "additionalProperties": _FIELD_SCHEMA,
        },
        "monomer_roles": {
            "type": "object",
            "description": (
                "Monomer name -> structural role. Same keys as building_blocks. "
                "Value is one of: 'linker' (ditopic or polytopic organic strut connecting nodes), "
                "'node' (vertex monomer, often a trigonal or tetrahedral amine/aldehyde), "
                "'core' (central unit in star-shaped topologies). "
                "For a typical imine COF: the triamine (e.g. TAPB) is the node, the dialdehyde "
                "(e.g. PDA) is the linker."
            ),
            "additionalProperties": _FIELD_SCHEMA,
        },
        "stoichiometry": {
            "type": "object",
            "description": (
                "Monomer name -> molar equivalents or ratio. Same keys as building_blocks. "
                "Report as stated in the paper (e.g. '2:3 molar ratio', '0.15 mmol', '35.1 mg'). "
                "Include exact masses/mmol when the paper states them — these are critical for "
                "reproducibility."
            ),
            "additionalProperties": _FIELD_SCHEMA,
        },
        "synthesis_method": {
            **_FIELD_SCHEMA,
            "description": (
                "The synthesis approach. Value should be one of: 'solvothermal' (sealed tube or "
                "autoclave at elevated temperature — the most common for COFs), 'mechanochemical' "
                "(ball milling or grinding), 'room-temperature solution', 'interfacial' "
                "(liquid-liquid or liquid-air interface polymerization), 'vapor-assisted conversion', "
                "'microwave-assisted', or describe if none of these fit. "
                + _FIELD_SCHEMA.get("description", "")
            ),
        },
        "solvent": {
            **_FIELD_SCHEMA,
            "description": (
                "The full solvent system. For mixed solvents, include the ratio "
                "(e.g. '1,4-dioxane/mesitylene 1:1 v/v', 'n-BuOH/o-DCB 1:1 with 6M AcOH'). "
                "Solvent choice is a key optimization variable for COF crystallinity."
            ),
        },
        "catalyst": {
            **_FIELD_SCHEMA,
            "description": (
                "The catalyst, if used. Common examples: Sc(OTf)₃ for imine COFs, "
                "BF₃·OEt₂ for boronate esters. Report with loading (mol%, equivalents, or mass)."
            ),
        },
        "modulator": {
            **_FIELD_SCHEMA,
            "description": (
                "The modulator or additive that controls crystallinity, distinct from the catalyst. "
                "Common examples: aqueous acetic acid (often reported as 'X M AcOH(aq)'), "
                "aniline, trifluoroacetic acid. Report with loading/concentration."
            ),
        },
        "temperature_c": _FIELD_SCHEMA,
        "time_hours": _FIELD_SCHEMA,
        "concentration_molar": {
            **_FIELD_SCHEMA,
            "description": (
                "Total monomer concentration or reaction volume. Report whichever the paper states. "
                "If the paper gives individual masses in a known volume, report the concentration."
            ),
        },
        "atmosphere": {
            **_FIELD_SCHEMA,
            "description": (
                "Reaction atmosphere if specified: N₂, Ar, vacuum, ambient/air. Many COF syntheses "
                "are run under inert atmosphere; report only if the paper states it."
            ),
        },
        "activation_method": {
            **_FIELD_SCHEMA,
            "description": (
                "How the COF was activated (guest removal) after synthesis. Common methods: "
                "Soxhlet extraction with a solvent (THF, MeOH), solvent exchange followed by "
                "vacuum drying, supercritical CO₂ drying. Activation critically affects measured "
                "surface area and porosity — report the full procedure if stated."
            ),
        },
        "purification": {
            **_FIELD_SCHEMA,
            "description": (
                "Workup/purification steps before activation: filtration, washing solvents and "
                "number of washes, Soxhlet extraction duration and solvent. Report as stated."
            ),
        },
        "yield_percent": {
            **_FIELD_SCHEMA,
            "description": "Isolated yield as a percentage, if reported.",
        },
        "characterization_notes": {
            **_FIELD_SCHEMA,
            "description": (
                "Key characterization data confirming successful synthesis. Include: "
                "PXRD (does the pattern match the simulated structure?), "
                "BET surface area in m²/g if stated, "
                "solid-state ¹³C CP-MAS NMR confirmation of linkage formation, "
                "FT-IR bands confirming bond formation (e.g. C=N stretch at ~1620 cm⁻¹ for imines), "
                "TGA decomposition onset. Summarize what was measured and the key values."
            ),
        },
        "measured_outcomes": {
            "type": "array",
            "description": (
                "Quantitative results reported for the synthesized material in the SAME paper. "
                "These are real measurements — not your estimates. Extract each as a separate "
                "entry. Common examples for COFs:\n"
                "  - PXRD crystallinity (peak area ratio, FWHM of the (100) peak)\n"
                "  - BET surface area (m²/g, from N₂ adsorption at 77 K)\n"
                "  - Isolated yield (%)\n"
                "  - Pore size distribution (nm)\n"
                "  - Thermal stability onset (°C, from TGA)\n"
                "  - CO₂ uptake, H₂ uptake, or other gas sorption (mmol/g or cm³/g at stated P/T)\n"
                "Only report values the paper explicitly states with a number. Include the "
                "measurement method so downstream systems know how to compare across papers."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Standardized name: 'crystallinity', 'BET_surface_area', "
                        "'yield', 'pore_size', 'thermal_stability', 'CO2_uptake', etc.",
                    },
                    "value": {"type": "number", "description": "The numeric value."},
                    "unit": {"type": "string", "description": "e.g. 'm²/g', '%', 'ratio', '°C', 'mmol/g'."},
                    "uncertainty": {
                        "type": ["number", "null"],
                        "description": "Stated uncertainty/error bar if reported, null otherwise.",
                    },
                    "measurement_method": {
                        "type": "string",
                        "description": "How it was measured, e.g. 'N₂ adsorption at 77 K', "
                        "'PXRD peak area ratio of (100) reflection', 'TGA under N₂'.",
                    },
                    "excerpt": {
                        "type": ["string", "null"],
                        "description": "Exact text from the paper stating this value.",
                    },
                    "inferred": {
                        "type": "boolean",
                        "description": "true only if you computed/derived this value rather than "
                        "reading it directly (e.g. converting units). Should almost always be false "
                        "for measured outcomes.",
                    },
                },
                "required": ["metric_name", "value", "unit", "measurement_method", "inferred"],
            },
        },
        "found_protocol": {
            "type": "boolean",
            "description": (
                "false if this paper does not describe an original synthesis protocol for the "
                "target material (e.g. computational study, review citing prior work, application "
                "study using a commercially obtained COF, or only referencing another paper's "
                "procedure without restating it). Do not set true for papers that merely mention "
                "the material without providing synthesis details."
            ),
        },
    },
    "required": ["found_protocol"],
}


def _build_prompt(target: Target, paper: Paper, full_text_excerpt: Optional[str] = None) -> str:
    if full_text_excerpt:
        text_block = f"""--- EXPERIMENTAL SECTION EXCERPT (from the paper's own full text) ---
{full_text_excerpt}
--- END EXCERPT ---"""
    else:
        text_block = f"""--- ABSTRACT ---
{paper.abstract or "(no abstract available)"}
--- END ABSTRACT ---"""

    target_block_parts = []
    if target.name:
        target_block_parts.append(f"- Material name: {target.name}")
    if target.functional_groups:
        target_block_parts.append(f"- Functional groups: {', '.join(target.functional_groups)}")
    target_block_parts.append(f"- Linkage chemistry: {target.linkage_chemistry}")
    if target.application:
        target_block_parts.append(f"- Target application: {target.application}")
    objectives = target.all_objectives
    if objectives:
        obj_strs = [f"{o.name} ({o.measurement_method}, {o.direction.value})" for o in objectives]
        target_block_parts.append(f"- Optimization objectives: {'; '.join(obj_strs)}")
    target_block = "\n".join(target_block_parts)

    return f"""You are a materials scientist specializing in reticular chemistry and the design, \
synthesis, and characterization of covalent organic frameworks (COFs). You are reading a research \
paper to extract a complete, reproducible synthesis protocol.

BACKGROUND
COFs are crystalline porous polymers assembled from organic monomers through reversible covalent \
bonds — imine (Schiff base, C=N), boronate ester (B-O), boroxine (B₃O₃), hydrazone (C=N-NH), \
beta-ketoenamine, triazine, and others. Their synthesis typically involves:

  1. MONOMERS: A node monomer (often trigonal, e.g. 1,3,5-tris(4-aminophenyl)benzene / TAPB, or \
hexahydroxytriphenylene / HHTP) and a linker monomer (often ditopic, e.g. terephthalaldehyde / PDA, \
or 1,4-phenylenediboronic acid / BDBA). Identifying the correct monomers and their stoichiometric \
ratio is essential — the topology (e.g. hcb, sql, kgm for 2D; dia, ctn, bor for 3D) is set by the \
monomer geometry.

  2. SYNTHESIS METHOD: Most COFs are made solvothermally (sealed Pyrex tube or autoclave, 80-200 C, \
3-7 days), but other routes include room-temperature solution, mechanochemical (ball milling), \
interfacial polymerization (at a liquid-liquid interface), microwave-assisted, and vapor-assisted \
conversion. The method choice affects crystallinity, morphology, and scalability.

  3. SOLVENT AND MODULATOR: Solvent mixtures are critical for crystallinity. Common systems include \
1,4-dioxane/mesitylene, n-BuOH/o-dichlorobenzene, and DMF/DMSO. Modulators (monoaldehyde or \
monoacid additives like aniline, acetic acid, or trifluoroacetic acid) slow the condensation to \
favor thermodynamic (crystalline) over kinetic (amorphous) product. Catalysts (e.g. Sc(OTf)₃, \
BF₃·OEt₂) may also be used.

  4. WORKUP AND ACTIVATION: Post-synthesis, COFs are typically collected by filtration, washed \
(THF, acetone, MeOH), sometimes Soxhlet-extracted, then activated (guest removal) by solvent \
exchange followed by vacuum drying or supercritical CO₂ drying. Activation is critical — \
incomplete activation collapses pores and depresses the measured BET surface area.

  5. CHARACTERIZATION: Successful synthesis is confirmed by PXRD (pattern matching the simulated \
structure), BET surface area (N₂ adsorption at 77 K), solid-state ¹³C CP-MAS NMR (confirming \
linkage bond formation, e.g. imine C=N at ~158 ppm), FT-IR (e.g. C=N stretch at ~1620 cm⁻¹), \
and TGA (thermal stability).

TARGET MATERIAL
{target_block}

PAPER
"{paper.title}" ({paper.year or "year unknown"})
{text_block}

INSTRUCTIONS
Call record_protocol to extract the synthesis protocol from this paper. Extract every detail the \
paper provides — exact masses, mmol, volumes, temperatures, durations, ramp rates. For each \
monomer, use its common abbreviation as the key (TAPB, PDA, HHTP, etc.) and report a SMILES \
string as the value. Identify each monomer's role (node vs. linker) in the monomer_roles field.

WHAT TO LOOK FOR:
- Monomers: names, structures (SMILES), amounts (mg, mmol), and their node/linker roles
- Solvent system: solvents with their ratio (e.g. "dioxane/mesitylene 1:1 v/v, 2 mL total")
- Modulator or catalyst: identity and loading (equivalents, volume, or concentration)
- Temperature and time: include ramp rates or staged heating if described
- Atmosphere: N₂, Ar, vacuum, or ambient — report if stated
- Synthesis method: solvothermal, mechanochemical, interfacial, etc.
- Workup: filtration, washing solvents, Soxhlet extraction
- Activation: solvent exchange protocol, drying conditions (temperature, vacuum level, duration)
- Yield: isolated yield percentage if reported
- Characterization: PXRD match quality, BET surface area (m²/g), NMR/IR confirmation of linkage

MEASURED OUTCOMES (critical — these seed the optimization model):
Extract every quantitative characterization result as a measured_outcomes entry. The optimizer \
uses these as real data points to build its surrogate model, so precision matters:
- BET surface area: the number in m²/g, with measurement method (N₂ at 77 K)
- PXRD crystallinity: any quantitative metric (peak area ratio, FWHM) — not just "matches well"
- Yield: the percentage, if an isolated yield is reported
- Pore size: from NLDFT/BJH analysis, in nm
- Gas uptake: CO₂, H₂, N₂ uptake at stated conditions (mmol/g or cm³/g)
- Thermal stability: decomposition onset from TGA, in °C
If the paper only says "PXRD matches the simulated pattern" without a quantitative metric, \
do NOT invent a number — report it in characterization_notes instead.

CITATION DISCIPLINE
Every value you report must carry either:
- An `excerpt`: the exact words from the paper (copy verbatim, do not paraphrase)
- Or `inferred: true`: you reasoned to the value (e.g. inferring SMILES from a known monomer name)
Never report a value with no excerpt and inferred=false.

If the paper does NOT describe an original synthesis protocol — it is a computational study, a \
review, an application study using pre-made material, or it only cites another paper's procedure \
without restating the conditions — set found_protocol=false and leave other fields empty. \
Do not fill fields from general knowledge when the paper itself does not provide the information."""


def estimate_extraction_cost(
    target: Target, paper: Paper, model: Optional[str] = None, full_text_excerpt: Optional[str] = None
) -> float:
    """Rough pre-call cost estimate in USD, for the confirm-before-spending UI. Token count is
    approximated at ~4 chars/token -- adequate for a pre-call estimate, not billing-accurate.
    `model=None` resolves to whichever model the configured provider will actually use (see
    cli/llm_config.get_configured_model) -- not always Anthropic's default, now that there are
    four providers. Pass the same `full_text_excerpt` you'll pass to `extract_protocol` -- a
    full-text prompt is real input tokens, not a rounding error against an abstract-only estimate."""
    from materials_synthesis_agent.cli.llm_config import get_configured_model

    prompt = _build_prompt(target, paper, full_text_excerpt=full_text_excerpt)
    output_tokens = 500  # a generous estimate for a filled-out protocol tool call
    resolved_model = get_configured_model(model)
    return estimate_cost_usd(prompt, output_tokens, resolved_model)


def _field_value(raw: Optional[dict], paper: Paper) -> Optional[FieldValue]:
    if raw is None:
        return None
    inferred = bool(raw.get("inferred", False))
    excerpt = raw.get("excerpt")
    citation = None if inferred and not excerpt else Citation(source_id=paper.source_id, title=paper.title, excerpt=excerpt)
    return FieldValue(value=raw["value"], citation=citation, inferred=inferred and not excerpt)


def extract_protocol(
    target: Target,
    paper: Paper,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
    full_text_excerpt: Optional[str] = None,
) -> ProtocolCandidate | float:
    """Extract a ProtocolCandidate from one paper. Returns the estimated cost (float, USD) if
    dry_run=True, without making a call. `client` is dependency-injected so this is testable with
    a fake client that never hits the network -- see tests/test_extraction.py. If `client` is
    None, one is built from the configured provider (see cli/llm_config.py).

    `full_text_excerpt`, if given (see literature/fulltext.py), is used instead of `paper.abstract`
    -- this function makes no network calls itself and does no OA/PDF resolution; that's the
    caller's job (literature/agent.py), keeping this module's only responsibility "build a prompt
    from whatever text I'm handed, call the LLM, parse the result.\""""
    if dry_run:
        return estimate_extraction_cost(target, paper, model=model, full_text_excerpt=full_text_excerpt)

    if client is None:
        from materials_synthesis_agent.cli.llm_config import get_configured_llm_client

        client = get_configured_llm_client(model=model)

    data = client.call_tool(
        prompt=_build_prompt(target, paper, full_text_excerpt=full_text_excerpt),
        tool_name=EXTRACTION_TOOL_NAME,
        tool_description=EXTRACTION_TOOL_DESCRIPTION,
        tool_schema=EXTRACTION_TOOL_SCHEMA,
        max_tokens=2000,
    )

    if not data.get("found_protocol", False):
        return None

    def _dict_fields(key: str) -> dict[str, FieldValue]:
        return {name: fv for name, raw in (data.get(key) or {}).items() if (fv := _field_value(raw, paper))}

    def _parse_outcomes(raw_list: list[dict] | None) -> list[MeasuredOutcome]:
        if not raw_list:
            return []
        outcomes = []
        for raw in raw_list:
            excerpt = raw.get("excerpt")
            inferred = bool(raw.get("inferred", False))
            citation = None if inferred and not excerpt else Citation(
                source_id=paper.source_id, title=paper.title, excerpt=excerpt,
            )
            outcomes.append(MeasuredOutcome(
                metric_name=raw["metric_name"],
                value=float(raw["value"]),
                unit=raw["unit"],
                uncertainty=float(raw["uncertainty"]) if raw.get("uncertainty") is not None else None,
                measurement_method=raw["measurement_method"],
                citation=citation,
                inferred=inferred and not excerpt,
            ))
        return outcomes

    return ProtocolCandidate(
        target_id=target.id,
        source=ProtocolSource.LITERATURE,
        building_blocks=_dict_fields("building_blocks"),
        monomer_roles=_dict_fields("monomer_roles"),
        stoichiometry=_dict_fields("stoichiometry"),
        synthesis_method=_field_value(data.get("synthesis_method"), paper),
        solvent=_field_value(data.get("solvent"), paper),
        catalyst=_field_value(data.get("catalyst"), paper),
        modulator=_field_value(data.get("modulator"), paper),
        temperature_c=_field_value(data.get("temperature_c"), paper),
        time_hours=_field_value(data.get("time_hours"), paper),
        concentration_molar=_field_value(data.get("concentration_molar"), paper),
        atmosphere=_field_value(data.get("atmosphere"), paper),
        activation_method=_field_value(data.get("activation_method"), paper),
        purification=_field_value(data.get("purification"), paper),
        yield_percent=_field_value(data.get("yield_percent"), paper),
        characterization_notes=_field_value(data.get("characterization_notes"), paper),
        measured_outcomes=_parse_outcomes(data.get("measured_outcomes")),
    )
