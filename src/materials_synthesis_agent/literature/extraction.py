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
from materials_synthesis_agent.schema import Citation, FieldValue, ProtocolCandidate, ProtocolSource, Target

DEFAULT_MODEL = PROVIDERS["anthropic"].default_model  # kept for backwards-compatible callers

# Rough, approximate per-million-token pricing for cost estimation -- versioned data (CLAUDE.md
# convention: prices are never a silent constant baked into logic), meant to be updated as pricing
# changes. This is an ESTIMATE for the pre-call cost display, not a source of billing truth. Looked
# up directly, not guessed -- but this space moves fast, so treat these as approximate.
_PRICING_PER_MTOK_USD = {
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "gpt-5.6": {"input": 5.00, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "grok-4.3": {"input": 1.25, "output": 2.50},
    "kimi-k3": {"input": 3.00, "output": 15.00},
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}

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
    "Record one candidate synthesis protocol extracted from the paper text. Every field must "
    "carry either a direct excerpt from the text, or inferred=true if you reasoned to the value "
    "rather than reading it directly. Never provide a value with no excerpt and inferred=false."
)
EXTRACTION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "building_blocks": {
            "type": "object",
            "description": "name -> field (value is a SMILES string)",
            "additionalProperties": _FIELD_SCHEMA,
        },
        "stoichiometry": {
            "type": "object",
            "description": "building block name -> field (value is an equivalents/ratio string)",
            "additionalProperties": _FIELD_SCHEMA,
        },
        "solvent": _FIELD_SCHEMA,
        "modulator": _FIELD_SCHEMA,
        "temperature_c": _FIELD_SCHEMA,
        "time_hours": _FIELD_SCHEMA,
        "concentration_molar": _FIELD_SCHEMA,
        "found_protocol": {
            "type": "boolean",
            "description": "false if this paper does not actually describe a synthesis protocol for the target.",
        },
    },
    "required": ["found_protocol"],
}


def _build_prompt(target: Target, paper: Paper) -> str:
    return f"""You are extracting a covalent organic framework (or related material) synthesis
protocol from a research paper, for a researcher targeting:

- Functional groups: {", ".join(target.functional_groups)}
- Linkage chemistry: {target.linkage_chemistry}
- Application: {target.application}

Paper: "{paper.title}" ({paper.year or "year unknown"})
Abstract/text:
{paper.abstract or "(no abstract available)"}

Call record_protocol with what you can extract. If the abstract doesn't describe a synthesis
protocol in enough detail, set found_protocol=false and leave other fields empty rather than
guessing. Every field you do report needs either a direct excerpt or inferred=true -- never both
missing."""


def estimate_extraction_cost(target: Target, paper: Paper, model: Optional[str] = None) -> float:
    """Rough pre-call cost estimate in USD, for the confirm-before-spending UI. Token count is
    approximated at ~4 chars/token -- adequate for a pre-call estimate, not billing-accurate.
    `model=None` resolves to whichever model the configured provider will actually use (see
    cli/llm_config.get_configured_model) -- not always Anthropic's default, now that there are
    four providers."""
    from materials_synthesis_agent.cli.llm_config import get_configured_model

    prompt = _build_prompt(target, paper)
    input_tokens = max(len(prompt) // 4, 1)
    output_tokens = 500  # a generous estimate for a filled-out protocol tool call
    resolved_model = get_configured_model(model)
    pricing = _PRICING_PER_MTOK_USD.get(resolved_model, _DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


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
) -> ProtocolCandidate | float:
    """Extract a ProtocolCandidate from one paper. Returns the estimated cost (float, USD) if
    dry_run=True, without making a call. `client` is dependency-injected so this is testable with
    a fake client that never hits the network -- see tests/test_extraction.py. If `client` is
    None, one is built from the configured provider (see cli/llm_config.py)."""
    if dry_run:
        return estimate_extraction_cost(target, paper, model=model)

    if client is None:
        from materials_synthesis_agent.cli.llm_config import get_configured_llm_client

        client = get_configured_llm_client(model=model)

    data = client.call_tool(
        prompt=_build_prompt(target, paper),
        tool_name=EXTRACTION_TOOL_NAME,
        tool_description=EXTRACTION_TOOL_DESCRIPTION,
        tool_schema=EXTRACTION_TOOL_SCHEMA,
        max_tokens=2000,
    )

    if not data.get("found_protocol", False):
        return None  # caller should treat None as "this paper wasn't useful," not an error

    building_blocks = {
        name: fv for name, raw in (data.get("building_blocks") or {}).items() if (fv := _field_value(raw, paper))
    }
    stoichiometry = {
        name: fv for name, raw in (data.get("stoichiometry") or {}).items() if (fv := _field_value(raw, paper))
    }

    return ProtocolCandidate(
        target_id=target.id,
        source=ProtocolSource.LITERATURE,
        building_blocks=building_blocks,
        stoichiometry=stoichiometry,
        solvent=_field_value(data.get("solvent"), paper),
        modulator=_field_value(data.get("modulator"), paper),
        temperature_c=_field_value(data.get("temperature_c"), paper),
        time_hours=_field_value(data.get("time_hours"), paper),
        concentration_molar=_field_value(data.get("concentration_molar"), paper),
    )
