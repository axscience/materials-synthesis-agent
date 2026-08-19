"""Free-text request parsing: the researcher's own words in ("here's a CIF of a COF, tell me how
to synthesize it to maximize crystallinity via PXRD peak ratio"), a structured ParsedRequest out.

Same forced-tool-use discipline as literature/extraction.py (CLAUDE.md guardrail #2, extended from
"grounded in a paper" to "grounded in what the user typed"): the model must call one tool, and the
tool schema requires it to separate what the text actually stated from what it filled in by
inference (`inferred_fields`) or could not determine at all (`clarifications_needed` -- a question
handed back to the user, never a silent guess at a metric, application, or objective direction
nobody mentioned).

A `cif_path` the model reports is a string it read out of prose, not a verified fact -- `parse_request`
checks the path actually exists before trusting it, same principle as CLAUDE.md guardrail #4 (no
dynamic execution of model output): the model's output drives what gets *asked* of the local
filesystem, never gets executed or trusted blindly.

Every call that costs money supports `dry_run=True` (CLAUDE.md convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from materials_synthesis_agent.llm import LLMClient
from materials_synthesis_agent.llm.pricing import estimate_cost_usd
from materials_synthesis_agent.schema import ObjectiveDirection, ParsedRequest, TargetObjective

PARSE_TOOL_NAME = "record_parsed_request"
PARSE_TOOL_DESCRIPTION = (
    "Record what you understood from the researcher's free-text request about a covalent organic "
    "framework (COF) synthesis. Only fill in a field if the text actually states it, directly or by "
    "an unambiguous synonym. If you filled in a field by inference or a reasonable default rather "
    "than reading it directly, still fill it in, but add its name to inferred_fields. Never invent "
    "a target application, metric, or optimization direction the text doesn't mention -- put that "
    "under clarifications_needed instead, as a specific question for the researcher."
)
PARSE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "cof_name": {
            "type": ["string", "null"],
            "description": "A specific named COF if one is mentioned, e.g. 'COF-5' or 'TAPB-PDA COF'. Null if the request doesn't name one.",
        },
        "cif_path": {
            "type": ["string", "null"],
            "description": "A file path or filename ending in .cif that the text references. Null if none is mentioned.",
        },
        "functional_groups": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Functional groups explicitly named (e.g. 'imine', 'boronate ester'). Empty if none are stated.",
        },
        "linkage_chemistry": {
            "type": ["string", "null"],
            "description": "The linkage chemistry if named (e.g. 'imine condensation'). Null if not stated.",
        },
        "application": {
            "type": ["string", "null"],
            "description": "The stated target application, e.g. 'CO2 capture'. Null if not stated.",
        },
        "objectives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The metric to optimize, e.g. 'crystallinity'."},
                    "measurement_method": {"type": "string", "description": "How it's measured, e.g. 'PXRD peak area ratio'."},
                    "direction": {"type": "string", "enum": ["maximize", "minimize"]},
                },
                "required": ["name", "measurement_method", "direction"],
            },
            "description": "Metrics to optimize, in the order mentioned. Empty if the text doesn't state what to optimize -- do not invent one.",
        },
        "inferred_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of fields above you filled in by inference/default rather than reading directly from the text, e.g. 'objectives[0].direction'.",
        },
        "clarifications_needed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Plain-language questions about anything essential that's missing or ambiguous, which you did NOT guess at -- e.g. 'What should I optimize for, and how would you measure it?' if no metric was mentioned at all.",
        },
    },
    "required": ["functional_groups", "objectives", "inferred_fields", "clarifications_needed"],
}


def _build_prompt(text: str) -> str:
    return f"""A researcher is describing, in their own words, what they want from a covalent
organic framework (COF) synthesis-planning tool.

Request: "{text}"

Call {PARSE_TOOL_NAME} with what you can determine. Read the field descriptions carefully --
particularly: never invent a target application, metric, or optimization direction the request
doesn't state; put anything you had to guess under inferred_fields, and anything missing that you
did NOT guess under clarifications_needed instead."""


def estimate_parse_cost(text: str, model: Optional[str] = None) -> float:
    """Rough pre-call cost estimate in USD, for the confirm-before-spending UI. `model=None`
    resolves to whichever model the configured provider will actually use."""
    from materials_synthesis_agent.cli.llm_config import get_configured_model

    prompt = _build_prompt(text)
    output_tokens = 400  # a generous estimate for a filled-out parsed-request tool call
    resolved_model = get_configured_model(model)
    return estimate_cost_usd(prompt, output_tokens, resolved_model)


def parse_request(
    text: str,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
) -> ParsedRequest | float:
    """Parse a free-text request into a ParsedRequest. Returns the estimated cost (float, USD) if
    dry_run=True, without making a call. `client` is dependency-injected so this is testable with a
    fake client that never hits the network -- see tests/test_nl_parser.py. If `client` is None,
    one is built from the configured provider (see cli/llm_config.py)."""
    if dry_run:
        return estimate_parse_cost(text, model=model)

    if client is None:
        from materials_synthesis_agent.cli.llm_config import get_configured_llm_client

        client = get_configured_llm_client(model=model)

    data = client.call_tool(
        prompt=_build_prompt(text),
        tool_name=PARSE_TOOL_NAME,
        tool_description=PARSE_TOOL_DESCRIPTION,
        tool_schema=PARSE_TOOL_SCHEMA,
        max_tokens=1000,
    )

    objectives = [
        TargetObjective(
            name=o["name"], measurement_method=o["measurement_method"], direction=ObjectiveDirection(o["direction"])
        )
        for o in (data.get("objectives") or [])
    ]

    clarifications = list(data.get("clarifications_needed") or [])
    raw_cif_path = data.get("cif_path")
    cif_path: Optional[str] = None
    if raw_cif_path:
        if Path(raw_cif_path).exists():
            cif_path = raw_cif_path
        else:
            # A path the model read out of prose is not a verified fact -- surface the mismatch as
            # a clarification rather than silently dropping it or trusting it downstream.
            clarifications.append(f"I couldn't find a file at '{raw_cif_path}' -- check the path and try again.")

    return ParsedRequest(
        raw_text=text,
        cof_name=data.get("cof_name"),
        cif_path=cif_path,
        functional_groups=list(data.get("functional_groups") or []),
        linkage_chemistry=data.get("linkage_chemistry"),
        application=data.get("application"),
        objectives=objectives,
        inferred_fields=list(data.get("inferred_fields") or []),
        clarifications_needed=clarifications,
    )
