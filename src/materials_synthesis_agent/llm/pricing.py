"""Shared per-million-token pricing table for pre-call cost estimation.

Versioned data (CLAUDE.md convention: prices are never a silent constant baked into logic), meant
to be updated as pricing changes. This is an ESTIMATE for the pre-call cost display, not a source
of billing truth. Looked up directly, not guessed -- but this space moves fast, so treat these as
approximate. Shared by every module that estimates an LLM call's cost before running it
(literature/extraction.py, nl/parser.py) so the numbers can't drift between them.
"""

from __future__ import annotations

PRICING_PER_MTOK_USD = {
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "gpt-5.6": {"input": 5.00, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "grok-4.3": {"input": 1.25, "output": 2.50},
    "kimi-k3": {"input": 3.00, "output": 15.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


def estimate_cost_usd(prompt: str, output_tokens: int, model: str) -> float:
    """~4 chars/token approximation -- adequate for a pre-call estimate, not billing-accurate."""
    input_tokens = max(len(prompt) // 4, 1)
    pricing = PRICING_PER_MTOK_USD.get(model, DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
