"""Anthropic's native tool-use format. See llm/base.py for why this is the only provider that
doesn't go through the OpenAI-compatible adapter -- Anthropic's wire format is genuinely different
(input_schema at the top level, tool_use content blocks in the response), not just a different
base_url."""

from __future__ import annotations

from typing import Optional

import anthropic


class AnthropicProvider:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def call_tool(
        self, prompt: str, tool_name: str, tool_description: str, tool_schema: dict, max_tokens: int = 2000
    ) -> dict:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[{"name": tool_name, "description": tool_description, "input_schema": tool_schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_use = next(block for block in response.content if block.type == "tool_use")
        return tool_use.input
