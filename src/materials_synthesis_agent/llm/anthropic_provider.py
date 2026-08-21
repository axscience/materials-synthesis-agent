"""Anthropic's native tool-use format. See llm/base.py for why this is the only provider that
doesn't go through the OpenAI-compatible adapter -- Anthropic's wire format is genuinely different
(input_schema at the top level, tool_use content blocks in the response), not just a different
base_url."""

from __future__ import annotations

from typing import Optional

import anthropic


class AnthropicProvider:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5", base_url: Optional[str] = None):
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model

    def call_tool(
        self, prompt: str, tool_name: str, tool_description: str, tool_schema: dict, max_tokens: int = 4000
    ) -> dict:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[{"name": tool_name, "description": tool_description, "input_schema": tool_schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            raise ValueError(f"Model did not return a tool_use block (stop_reason={response.stop_reason})")
        return tool_use.input

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text")
