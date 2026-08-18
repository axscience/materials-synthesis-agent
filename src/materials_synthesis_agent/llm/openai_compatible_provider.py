"""One adapter for OpenAI, xAI (Grok), and Moonshot AI (Kimi) -- all three expose the same
OpenAI-compatible chat-completions + tool-calling wire format, differing only in base_url and
model name (see llm/base.py's PROVIDERS registry). Confirmed directly: xAI's docs describe an
OpenAI-compatible endpoint at api.x.ai/v1, and Moonshot's docs describe the same at
api.moonshot.ai/v1, both explicitly supporting tool calling.

NOT verified against real xAI/Kimi API calls in this codebase's development environment -- only
against real OpenAI-shaped requests via the `openai` SDK's own client construction. `max_tokens`
(not the newer OpenAI-specific `max_completion_tokens`) is used deliberately here for the widest
compatibility across third-party OpenAI-compatible surfaces; switch per-provider if one of them
requires otherwise.
"""

from __future__ import annotations

import json
from typing import Optional

import openai


class OpenAICompatibleProvider:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.6", base_url: Optional[str] = None):
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def call_tool(
        self, prompt: str, tool_name: str, tool_description: str, tool_schema: dict, max_tokens: int = 2000
    ) -> dict:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[
                {
                    "type": "function",
                    "function": {"name": tool_name, "description": tool_description, "parameters": tool_schema},
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
