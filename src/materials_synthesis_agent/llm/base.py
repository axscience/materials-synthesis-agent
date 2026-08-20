"""Provider-agnostic LLM client interface.

Two capabilities:
  1. `call_tool` — force a single named tool call and get back its parsed arguments (used for
     citation-grounded protocol extraction and NL parsing).
  2. `chat` — free-form conversational reasoning (used by the expert agent for planning,
     advising, and orchestration).

Keeping the interface this narrow is what makes supporting four providers tractable instead of
four bespoke integrations.

Real, current (as of this writing) provider details below -- base URLs and default models were
looked up directly, not guessed. Model names in this space change often; override via `configure`
or the MATERIALS_AGENT_LLM_MODEL env var if a default here goes stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderConfig:
    key: str
    display_name: str
    env_var: str
    base_url: str | None  # None = provider's native SDK default endpoint (Anthropic, OpenAI)
    default_model: str


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        key="anthropic", display_name="Anthropic (Claude)", env_var="ANTHROPIC_API_KEY",
        base_url=None, default_model="claude-opus-4-5",
    ),
    "openai": ProviderConfig(
        key="openai", display_name="OpenAI (GPT)", env_var="OPENAI_API_KEY",
        base_url=None, default_model="gpt-5.6",
    ),
    "grok": ProviderConfig(
        key="grok", display_name="xAI (Grok)", env_var="GROK_API_KEY",
        base_url="https://api.x.ai/v1", default_model="grok-4.3",
    ),
    "kimi": ProviderConfig(
        key="kimi", display_name="Moonshot AI (Kimi)", env_var="KIMI_API_KEY",
        base_url="https://api.moonshot.ai/v1", default_model="kimi-k3",
    ),
}


class LLMClient(Protocol):
    def call_tool(
        self, prompt: str, tool_name: str, tool_description: str, tool_schema: dict, max_tokens: int = 2000
    ) -> dict:
        """Force the model to call `tool_name` and return its parsed arguments as a dict.
        `tool_schema` is a plain JSON Schema object (the same shape for every provider -- each
        adapter wraps it in its own envelope internally)."""
        ...

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4000,
    ) -> str:
        """Free-form conversational exchange. `messages` is a list of
        {"role": "user"|"assistant", "content": "..."} dicts. Returns the assistant's
        text response. Used by the expert agent for reasoning, planning, and advising."""
        ...


def build_client(provider_key: str, api_key: str, model: str | None = None) -> LLMClient:
    """Factory: instantiate the right adapter for a provider key. `model` overrides the provider's
    default (see ProviderConfig.default_model)."""
    if provider_key not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider_key}'. Choose one of: {', '.join(PROVIDERS)}")
    config = PROVIDERS[provider_key]
    resolved_model = model or config.default_model

    if provider_key == "anthropic":
        from materials_synthesis_agent.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=resolved_model)

    from materials_synthesis_agent.llm.openai_compatible_provider import OpenAICompatibleProvider

    return OpenAICompatibleProvider(api_key=api_key, model=resolved_model, base_url=config.base_url)
