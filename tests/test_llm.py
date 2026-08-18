"""Tests for the provider-agnostic LLM abstraction (llm/) -- the piece that lets suggest-protocols
work against Anthropic, OpenAI, xAI/Grok, or Moonshot/Kimi. Uses fakes at the SDK-client boundary
(monkeypatching the real `anthropic`/`openai` client objects' request methods), not the network --
no real provider was called to write these.
"""

import json
from types import SimpleNamespace

import pytest

from materials_synthesis_agent.llm import PROVIDERS, build_client
from materials_synthesis_agent.llm.anthropic_provider import AnthropicProvider
from materials_synthesis_agent.llm.openai_compatible_provider import OpenAICompatibleProvider

SCHEMA = {"type": "object", "properties": {"found_protocol": {"type": "boolean"}}}


def test_providers_registry_has_all_four():
    assert set(PROVIDERS) == {"anthropic", "openai", "grok", "kimi"}
    for key, config in PROVIDERS.items():
        assert config.key == key
        assert config.env_var
        assert config.default_model


def test_build_client_rejects_unknown_provider():
    with pytest.raises(ValueError):
        build_client("not_a_real_provider", api_key="x")


def test_build_client_anthropic_uses_native_provider():
    client = build_client("anthropic", api_key="fake", model="claude-opus-4-5")
    assert isinstance(client, AnthropicProvider)
    assert client.model == "claude-opus-4-5"


@pytest.mark.parametrize("provider_key", ["openai", "grok", "kimi"])
def test_build_client_uses_openai_compatible_adapter_for_the_other_three(provider_key):
    client = build_client(provider_key, api_key="fake")
    assert isinstance(client, OpenAICompatibleProvider)
    assert client.model == PROVIDERS[provider_key].default_model


def test_grok_and_kimi_point_at_their_own_base_url_not_openais():
    grok_client = build_client("grok", api_key="fake")
    kimi_client = build_client("kimi", api_key="fake")
    assert str(grok_client._client.base_url).startswith("https://api.x.ai")
    assert str(kimi_client._client.base_url).startswith("https://api.moonshot.ai")


def test_build_client_explicit_model_overrides_provider_default():
    client = build_client("openai", api_key="fake", model="gpt-5.6-luna")
    assert client.model == "gpt-5.6-luna"


def test_anthropic_provider_call_tool_parses_tool_use_block(monkeypatch):
    provider = AnthropicProvider(api_key="fake-key")

    class FakeToolUseBlock:
        type = "tool_use"
        input = {"found_protocol": True}

    def fake_create(**kwargs):
        assert kwargs["tools"] == [{"name": "record_protocol", "description": "desc", "input_schema": SCHEMA}]
        assert kwargs["tool_choice"] == {"type": "tool", "name": "record_protocol"}
        return SimpleNamespace(content=[FakeToolUseBlock()])

    monkeypatch.setattr(provider._client.messages, "create", fake_create)
    result = provider.call_tool("a prompt", "record_protocol", "desc", SCHEMA)
    assert result == {"found_protocol": True}


def test_openai_compatible_provider_call_tool_parses_function_call_arguments(monkeypatch):
    provider = OpenAICompatibleProvider(api_key="fake-key", model="gpt-5.6")
    fake_args_json = json.dumps({"found_protocol": True})
    fake_tool_call = SimpleNamespace(function=SimpleNamespace(arguments=fake_args_json))
    fake_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[fake_tool_call]))])

    def fake_create(**kwargs):
        assert kwargs["tools"] == [
            {"type": "function", "function": {"name": "record_protocol", "description": "desc", "parameters": SCHEMA}}
        ]
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "record_protocol"}}
        return fake_response

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    result = provider.call_tool("a prompt", "record_protocol", "desc", SCHEMA)
    assert result == {"found_protocol": True}
