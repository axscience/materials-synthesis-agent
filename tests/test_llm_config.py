"""Tests for cli/llm_config.py's provider/key resolution: env vars must win over the stored
config, and nothing should silently fall back to a provider the user didn't choose."""

import pytest

from materials_synthesis_agent.cli import llm_config
from materials_synthesis_agent.llm import PROVIDERS


@pytest.fixture(autouse=True)
def isolated_global_config(tmp_path, monkeypatch):
    """Point the global config at a throwaway dir and strip every provider env var, so these
    tests can't accidentally read this machine's real config or environment."""
    monkeypatch.setattr("materials_synthesis_agent.cli.project.global_config_dir", lambda: tmp_path)
    for config in PROVIDERS.values():
        monkeypatch.delenv(config.env_var, raising=False)
    yield tmp_path


def test_nothing_configured_returns_none_provider():
    assert llm_config.get_configured_provider() is None


def test_nothing_configured_raises_a_clear_error_when_building_a_client():
    with pytest.raises(RuntimeError, match="materials-agent configure"):
        llm_config.get_configured_llm_client()


def test_env_var_is_detected_as_the_configured_provider(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "fake-grok-key")
    assert llm_config.get_configured_provider() == "grok"


def test_env_var_wins_over_a_stored_config(monkeypatch):
    llm_config.save_llm_config("anthropic", "fake-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    provider, key = llm_config._resolve_provider_and_key()
    assert provider == "openai"
    assert key == "fake-openai-key"


def test_stored_config_used_when_no_env_var_set():
    llm_config.save_llm_config("kimi", "fake-kimi-key")
    provider, key = llm_config._resolve_provider_and_key()
    assert provider == "kimi"
    assert key == "fake-kimi-key"


def test_save_llm_config_rejects_unknown_provider():
    with pytest.raises(ValueError):
        llm_config.save_llm_config("not_a_real_provider", "some-key")


def test_saved_config_file_is_only_readable_by_owner(isolated_global_config):
    llm_config.save_llm_config("anthropic", "fake-key")
    path = isolated_global_config / "config.json"
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_get_configured_model_prefers_explicit_over_stored_over_default():
    llm_config.save_llm_config("openai", "fake-key")  # no model saved -> provider default
    assert llm_config.get_configured_model() == PROVIDERS["openai"].default_model
    assert llm_config.get_configured_model("gpt-5.6-luna") == "gpt-5.6-luna"

    llm_config.save_llm_config("openai", "fake-key", model="gpt-5.6-terra")
    assert llm_config.get_configured_model() == "gpt-5.6-terra"
    assert llm_config.get_configured_model("gpt-5.6-luna") == "gpt-5.6-luna"  # explicit still wins


def test_get_configured_llm_client_builds_a_real_client_from_stored_config():
    llm_config.save_llm_config("anthropic", "fake-key")
    client = llm_config.get_configured_llm_client()
    assert client.model == PROVIDERS["anthropic"].default_model
