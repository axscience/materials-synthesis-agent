"""Local LLM provider configuration: which of the 4 providers (llm.PROVIDERS) this installation
uses, and which model. Stored in ~/.materials-agent/config.json alongside the retrosynthesis
config (cli/project.py) -- it's a property of this installation, not of any one project.

The API key is written to that file with chmod 600 (owner read/write only) and is never
transmitted anywhere except in the literal request to the provider you chose. Env vars always win
over the stored config, so a power user can override per-shell without touching the file --
ANTHROPIC_API_KEY, OPENAI_API_KEY, GROK_API_KEY, KIMI_API_KEY.
"""

from __future__ import annotations

import json
import os
import stat
from typing import Optional

from materials_synthesis_agent.cli.project import _read_global_config, global_config_dir, global_config_path
from materials_synthesis_agent.llm import PROVIDERS, LLMClient, build_client


def save_llm_config(provider: str, api_key: str, model: Optional[str] = None) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    global_config_dir().mkdir(parents=True, exist_ok=True)
    data = _read_global_config()
    data["llm_provider"] = provider
    data["llm_api_key"] = api_key
    if model:
        data["llm_model"] = model
    else:
        data.pop("llm_model", None)
    path = global_config_path()
    path.write_text(json.dumps(data, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600 -- this file holds a secret, nothing else can read it


def get_configured_provider() -> Optional[str]:
    """Best-effort provider lookup that does NOT require a key to be present -- used for cost
    estimation, where we need to know which pricing table to consult, not to actually call
    anything. Checks env vars first (any provider's key present -> that provider wins), then the
    stored config. Returns None if nothing is configured anywhere."""
    for key, config in PROVIDERS.items():
        if os.environ.get(config.env_var):
            return key
    return _read_global_config().get("llm_provider")


def get_configured_model(explicit_model: Optional[str] = None) -> str:
    """Resolve which model name to use or price against: explicit override > a model saved by
    `configure` > the configured provider's own default > Anthropic's default as a last-resort
    fallback for estimation when nothing is configured yet."""
    if explicit_model:
        return explicit_model
    stored_model = _read_global_config().get("llm_model")
    if stored_model:
        return stored_model
    provider = get_configured_provider()
    if provider in PROVIDERS:
        return PROVIDERS[provider].default_model
    return PROVIDERS["anthropic"].default_model


def _resolve_provider_and_key() -> tuple[str, str]:
    for key, config in PROVIDERS.items():
        value = os.environ.get(config.env_var)
        if value:
            return key, value

    data = _read_global_config()
    provider = data.get("llm_provider")
    api_key = data.get("llm_api_key")
    if provider and api_key:
        return provider, api_key

    raise RuntimeError(
        "No LLM provider configured. Run `materials-agent configure`, or set one of: "
        + ", ".join(c.env_var for c in PROVIDERS.values())
    )


def get_configured_llm_client(model: Optional[str] = None) -> LLMClient:
    """Builds a real LLMClient from whatever's configured (env var or `configure`). Raises
    RuntimeError with actionable instructions if nothing is set up -- never silently falls back
    to a default provider a user didn't choose."""
    provider, api_key = _resolve_provider_and_key()
    resolved_model = model or _read_global_config().get("llm_model")
    return build_client(provider, api_key, model=resolved_model)
