"""Local project config: where a project's SQLite store and parameter-space definition live on
disk. A project is just a directory -- `<name>/agent.db` (Store) + `<name>/parameter_space.json`.

The parameter space is defined explicitly by the user, not inferred from literature-extracted text.
Free-text protocol fields ("value": "120", "excerpt": "heated at 120 C") are not automatically
trustworthy as BO input types/bounds -- silently guessing them would be exactly the kind of
unfounded inference CLAUDE.md guardrail #2 exists to prevent for citations, and the same principle
applies here: the user states the space, the tool doesn't guess it.
"""

from __future__ import annotations

import json
from pathlib import Path

from materials_synthesis_agent.optimize.space import ParameterSpec


def project_dir(name: str) -> Path:
    return Path.cwd() / name


def db_path(name: str) -> Path:
    return project_dir(name) / "agent.db"


def parameter_space_path(name: str) -> Path:
    return project_dir(name) / "parameter_space.json"


def target_path(name: str) -> Path:
    return project_dir(name) / "target_id.txt"


EXAMPLE_PARAMETER_SPACE = [
    {"name": "temperature_c", "kind": "continuous", "bounds": [20, 150]},
    {"name": "time_hours", "kind": "continuous", "bounds": [1, 96]},
    {"name": "solvent", "kind": "categorical", "categories": ["dioxane", "mesitylene", "DMAc"]},
]


def write_example_parameter_space(name: str) -> Path:
    path = parameter_space_path(name)
    path.write_text(json.dumps(EXAMPLE_PARAMETER_SPACE, indent=2))
    return path


def load_parameter_space(name: str) -> list[ParameterSpec]:
    path = parameter_space_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No parameter_space.json in '{name}/' -- edit the example written by `init` "
            f"to match this target's real synthesis parameters before running suggest-next."
        )
    raw = json.loads(path.read_text())
    specs = []
    for item in raw:
        bounds = tuple(item["bounds"]) if "bounds" in item else None
        categories = tuple(item["categories"]) if "categories" in item else None
        specs.append(ParameterSpec(name=item["name"], kind=item["kind"], bounds=bounds, categories=categories))
    return specs


# -- Global (cross-project) config -------------------------------------------------------------
# Retrosynthesis model/stock data (~759 MB) is shared across every local project -- there's no
# reason to redownload it per project, so its location lives outside any single project directory.

def global_config_dir() -> Path:
    return Path.home() / ".materials-agent"


def global_config_path() -> Path:
    return global_config_dir() / "config.json"


def default_retrosynthesis_data_dir() -> Path:
    return global_config_dir() / "retrosynthesis-data"


def default_structure_database_data_dir() -> Path:
    return global_config_dir() / "structure-database"


def _read_global_config() -> dict:
    path = global_config_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def set_global_retrosynthesis_config(config_yml_path: str) -> None:
    global_config_dir().mkdir(parents=True, exist_ok=True)
    data = _read_global_config()
    data["retrosynthesis_config"] = config_yml_path
    global_config_path().write_text(json.dumps(data, indent=2))


def get_global_retrosynthesis_config() -> str | None:
    """Returns the configured retrosynthesis config.yml path, or None if setup-retrosynthesis
    hasn't been run -- callers treat None as "retrosynthesis not available," not an error."""
    return _read_global_config().get("retrosynthesis_config")


def set_global_structure_database_index(index_path: str) -> None:
    global_config_dir().mkdir(parents=True, exist_ok=True)
    data = _read_global_config()
    data["structure_database_index"] = index_path
    global_config_path().write_text(json.dumps(data, indent=2))


def get_global_structure_database_index() -> str | None:
    """Returns the built CURATED-COFs fingerprint index path, or None if
    setup-structure-database hasn't been run -- callers treat None as "no local structure
    database," not an error."""
    return _read_global_config().get("structure_database_index")
