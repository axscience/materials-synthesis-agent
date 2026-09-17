"""Cross-campaign warm-start prior -- the one learning feature in v1, built as what it actually is:
a SQL aggregate over past experiments, not a knowledge graph.

Every logged experiment is recorded in a shared `priors.db`, indexed by linkage chemistry + metric.
When a new campaign targets a chemistry the user has worked before, `build_warm_prior` returns the
best-performing past conditions as optimizer anchors, so campaign 5 starts where campaigns 1-4 left
off instead of cold. The optimizer already accepts `literature_anchors` to bias where its search
starts -- the warm prior feeds that same seam with the user's real prior results.

Deliberately torch-free: it takes plain param requirements, not a ParameterSpace, so the
optimizer's heavy import stays out of this module and its tests. The handler bridges the two.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prior_experiments (
    id                TEXT PRIMARY KEY,
    linkage_chemistry TEXT NOT NULL,
    metric_name       TEXT NOT NULL,
    params            TEXT NOT NULL,   -- JSON {param: value}
    metric_value      REAL NOT NULL,
    campaign_id       TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prior_lookup ON prior_experiments(linkage_chemistry, metric_name);
"""


class GaussianPrior(BaseModel):
    mean: float
    std: float
    n_observations: int


class WarmPrior(BaseModel):
    linkage_chemistry: str
    metric_name: str
    n_prior_experiments: int = 0
    # Best-performing past conditions, filtered to the target's parameter space. Feed these to the
    # optimizer via `as_literature_anchors`.
    anchor_params: list[dict[str, Any]] = []
    # Per-parameter aggregate -- the documented extension point for a GP prior-mean function. Anchors
    # are enough for v1; these are computed and returned for when the optimizer grows a prior mean.
    parameter_priors: dict[str, GaussianPrior] = {}
    source_campaigns: list[str] = []

    @property
    def is_empty(self) -> bool:
        return self.n_prior_experiments == 0


class PriorStore:
    """The shared cross-campaign experiment index. One file for all campaigns."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PriorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def record(
        self,
        linkage_chemistry: str,
        metric_name: str,
        params: dict[str, Any],
        metric_value: float,
        campaign_id: Optional[str] = None,
    ) -> None:
        """Called after each experiment is logged -- this is the flywheel feed."""
        self._conn.execute(
            "INSERT INTO prior_experiments "
            "(id, linkage_chemistry, metric_name, params, metric_value, campaign_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                linkage_chemistry,
                metric_name,
                json.dumps(params),
                float(metric_value),
                campaign_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def _rows(self, linkage_chemistry: str, metric_name: str) -> list[tuple[dict, float, str]]:
        rows = self._conn.execute(
            "SELECT params, metric_value, campaign_id FROM prior_experiments "
            "WHERE linkage_chemistry = ? AND metric_name = ?",
            (linkage_chemistry, metric_name),
        ).fetchall()
        return [(json.loads(p), v, c) for p, v, c in rows]


def _params_cover_space(
    params: dict[str, Any],
    required_params: Optional[set[str]],
    categorical_values: Optional[dict[str, set]],
) -> bool:
    """An anchor is usable only if it supplies every parameter the optimizer's space needs, with
    valid categorical values -- otherwise `space.encode` would fail on it."""
    if required_params is not None:
        if not required_params.issubset(params.keys()):
            return False
    if categorical_values:
        for name, allowed in categorical_values.items():
            if name in params and params[name] not in allowed:
                return False
    return True


def build_warm_prior(
    store: PriorStore,
    linkage_chemistry: str,
    metric_name: str,
    maximize: bool = True,
    required_params: Optional[set[str]] = None,
    categorical_values: Optional[dict[str, set]] = None,
    limit: int = 5,
) -> WarmPrior:
    """Aggregate past experiments for this chemistry/metric into anchors + per-parameter priors.

    `required_params` / `categorical_values` come from the target's ParameterSpace (derived cheaply
    by the caller, so this module never imports the optimizer). Rows that don't cover the space are
    excluded from anchors but still count toward the per-parameter priors."""
    rows = store._rows(linkage_chemistry, metric_name)
    if not rows:
        return WarmPrior(linkage_chemistry=linkage_chemistry, metric_name=metric_name)

    # Best-performing conditions first, filtered to what the space can encode.
    ranked = sorted(rows, key=lambda r: r[1], reverse=maximize)
    anchors: list[dict[str, Any]] = []
    sources: list[str] = []
    for params, _value, campaign_id in ranked:
        if _params_cover_space(params, required_params, categorical_values):
            anchors.append(params)
            if campaign_id:
                sources.append(campaign_id)
        if len(anchors) >= limit:
            break

    # Per-parameter Gaussian priors over the numeric parameters seen (all rows, not just anchors).
    numeric: dict[str, list[float]] = {}
    for params, _value, _cid in rows:
        for name, val in params.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric.setdefault(name, []).append(float(val))
    priors: dict[str, GaussianPrior] = {}
    for name, vals in numeric.items():
        mean = sum(vals) / len(vals)
        if len(vals) >= 2:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        priors[name] = GaussianPrior(mean=mean, std=std, n_observations=len(vals))

    return WarmPrior(
        linkage_chemistry=linkage_chemistry,
        metric_name=metric_name,
        n_prior_experiments=len(rows),
        anchor_params=anchors,
        parameter_priors=priors,
        source_campaigns=sorted(set(sources)),
    )


def as_literature_anchors(warm_prior: WarmPrior) -> list:
    """Convert to optimizer LiteratureAnchors. Lazy import so this module stays torch-free."""
    from materials_synthesis_agent.optimize import LiteratureAnchor

    return [LiteratureAnchor(params=p) for p in warm_prior.anchor_params]
