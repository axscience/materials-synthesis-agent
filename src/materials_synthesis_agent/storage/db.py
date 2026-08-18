"""Local SQLite persistence. Zero setup by design (CLAUDE.md convention): a fresh clone should
work the moment someone runs `materials-agent init`, without standing up a database server.

Schema objects are stored as JSON blobs keyed by id, mirroring the Pydantic models in
`schema.models` -- this module has no independent notion of "what a protocol is," it just persists
and retrieves the schema objects verbatim. That keeps the schema module the single source of truth,
matching how `materials-copilot`'s Postgres layer is documented to consume the exact same models.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Type, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS targets (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS protocol_candidates (
    id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    protocol_candidate_id TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bo_suggestions (
    id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    followed_from TEXT,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    estimated_cost REAL NOT NULL,
    actual_cost REAL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""


class Store:
    """One SQLite file per local project. Not thread-safe across processes by design -- this is a
    single-user local tool, not a server."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- generic helpers -------------------------------------------------

    def _insert(self, table: str, id_: str, extra_cols: dict, obj: BaseModel) -> None:
        cols = ["id", *extra_cols.keys(), "data", "created_at"]
        placeholders = ",".join("?" for _ in cols)
        values = [id_, *extra_cols.values(), obj.model_dump_json(), _iso_now()]
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def _get(self, table: str, id_: str, model: Type[ModelT]) -> Optional[ModelT]:
        row = self._conn.execute(f"SELECT data FROM {table} WHERE id = ?", (id_,)).fetchone()
        return model.model_validate_json(row[0]) if row else None

    def _list(self, table: str, model: Type[ModelT], where: str = "", params: tuple = ()) -> list[ModelT]:
        query = f"SELECT data FROM {table}"
        if where:
            query += f" WHERE {where}"
        rows = self._conn.execute(query, params).fetchall()
        return [model.model_validate_json(r[0]) for r in rows]

    # -- typed entry points -----------------------------------------------
    # Kept thin and explicit rather than fully generic, so each entity's identifying columns
    # (target_id, protocol_candidate_id, etc.) stay queryable without a JSON scan.

    def save_target(self, target) -> None:
        self._insert("targets", target.id, {}, target)

    def get_target(self, id_: str):
        from materials_synthesis_agent.schema import Target

        return self._get("targets", id_, Target)

    def save_protocol_candidate(self, candidate) -> None:
        self._insert(
            "protocol_candidates",
            candidate.id,
            {"target_id": candidate.target_id, "version": candidate.version},
            candidate,
        )

    def list_protocol_candidates(self, target_id: str):
        from materials_synthesis_agent.schema import ProtocolCandidate

        return self._list(
            "protocol_candidates", ProtocolCandidate, "target_id = ?", (target_id,)
        )

    def save_experiment(self, experiment) -> None:
        self._insert(
            "experiments",
            experiment.id,
            {"protocol_candidate_id": experiment.protocol_candidate_id},
            experiment,
        )

    def list_experiments(self, target_id: str):
        """Experiments for a target, joined through its protocol candidates."""
        from materials_synthesis_agent.schema import Experiment

        candidate_ids = {c.id for c in self.list_protocol_candidates(target_id)}
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self._conn.execute(
            f"SELECT data FROM experiments WHERE protocol_candidate_id IN ({placeholders})",
            tuple(candidate_ids),
        ).fetchall()
        return [Experiment.model_validate_json(r[0]) for r in rows]

    def save_bo_suggestion(self, suggestion) -> None:
        self._insert("bo_suggestions", suggestion.id, {"target_id": suggestion.target_id}, suggestion)

    def list_bo_suggestions(self, target_id: str):
        from materials_synthesis_agent.schema import BOSuggestion

        return self._list("bo_suggestions", BOSuggestion, "target_id = ?", (target_id,))

    def save_decision(self, decision) -> None:
        self._insert(
            "decisions", decision.id, {"followed_from": decision.followed_from}, decision
        )

    def latest_decision(self):
        """Most recent decision, for chaining `followed_from` -- mirrors the walkable-trajectory
        pattern from chat-to-lab's episodic decision log."""
        from materials_synthesis_agent.schema import Decision

        row = self._conn.execute(
            "SELECT data FROM decisions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return Decision.model_validate_json(row[0]) if row else None

    def reconstruct_trajectory(self):
        """Walk the followed_from chain from the earliest decision forward."""
        from materials_synthesis_agent.schema import Decision

        rows = self._conn.execute("SELECT data FROM decisions").fetchall()
        by_id = {}
        for r in rows:
            d = Decision.model_validate_json(r[0])
            by_id[d.id] = d
        children_of = {}
        roots = []
        for d in by_id.values():
            if d.followed_from and d.followed_from in by_id:
                children_of[d.followed_from] = d
            elif not d.followed_from:
                roots.append(d)
        trajectory = []
        for root in sorted(roots, key=lambda d: d.created_at):
            node = root
            while node is not None:
                trajectory.append(node)
                node = children_of.get(node.id)
        return trajectory

    # -- usage / cost governance ------------------------------------------
    # Mirrors materials-copilot's usage_events table (DATA_MODEL.md) so the same idempotency
    # discipline applies locally, not just in the hosted product.

    def record_usage(
        self, kind: str, estimated_cost: float, idempotency_key: str, actual_cost: Optional[float] = None
    ) -> bool:
        """Returns False (no-op) if this idempotency_key was already recorded -- the caller should
        treat that as "already charged, don't call the API again."""
        try:
            self._conn.execute(
                "INSERT INTO usage_events (id, kind, estimated_cost, actual_cost, idempotency_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_new_uuid(), kind, estimated_cost, actual_cost, idempotency_key, _iso_now()),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def total_usage_cost(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(COALESCE(actual_cost, estimated_cost)), 0) FROM usage_events"
        ).fetchone()
        return float(row[0])


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    import uuid

    return str(uuid.uuid4())
