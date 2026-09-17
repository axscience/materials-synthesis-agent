"""Campaign persistence -- extends the existing `storage.Store` with campaign, session, and
characterization tables in the same SQLite file.

By subclassing Store, a single connection holds both the loop data (targets/protocols/experiments/
suggestions/decisions) and the campaign-level data, so one campaign is one portable `.db` file and
the loop data stays queryable through the parent's typed entry points -- no second source of truth.
"""

from __future__ import annotations

from pathlib import Path

from materials_synthesis_agent.harness.campaign import (
    Campaign,
    CharacterizationResult,
    Session,
)
from materials_synthesis_agent.storage.db import Store

_CAMPAIGN_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    data         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    campaign_id  TEXT NOT NULL,
    data         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS characterizations (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_campaign ON sessions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_char_campaign ON characterizations(campaign_id);
"""


class CampaignStore(Store):
    def __init__(self, path: Path | str):
        super().__init__(path)
        self._conn.executescript(_CAMPAIGN_SCHEMA)
        self._conn.commit()

    # -- campaigns --------------------------------------------------------

    def save_campaign(self, campaign: Campaign) -> None:
        campaign.touch()
        self._insert("campaigns", campaign.id, {"status": campaign.status.value}, campaign)

    def get_campaign(self, id_: str) -> Campaign | None:
        return self._get("campaigns", id_, Campaign)

    def list_campaigns(self) -> list[Campaign]:
        # Newest first, so the sidebar shows the most recently touched project on top.
        rows = self._conn.execute(
            "SELECT data FROM campaigns ORDER BY created_at DESC"
        ).fetchall()
        campaigns = [Campaign.model_validate_json(r[0]) for r in rows]
        return sorted(campaigns, key=lambda c: c.updated_at, reverse=True)

    # -- sessions ---------------------------------------------------------

    def save_session(self, session: Session) -> None:
        self._insert("sessions", session.id, {"campaign_id": session.campaign_id}, session)

    def get_session(self, id_: str) -> Session | None:
        return self._get("sessions", id_, Session)

    def list_sessions(self, campaign_id: str) -> list[Session]:
        sessions = self._list("sessions", Session, "campaign_id = ?", (campaign_id,))
        return sorted(sessions, key=lambda s: s.started_at)

    # -- characterizations ------------------------------------------------

    def save_characterization(self, campaign_id: str, char: CharacterizationResult) -> None:
        self._insert(
            "characterizations",
            char.id,
            {"campaign_id": campaign_id, "experiment_id": char.experiment_id},
            char,
        )

    def list_characterizations(self, campaign_id: str) -> list[CharacterizationResult]:
        return self._list(
            "characterizations", CharacterizationResult, "campaign_id = ?", (campaign_id,)
        )
