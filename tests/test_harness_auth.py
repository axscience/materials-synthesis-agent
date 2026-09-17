"""Auth seam + per-tenant isolation for the harness API.

Covers the three behaviors a Lovable/Supabase front end depends on: (1) disabled mode (default)
needs no token, (2) two tenants never see each other's campaigns, and (3) supabase mode rejects a
missing or invalid token.
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from materials_synthesis_agent.harness.api import create_app  # noqa: E402
from materials_synthesis_agent.harness.planner import PlannerResult  # noqa: E402


class _TextOnlyCaller:
    def run(self, system, messages, tools, dispatch, max_turns=12):
        return PlannerResult(final_text="ok", tool_calls=[], turns_used=1)


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "ws", caller_factory=lambda: _TextOnlyCaller()))


def _mk(client, name, headers=None):
    r = client.post("/api/campaigns", json={"name": name}, headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_health_reports_auth_mode(client, monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["auth_mode"] == "disabled"


def test_disabled_mode_needs_no_token(client, monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)
    _mk(client, "c1")
    assert [c["name"] for c in client.get("/api/campaigns").json()] == ["c1"]


def test_tenants_are_isolated(client, monkeypatch):
    """Two tenants (dev override header) must not see each other's campaigns."""
    monkeypatch.delenv("AUTH_MODE", raising=False)
    _mk(client, "alice-cof", headers={"X-Tenant-Id": "alice"})
    _mk(client, "bob-mof", headers={"X-Tenant-Id": "bob"})

    alice = client.get("/api/campaigns", headers={"X-Tenant-Id": "alice"}).json()
    bob = client.get("/api/campaigns", headers={"X-Tenant-Id": "bob"}).json()
    assert [c["name"] for c in alice] == ["alice-cof"]
    assert [c["name"] for c in bob] == ["bob-mof"]

    # Alice cannot open Bob's campaign id (it lives under Bob's tenant subtree only).
    bob_id = bob[0]["id"]
    assert client.get(f"/api/campaigns/{bob_id}", headers={"X-Tenant-Id": "alice"}).status_code == 404


def test_supabase_mode_rejects_missing_and_bad_tokens(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "supabase")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")
    # no token
    assert client.get("/api/campaigns").status_code == 401
    # garbage token
    assert client.get("/api/campaigns",
                      headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401


def test_supabase_mode_accepts_a_valid_hs256_token(client, monkeypatch):
    jwt = pytest.importorskip("jwt")
    monkeypatch.setenv("AUTH_MODE", "supabase")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_JWKS_URL", raising=False)
    token = jwt.encode(
        {"sub": "user-123", "email": "a@b.co", "aud": "authenticated", "exp": 9999999999},
        "test-secret", algorithm="HS256",
    )
    h = {"Authorization": f"Bearer {token}"}
    _mk(client, "authed-campaign", headers=h)
    names = [c["name"] for c in client.get("/api/campaigns", headers=h).json()]
    assert names == ["authed-campaign"]
    # A different subject sees nothing of user-123's.
    other = jwt.encode({"sub": "user-999", "aud": "authenticated", "exp": 9999999999},
                       "test-secret", algorithm="HS256")
    assert client.get("/api/campaigns", headers={"Authorization": f"Bearer {other}"}).json() == []
