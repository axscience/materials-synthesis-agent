"""Auth seam -- resolve the caller's tenant from a Supabase JWT, or a dev fallback.

This is the boundary a Lovable/Supabase front end connects through: the browser authenticates with
Supabase, then sends Supabase's access token as `Authorization: Bearer <jwt>` on every call to this
API. Here we verify that token and turn it into a `Tenant` the rest of the request is scoped to, so
one user's campaigns never touch another's (workspace isolation in api.py keys off `Tenant.id`).

Two modes, chosen by the `AUTH_MODE` env var:

  * `AUTH_MODE=supabase`  -- verify the Bearer JWT and use its `sub` claim as the tenant id. Works
      with both Supabase key types: an asymmetric project key via JWKS (set `SUPABASE_URL` or
      `SUPABASE_JWKS_URL`), or the legacy shared HS256 secret (`SUPABASE_JWT_SECRET`).
  * `AUTH_MODE=disabled` (default) -- no token required; everything runs as a single shared tenant
      (`local`). This keeps local dev, tests, and the pre-multi-user single-user deployment working
      unchanged. In this mode ONLY, an `X-Tenant-Id` header may override the tenant, so per-tenant
      isolation can be exercised without wiring Supabase. The header is ignored under `supabase`
      mode -- there, identity comes from the verified token and nothing else.

PyJWT (with the `crypto` extra) is required only for `supabase` mode; it is imported lazily so the
default install and `disabled` mode need no new dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException

_DEV_TENANT = "local"


@dataclass(frozen=True)
class Tenant:
    """The isolation key for a request. `id` scopes storage; `email` is for display/audit only."""
    id: str
    email: Optional[str] = None


def _auth_mode() -> str:
    return os.environ.get("AUTH_MODE", "disabled").strip().lower()


def _jwks_url() -> Optional[str]:
    explicit = os.environ.get("SUPABASE_JWKS_URL")
    if explicit:
        return explicit
    base = os.environ.get("SUPABASE_URL")
    if base:
        return base.rstrip("/") + "/auth/v1/.well-known/jwks.json"
    return None


def _bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token.")
    return authorization.split(" ", 1)[1].strip()


def _verify_supabase(token: str) -> Tenant:
    try:
        import jwt  # PyJWT
        from jwt import PyJWKClient
    except ModuleNotFoundError as exc:  # pragma: no cover - env guard
        raise HTTPException(
            500, "AUTH_MODE=supabase requires PyJWT: pip install 'PyJWT[crypto]'."
        ) from exc

    # Supabase tokens carry aud='authenticated'. We verify signature + expiry + audience.
    options = {"require": ["exp", "sub"]}
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    jwks_url = _jwks_url()
    try:
        if jwks_url:
            signing_key = PyJWKClient(jwks_url).get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token, signing_key, algorithms=["RS256", "ES256"],
                audience="authenticated", options=options,
            )
        elif secret:
            claims = jwt.decode(
                token, secret, algorithms=["HS256"],
                audience="authenticated", options=options,
            )
        else:
            raise HTTPException(
                500, "AUTH_MODE=supabase but neither SUPABASE_URL/SUPABASE_JWKS_URL nor "
                "SUPABASE_JWT_SECRET is set.",
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - any verification failure is a 401
        raise HTTPException(401, f"Invalid token: {exc}") from exc

    sub = claims.get("sub")
    if not sub:
        raise HTTPException(401, "Token has no subject (sub) claim.")
    return Tenant(id=str(sub), email=claims.get("email"))


def current_tenant(
    authorization: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None),
) -> Tenant:
    """FastAPI dependency: the tenant this request is scoped to. Raises 401 in supabase mode when the
    token is missing or invalid; returns the shared/dev tenant in disabled mode."""
    if _auth_mode() == "supabase":
        return _verify_supabase(_bearer(authorization))
    # disabled mode: single shared tenant, with an optional dev override for testing isolation.
    return Tenant(id=(x_tenant_id or _DEV_TENANT))
