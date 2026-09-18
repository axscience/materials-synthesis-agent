# Connecting a Lovable front end to the Discovery Harness API

This is the contract a Lovable/Supabase front end drives. **Lovable builds and publishes the front
end; this Python service is the compute backend it calls over HTTPS.** Supabase provides auth + user
data; this API runs the science (literature extraction, GP+BO, the planner loop).

```
Lovable/React (published on Lovable)
   │  Supabase Auth  ─────────────►  Supabase (Postgres + Auth)   [users, orgs — Lovable owns this]
   │  Bearer <supabase jwt>
   ▼
Discovery Harness API (this repo, deployed on Fly/Railway/Render)
   FastAPI + GP/BO + extractor.  Verifies the JWT → scopes everything to that user (tenant).
```

## 1. Deploy this backend (get its public URL)

A `Dockerfile` is included. On Fly/Railway/Render, `PORT` is injected and the server binds `0.0.0.0`
automatically. Set these environment variables on the service:

| Env var | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | the planner + extractor models |
| `AUTH_MODE` | for multi-user | `supabase` to require login; `disabled` (default) = single shared tenant |
| `SUPABASE_URL` | if `AUTH_MODE=supabase` | your Supabase project URL (JWKS is derived from it) |
| `SUPABASE_JWT_SECRET` | alt to `SUPABASE_URL` | legacy HS256 secret, if you use the shared-secret key type |
| `HARNESS_CORS_ORIGINS` | recommended | comma-separated allowed origins, e.g. `https://yourapp.lovable.app` (default `*`) |
| `HARNESS_WORKSPACE` | recommended | data dir; mount a volume so tenant data survives redeploys (Docker default `/data`) |
| `MATERIALS_AGENT_PLANNER_MODEL` | optional | override the planner model id |
| `MATERIALS_AGENT_EXTRACT_MODEL` | optional | override the extractor model id |

After deploy, confirm: `GET https://<your-backend>/api/health` → `{"status":"ok","auth_mode":"..."}`.

## 2. Auth (how the JWT flows)

- `AUTH_MODE=disabled` (default): no token needed; everything is one shared tenant `local`. Good for
  a first-connect demo. In this mode only, an `X-Tenant-Id` header can simulate separate tenants.
- `AUTH_MODE=supabase`: every `/api/*` call (except `/api/health`) must send
  `Authorization: Bearer <supabase access token>`. The token's `sub` (Supabase user id) becomes the
  tenant; each user sees only their own campaigns. No token / bad token → `401`.

In Lovable, after Supabase login, attach the access token to every backend request:
`headers: { Authorization: 'Bearer ' + session.access_token }`.

## 3. Endpoints

Base URL = your deployed backend. All paths are under `/api`. All bodies are JSON.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{status, auth_mode}` (no auth) |
| GET | `/api/campaigns` | — | `[{id, name, status, updated_at}]` |
| POST | `/api/campaigns` | `CreateCampaign` | `{id, name}` |
| GET | `/api/campaigns/{id}` | — | `{id, name, status, linkage_chemistry, space, target, session_ids}` |
| GET | `/api/campaigns/{id}/state` | — | `{protocols[], experiments[], suggestions[], characterizations[]}` |
| POST | `/api/campaigns/{id}/chat` | `{message, session_id?}` | `{job_id, session_id, status}` (async — poll the job) |
| GET | `/api/campaigns/{id}/jobs/{job_id}` | — | `{status, reply, tool_calls[], error, session_id}` |
| GET | `/api/campaigns/{id}/jobs` | — | `[{job_id, status, kind, message, created_at}]` |
| POST | `/api/campaigns/{id}/log` | `LogResult` | `{result, gate}` |
| GET | `/api/campaigns/{id}/sessions` | — | `[{id, started_at, turns, cost}]` |
| GET | `/api/campaigns/{id}/sessions/{sid}/export` | — | markdown (text/markdown) |

**CreateCampaign**
```json
{
  "name": "COF-LZU1 BET",
  "application": "gas storage",
  "linkage_chemistry": "imine",
  "functional_groups": ["amine", "aldehyde"],
  "metric_name": "bet_surface_area",
  "metric_measurement_method": "N2 physisorption"
}
```
`name` is required; the rest are optional but, when `metric_name` + `linkage_chemistry` are given, a
target is created so the loop can start immediately.

**Chat (async)** — this drives the whole loop. The planner may run for minutes (extraction), so the
call **enqueues a job and returns immediately**:
```json
{ "job_id": "…", "session_id": "…", "status": "queued" }
```
Then **poll** `GET /api/campaigns/{id}/jobs/{job_id}` until `status` is `done` (or `error`):
```json
{ "status": "done", "reply": "…assistant text…",
  "tool_calls": [{"name": "opt.suggest_next", "…": "…"}], "error": null, "session_id": "…" }
```
Poll every ~2s; show `tool_calls` as step chips while running. Pass the returned `session_id` back on
the next message to continue the conversation. `error` is populated (and `status="error"`) if the run
failed — show it to the user.

**LogResult** — record a bench measurement so the next suggestion improves:
```json
{
  "protocol_id": "…",
  "metrics": [
    {"name": "bet_surface_area", "value": 1980, "uncertainty": 60, "measurement_method": "N2 physisorption"}
  ],
  "notes": "optional"
}
```
Returns `{result, gate}`; `gate.blocked=true` with a message when a check fails (e.g. calibration).

## 4. Supabase setup (once)

1. Create a Supabase project; note the **Project URL** and **anon key** (Lovable uses these).
2. Enable email (or OAuth) auth.
3. On this backend set `AUTH_MODE=supabase` and `SUPABASE_URL=<project url>`.
4. Set `HARNESS_CORS_ORIGINS` to your Lovable app's origin.

(Campaign/experiment records currently live in this backend's per-tenant store. Making Supabase the
system-of-record with RLS is the Phase-4 Postgres port — tracked separately; not required to connect.)

## 5. Paste-into-Lovable prompt

> Build a chat-based research console for an AI materials-synthesis copilot. Auth with Supabase.
> After login, call my backend API at `https://<your-backend>/api`, sending
> `Authorization: Bearer <supabase access token>` on every request. Screens:
> (1) a sidebar listing campaigns (`GET /api/campaigns`) with a "New campaign" form
> (`POST /api/campaigns`); (2) a chat view per campaign: `POST /api/campaigns/{id}/chat` returns a
> `job_id`; poll `GET /api/campaigns/{id}/jobs/{job_id}` every 2s until `status` is `done`, then show
> `reply` and render each `tool_calls` entry as a step chip (spinner while `queued`/`running`, show
> `error` if it fails); (3) a "results" panel reading `GET /api/campaigns/{id}/state` (protocols,
> experiments, suggestions) and a "log a bench result" form (`POST /api/campaigns/{id}/log`). Show
> `gate.blocked` messages as warnings. Keep it clean and functional, not flashy.

## Scaling note

Chat/extraction runs **asynchronously** (Phase 4, P4.1): the request enqueues a job and the front end
polls, so a multi-minute extraction never times out. The default runner is an in-process thread pool
— correct for a single container. When you outgrow one container, swap in a Redis/arq runner (it
implements the same `submit` interface; nothing else changes). Jobs are stored per tenant alongside
the campaign, so a poll always finds its result.
