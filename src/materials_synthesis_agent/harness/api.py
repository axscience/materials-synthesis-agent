"""FastAPI backend for the Discovery Harness chat platform.

Local-first and single-user (no auth -- that belongs in the SaaS wrapper, not here). The chat
endpoint runs the planner tool-loop; campaigns, sessions, the Markdown export, and a direct
log-result form back the UI's sidebar, review, and measure screens.

`create_app(workspace, caller_factory)` is a factory so tests can inject a fake ToolCaller instead
of a live LLM -- the same discipline the rest of the codebase uses for provider calls.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from materials_synthesis_agent.harness.auth import Tenant, current_tenant
from materials_synthesis_agent.harness.campaign import Campaign, Session
from materials_synthesis_agent.harness.jobs import (
    Job,
    JobRunner,
    JobStatus,
    ThreadJobRunner,
)
from materials_synthesis_agent.harness.planner import Planner, ToolCaller
from materials_synthesis_agent.harness.session_export import session_to_markdown
from materials_synthesis_agent.harness.store import CampaignStore
from materials_synthesis_agent.harness.tools import ToolContext, run_tool
from materials_synthesis_agent.harness.warm_prior import PriorStore


class CreateCampaignIn(BaseModel):
    name: str
    # Optional target -- if given, a Target is created and linked so the loop can start immediately.
    application: Optional[str] = None
    linkage_chemistry: Optional[str] = None
    functional_groups: list[str] = []
    metric_name: Optional[str] = None
    metric_measurement_method: Optional[str] = None


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


class LogResultIn(BaseModel):
    protocol_id: str
    metrics: list[dict]
    notes: str = ""


def _default_caller_factory() -> ToolCaller:
    from materials_synthesis_agent.harness.planner import AnthropicToolCaller

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(503, "ANTHROPIC_API_KEY not set -- the planner needs a model to run.")
    model = os.environ.get("MATERIALS_AGENT_PLANNER_MODEL", "claude-opus-4-5")
    return AnthropicToolCaller(api_key=key, model=model)


def create_app(
    workspace: Path | str,
    caller_factory: Optional[Callable[[], ToolCaller]] = None,
    runner: Optional[JobRunner] = None,
) -> FastAPI:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    caller_factory = caller_factory or _default_caller_factory
    # ThreadJobRunner in production; tests inject InlineJobRunner so a job is done on submit.
    runner = runner or ThreadJobRunner()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        runner.shutdown()

    app = FastAPI(title="Discovery Harness", lifespan=_lifespan)

    # A Lovable/Supabase front end is served from a different origin, so it needs CORS. Lock this to
    # your front end's origin(s) in production via HARNESS_CORS_ORIGINS (comma-separated); the "*"
    # default is convenient for first-connect but should not ship to real users.
    origins_env = os.environ.get("HARNESS_CORS_ORIGINS", "*").strip()
    origins = ["*"] if origins_env == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],  # cannot use credentials with a wildcard origin
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _tenant_root(tenant_id: str) -> Path:
        # Per-tenant isolation on the current SQLite storage: every tenant gets its own subtree, so
        # one user's campaigns/priors/index never touch another's. (The Phase-4 Postgres port
        # replaces this file layout with RLS, but the isolation boundary -- tenant_id -- is the same.)
        root = workspace / "tenants" / tenant_id
        (root / "campaigns").mkdir(parents=True, exist_ok=True)
        return root

    def _store(tenant_id: str, campaign_id: str) -> CampaignStore:
        return CampaignStore(_tenant_root(tenant_id) / "campaigns" / campaign_id / "campaign.db")

    def _priors(tenant_id: str) -> PriorStore:
        # Per-tenant warm prior: cross-campaign learning stays within one tenant, never across.
        return PriorStore(_tenant_root(tenant_id) / "priors.db")

    def _index(tenant_id: str) -> CampaignStore:
        # A small per-tenant index DB listing that tenant's campaigns (rows duplicated here for the
        # sidebar; the authoritative copy lives in each campaign's own db).
        return CampaignStore(_tenant_root(tenant_id) / "index.db")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "auth_mode": os.environ.get("AUTH_MODE", "disabled")}

    # -- campaigns --------------------------------------------------------

    @app.get("/api/campaigns")
    def list_campaigns(tenant: Tenant = Depends(current_tenant)):
        idx = _index(tenant.id)
        out = [{"id": c.id, "name": c.name, "status": c.status.value,
                "updated_at": c.updated_at.isoformat()} for c in idx.list_campaigns()]
        idx.close()
        return out

    @app.post("/api/campaigns")
    def create_campaign(body: CreateCampaignIn, tenant: Tenant = Depends(current_tenant)):
        campaign = Campaign(name=body.name, linkage_chemistry=body.linkage_chemistry)
        store = _store(tenant.id, campaign.id)
        if body.metric_name and body.linkage_chemistry:
            from materials_synthesis_agent.schema import Target

            target = Target(
                name=body.name, functional_groups=body.functional_groups or ["unspecified"],
                linkage_chemistry=body.linkage_chemistry,
                application=body.application or "unspecified",
                metric_name=body.metric_name,
                metric_measurement_method=body.metric_measurement_method or "unspecified",
            )
            store.save_target(target)
            campaign.target_id = target.id
        store.save_campaign(campaign)
        store.close()
        idx = _index(tenant.id); idx.save_campaign(campaign); idx.close()
        return {"id": campaign.id, "name": campaign.name}

    @app.get("/api/campaigns/{campaign_id}")
    def get_campaign(campaign_id: str, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        c = store.get_campaign(campaign_id)
        if c is None:
            store.close()
            raise HTTPException(404, "No such campaign.")
        target = store.get_target(c.target_id) if c.target_id else None
        result = {
            "id": c.id, "name": c.name, "status": c.status.value,
            "linkage_chemistry": c.linkage_chemistry,
            "space": c.space, "target": target.model_dump() if target else None,
            "session_ids": c.session_ids,
        }
        store.close()
        return result

    # -- state (backs the review + log screens) --------------------------

    @app.get("/api/campaigns/{campaign_id}/state")
    def campaign_state(campaign_id: str, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        c = store.get_campaign(campaign_id)
        if c is None or not c.target_id:
            store.close()
            return {"protocols": [], "experiments": [], "suggestions": [], "characterizations": []}
        tid = c.target_id
        protocols = [p.model_dump() for p in store.list_protocol_candidates(tid)]
        experiments = [e.model_dump() for e in store.list_experiments(tid)]
        suggestions = [s.model_dump() for s in store.list_bo_suggestions(tid)]
        chars = [ch.model_dump() for ch in store.list_characterizations(campaign_id)]
        store.close()
        return {"protocols": protocols, "experiments": experiments,
                "suggestions": suggestions, "characterizations": chars}

    # -- chat -------------------------------------------------------------

    def _run_chat_job(tenant_id: str, campaign_id: str, job_id: str,
                      session_id: str, message: str) -> None:
        """Worker body (runs in the job runner's thread). Opens its OWN stores -- SQLite connections
        are per-thread -- runs the planner loop (which persists the session), and writes the reply or
        the error onto the job row. Never raises: a failure becomes an ERROR job the client can read."""
        store = _store(tenant_id, campaign_id)
        priors = _priors(tenant_id)
        try:
            job = store.get_job(job_id)
            campaign = store.get_campaign(campaign_id)
            if job is None or campaign is None:
                return
            job.status = JobStatus.RUNNING
            store.save_job(job)

            session = store.get_session(session_id) or Session(id=session_id, campaign_id=campaign_id)
            # The Extractor role calls an LLM when a key is set; a fine-tuned extractor slots in via
            # the same build_client seam later.
            llm = None
            key = os.environ.get("ANTHROPIC_API_KEY")
            extract_model = os.environ.get("MATERIALS_AGENT_EXTRACT_MODEL", "claude-sonnet-4-6")
            if key:
                from materials_synthesis_agent.llm import build_client

                llm = build_client("anthropic", key, extract_model)
            ctx = ToolContext(
                store=store, prior_store=priors, campaign=campaign,
                llm=llm, model=extract_model,
                unpaywall_email=os.environ.get("UNPAYWALL_EMAIL"),
            )
            planner = Planner(ctx, caller_factory())
            reply = planner.respond(message, session)
            last = session.turns[-1] if session.turns else None
            job.reply = reply
            job.tool_calls = last.tool_calls if last else []
            job.status = JobStatus.DONE
            store.save_job(job)
            idx = _index(tenant_id); idx.save_campaign(campaign); idx.close()  # keep the sidebar fresh
        except Exception as exc:  # noqa: BLE001 - surface the failure as an ERROR job, never crash the worker
            job = store.get_job(job_id)
            if job is not None:
                job.status = JobStatus.ERROR
                job.error = str(exc)
                store.save_job(job)
        finally:
            store.close(); priors.close()

    @app.post("/api/campaigns/{campaign_id}/chat")
    def chat(campaign_id: str, body: ChatIn, tenant: Tenant = Depends(current_tenant)):
        """Enqueue a chat turn and return immediately -- the planner loop (esp. extraction) runs too
        long for an HTTP request. Poll GET .../jobs/{job_id} for the reply."""
        store = _store(tenant.id, campaign_id)
        campaign = store.get_campaign(campaign_id)
        if campaign is None:
            store.close()
            raise HTTPException(404, "No such campaign.")
        # Fix the session id up front so the client can keep the thread; the worker reuses it.
        session = (store.get_session(body.session_id) if body.session_id else None)
        if session is None:
            session = Session(campaign_id=campaign_id)
            store.save_session(session)
        job = Job(campaign_id=campaign_id, session_id=session.id, message=body.message)
        store.save_job(job)
        session_id, job_id = session.id, job.id
        store.close()  # close before submitting; the worker opens its own connection

        runner.submit(lambda: _run_chat_job(tenant.id, campaign_id, job_id, session_id, body.message))
        return {"job_id": job_id, "session_id": session_id, "status": JobStatus.QUEUED.value}

    @app.get("/api/campaigns/{campaign_id}/jobs/{job_id}")
    def get_job(campaign_id: str, job_id: str, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        job = store.get_job(job_id)
        store.close()
        if job is None or job.campaign_id != campaign_id:
            raise HTTPException(404, "No such job.")
        return {"job_id": job.id, "status": job.status.value, "session_id": job.session_id,
                "reply": job.reply, "tool_calls": job.tool_calls, "error": job.error,
                "updated_at": job.updated_at.isoformat()}

    @app.get("/api/campaigns/{campaign_id}/jobs")
    def list_jobs(campaign_id: str, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        out = [{"job_id": j.id, "status": j.status.value, "kind": j.kind,
                "message": j.message, "created_at": j.created_at.isoformat()}
               for j in store.list_jobs(campaign_id)]
        store.close()
        return out

    # -- direct log-result form ------------------------------------------

    @app.post("/api/campaigns/{campaign_id}/log")
    def log_result(campaign_id: str, body: LogResultIn, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        campaign = store.get_campaign(campaign_id)
        if campaign is None:
            store.close()
            raise HTTPException(404, "No such campaign.")
        priors = _priors(tenant.id)
        ctx = ToolContext(store=store, prior_store=priors, campaign=campaign)
        try:
            output, gate = run_tool(ctx, "store.log_result", body.model_dump())
        except Exception as e:  # noqa: BLE001
            store.close(); priors.close()
            raise HTTPException(400, str(e))
        store.close(); priors.close()
        return {"result": output, "gate": gate.model_dump()}

    # -- sessions + export ------------------------------------------------

    @app.get("/api/campaigns/{campaign_id}/sessions")
    def list_sessions(campaign_id: str, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        out = [{"id": s.id, "started_at": s.started_at.isoformat(),
                "turns": len(s.turns), "cost": s.total_cost}
               for s in store.list_sessions(campaign_id)]
        store.close()
        return out

    @app.get("/api/campaigns/{campaign_id}/sessions/{session_id}/export")
    def export_session(campaign_id: str, session_id: str, tenant: Tenant = Depends(current_tenant)):
        store = _store(tenant.id, campaign_id)
        campaign = store.get_campaign(campaign_id)
        session = store.get_session(session_id)
        if campaign is None or session is None:
            store.close()
            raise HTTPException(404, "No such campaign or session.")
        md = session_to_markdown(store, campaign, session)
        store.close()
        return PlainTextResponse(md, media_type="text/markdown")

    return app
