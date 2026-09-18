"""Async job execution for the harness API (Phase 4, P4.1).

The planner loop -- especially literature extraction -- can run for many minutes, far longer than an
HTTP request should live. So a chat turn is enqueued as a `Job`: the request returns immediately with
a `job_id`, a worker runs the planner in the background (it persists the session itself), and the
front end polls the job for the reply.

Two runners behind one interface, injected the same way `caller_factory` is, so tests stay
deterministic:

  * `ThreadJobRunner`  -- a thread pool; the production default on a single container. No Redis, so no
      extra infra for the first multi-user deployment. (When you outgrow one container, a Redis/arq
      runner implements the same `submit` and nothing else changes.)
  * `InlineJobRunner`  -- runs the work immediately, in-thread; used by tests so a job is `done` the
      moment it is submitted.

The job function opens its OWN stores inside the worker thread (SQLite connections are per-thread),
so the request handler must finish its own DB work and close before submitting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, Field

from materials_synthesis_agent.harness.campaign import _new_id  # id factory used across the harness


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Job(BaseModel):
    """One background unit of work (currently always a chat turn) and its result."""
    id: str = Field(default_factory=_new_id)
    campaign_id: str
    session_id: Optional[str] = None
    kind: str = "chat"
    status: JobStatus = JobStatus.QUEUED
    message: str = ""                       # the user input, for display while running
    reply: Optional[str] = None
    tool_calls: list[Any] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()


class JobRunner(Protocol):
    def submit(self, fn: Callable[[], None]) -> None: ...
    def shutdown(self) -> None: ...


class InlineJobRunner:
    """Runs the work synchronously on submit -- for tests and any caller that wants blocking behavior.
    A job is `done` (or `error`) by the time `submit` returns."""

    def submit(self, fn: Callable[[], None]) -> None:
        fn()

    def shutdown(self) -> None:  # nothing to tear down
        pass


class ThreadJobRunner:
    """Background thread pool. The production default: no external broker, fine for one container."""

    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="harness-job")

    def submit(self, fn: Callable[[], None]) -> None:
        self._pool.submit(fn)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
