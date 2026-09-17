"""The planner -- one LLM role driving the native tool-use loop over the deterministic tool
registry. This is the whole orchestrator: not a hand-built DAG of ten agents, but one model that
emits tool calls, reads results (with gates applied), and decides what's next until it stops.

The loop lives behind a `ToolCaller` interface so it can be driven by a real Anthropic client in
production or a scripted fake in tests (the same discipline the rest of this codebase uses for
provider calls). `dispatch` turns a blocked gate or a CalibrationError into an error tool_result the
model must handle, rather than crashing or silently proceeding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from materials_synthesis_agent.harness.campaign import Session, Turn
from materials_synthesis_agent.harness.tools import TOOL_SPECS, ToolContext, run_tool


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False
    cost: float = 0.0


@dataclass
class PlannerResult:
    final_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns_used: int = 0
    hit_cap: bool = False


DispatchFn = Callable[[str, dict], ToolOutcome]


class ToolCaller(Protocol):
    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        dispatch: DispatchFn,
        max_turns: int = 12,
    ) -> PlannerResult: ...


SYSTEM_PROMPT = (
    "You are the research director for a covalent-organic-framework discovery campaign. You reach "
    "the science only through the provided tools; never invent results. Plan the smallest sequence "
    "of tool calls that answers the user, call them, and read their results. A tool result marked "
    "as an error (for example, the calibration gate blocking a suggestion because the surrogate is "
    "overconfident) is not something to work around silently -- explain it to the user and say what "
    "would resolve it (usually: log more results). Every number you report must come from a tool "
    "result and carry the uncertainty that tool gave. Be concise and specific."
)


def build_context_block(ctx: ToolContext) -> str:
    """A compact summary of where the campaign stands, prepended to the user's message."""
    c = ctx.campaign
    lines = [f"[Campaign '{c.name}' — status {c.status.value}]"]
    target = ctx.store.get_target(c.target_id) if c.target_id else None
    if target is not None:
        lines.append(f"Target: {target.application}; linkage={target.linkage_chemistry}; "
                     f"objective={target.metric_name} ({'maximize' if target.maximize else 'minimize'}).")
        n_exp = len(ctx.store.list_experiments(target.id))
        n_prot = len(ctx.store.list_protocol_candidates(target.id))
        lines.append(f"{n_prot} protocol candidate(s), {n_exp} logged experiment(s).")
    else:
        lines.append("No target set yet.")
    if c.space:
        lines.append(f"Parameter space: {[s['name'] for s in c.space.get('specs', [])]}.")
    return "\n".join(lines)


class Planner:
    def __init__(self, ctx: ToolContext, caller: ToolCaller, max_turns: int = 12):
        self.ctx = ctx
        self.caller = caller
        self.max_turns = max_turns

    def _dispatch(self, name: str, tool_input: dict) -> ToolOutcome:
        try:
            output, gate = run_tool(self.ctx, name, tool_input)
        except KeyError:
            return ToolOutcome(content=f"Unknown tool: {name}", is_error=True)
        except Exception as e:  # noqa: BLE001 -- surface any handler failure to the model, don't crash
            if hasattr(e, "report"):  # CalibrationError carries a .report
                return ToolOutcome(
                    content="Calibration gate blocked this call — the surrogate is overconfident:\n"
                    + e.report.summary(),
                    is_error=True,
                )
            return ToolOutcome(content=f"Tool error ({type(e).__name__}): {e}", is_error=True)

        if gate.blocked:
            return ToolOutcome(content=f"Blocked by gate: {gate.message}", is_error=True)
        prefix = f"[gate: {gate.message}]\n" if gate.severity == "warning" else ""
        cost = float(output.get("cost", 0.0)) if isinstance(output, dict) else 0.0
        return ToolOutcome(content=prefix + json.dumps(output, default=str), cost=cost)

    def respond(self, user_message: str, session: Optional[Session] = None) -> str:
        """Run one user turn through the planner loop; persist the exchange to the session."""
        context = build_context_block(self.ctx)
        messages = self._history(session) + [
            {"role": "user", "content": f"{user_message}\n\n{context}"}
        ]
        result = self.caller.run(
            SYSTEM_PROMPT, messages, TOOL_SPECS, self._dispatch, max_turns=self.max_turns
        )

        if session is not None:
            session.turns.append(Turn(role="user", content=user_message))
            session.turns.append(Turn(
                role="assistant", content=result.final_text,
                tool_calls=result.tool_calls,
                cost=sum(tc.get("cost", 0.0) for tc in result.tool_calls),
            ))
            self.ctx.store.save_session(session)
            if session.campaign_id and self.ctx.campaign.id == session.campaign_id \
                    and session.id not in self.ctx.campaign.session_ids:
                self.ctx.campaign.session_ids.append(session.id)
                self.ctx.store.save_campaign(self.ctx.campaign)
        return result.final_text

    def _history(self, session: Optional[Session]) -> list[dict]:
        if session is None:
            return []
        msgs: list[dict] = []
        if session.compressed_history:
            msgs.append({"role": "user", "content": f"[Earlier in this session] {session.compressed_history}"})
        for turn in session.turns[-session.context_window_turns:]:
            if turn.role in ("user", "assistant"):
                msgs.append({"role": turn.role, "content": turn.content})
        return msgs


class AnthropicToolCaller:
    """Production ToolCaller: drives the real Anthropic tool-use loop."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-5", max_tokens: int = 4096):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def run(self, system, messages, tools, dispatch, max_turns=12) -> PlannerResult:
        msgs = list(messages)
        tool_calls: list[dict] = []
        for turn in range(max_turns):
            resp = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=system, tools=tools, messages=msgs,
            )
            msgs.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                text = "".join(getattr(b, "text", "") for b in resp.content
                               if getattr(b, "type", None) == "text")
                return PlannerResult(final_text=text, tool_calls=tool_calls, turns_used=turn + 1)

            results = []
            for tu in tool_uses:
                outcome = dispatch(tu.name, tu.input)
                tool_calls.append({"name": tu.name, "is_error": outcome.is_error, "cost": outcome.cost})
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": outcome.content, "is_error": outcome.is_error})
            msgs.append({"role": "user", "content": results})
        return PlannerResult(final_text="", tool_calls=tool_calls, turns_used=max_turns, hit_cap=True)
