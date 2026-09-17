"""Markdown export of a session -- the human-readable lab-notebook log.

Markdown is the wrong format for the system to read back (that's the SQLite campaign state) and the
right format for a person: the researcher can paste this into a lab notebook, share it with a
collaborator, or drop it into a paper's supplementary information. Generated from the structured
store, never hand-maintained.
"""

from __future__ import annotations

from materials_synthesis_agent.harness.campaign import Campaign, Session
from materials_synthesis_agent.harness.store import CampaignStore


def _duration_minutes(session: Session) -> float:
    end = session.ended_at or (session.turns[-1].timestamp if session.turns else session.started_at)
    return max(0.0, (end - session.started_at).total_seconds() / 60.0)


def session_to_markdown(store: CampaignStore, campaign: Campaign, session: Session) -> str:
    lines: list[str] = []
    date = session.started_at.date().isoformat()
    lines.append(f"# Session — {date}")
    lines.append(f"Campaign: {campaign.name}")
    n_tools = sum(len(t.tool_calls) for t in session.turns)
    lines.append(
        f"Duration: {_duration_minutes(session):.0f} min | Turns: {len(session.turns)} | "
        f"Tool calls: {n_tools} | API cost: ${session.total_cost:.2f}"
    )
    lines.append("")

    target = store.get_target(campaign.target_id) if campaign.target_id else None
    if target is not None:
        experiments = store.list_experiments(target.id)
        lines.append("## State at start")
        lines.append(f"- Target: {target.application} ({target.linkage_chemistry})")
        lines.append(f"- Objective: {target.metric_name} "
                     f"({'maximize' if target.maximize else 'minimize'})")
        lines.append(f"- {len(store.list_protocol_candidates(target.id))} protocol candidate(s), "
                     f"{len(experiments)} logged experiment(s)")
        best = _best_experiment(experiments, target.metric_name, target.maximize)
        if best is not None:
            lines.append(f"- Best {target.metric_name} so far: {best}")
        lines.append("")

    lines.append("## What happened")
    if not session.turns:
        lines.append("_(no turns recorded)_")
    step = 0
    for turn in session.turns:
        if turn.role == "user":
            step += 1
            lines.append(f"{step}. **You:** {turn.content}")
        elif turn.role == "assistant":
            if turn.tool_calls:
                names = ", ".join(
                    f"{tc['name']}{' (error)' if tc.get('is_error') else ''}"
                    for tc in turn.tool_calls
                )
                lines.append(f"   - Tools run: {names}")
            if turn.content:
                lines.append(f"   - **Assistant:** {turn.content}")
    lines.append("")

    decisions = store.reconstruct_trajectory()
    if decisions:
        lines.append("## Decisions")
        lines.append("| # | Context | Chose | Rationale |")
        lines.append("|---|---------|-------|-----------|")
        for i, d in enumerate(decisions[-10:], 1):
            chose = (d.chosen_protocol_candidate_id or "")[:8]
            lines.append(f"| D-{i} | {d.context} | {chose} | {d.rationale} |")
        lines.append("")

    if target is not None:
        chars = store.list_characterizations(campaign.id)
        latest = chars[-1] if chars else None
        lines.append("## State at end")
        lines.append(f"- {len(store.list_experiments(target.id))} experiment(s) logged")
        if latest and latest.next_action_suggestion:
            lines.append(f"- Next: {latest.next_action_suggestion}")

    return "\n".join(lines).rstrip() + "\n"


def _best_experiment(experiments, metric_name: str, maximize: bool):
    values = []
    for exp in experiments:
        m = next((mt for mt in exp.metrics if mt.name == metric_name), None)
        if m is not None:
            u = f" ± {m.uncertainty}" if m.uncertainty is not None else ""
            values.append((m.value, f"{m.value}{u}"))
    if not values:
        return None
    return (max if maximize else min)(values, key=lambda v: v[0])[1]
