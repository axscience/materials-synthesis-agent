"""Tool registry -- the deterministic tools the planner calls.

Each tool is an Anthropic tool-use definition plus a handler that wraps an existing module. Handlers
operate on a ToolContext (the campaign, its stores, and an LLM client for the extraction role). The
loop-critical handlers reuse the exact logic the CLI/web front ends already use, so there is one
implementation of the science, not a second.

`run_tool` executes a handler and runs its gate; the planner turns a blocked gate into an error
result the model must handle. Design-side tools are guarded behind the optional inverse_design_agent
import, so the synthesis loop is fully functional without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from materials_synthesis_agent.harness.campaign import (
    Campaign,
    CharacterizationResult,
    PropertyComparison,
    SynthesisOutcome,
)
from materials_synthesis_agent.harness.gates import Gate, run_gates
from materials_synthesis_agent.harness.store import CampaignStore
from materials_synthesis_agent.harness.warm_prior import (
    PriorStore,
    as_literature_anchors,
    build_warm_prior,
)


@dataclass
class ToolContext:
    store: CampaignStore
    prior_store: PriorStore
    campaign: Campaign
    llm: Optional[Any] = None                 # LLMClient, for the extraction role
    model: str = "claude-sonnet-4-6"
    unpaywall_email: Optional[str] = None


class ToolError(RuntimeError):
    """A handler failure the planner should see as an error result (not a crash)."""


# --------------------------------------------------------------------------- #
# Parameter-space helpers (kept torch-free where possible)
# --------------------------------------------------------------------------- #

def _space_param_names(specs: list[dict]) -> set[str]:
    return {s["name"] for s in specs}


def _space_categoricals(specs: list[dict]) -> dict[str, set]:
    return {s["name"]: set(s["categories"]) for s in specs if s.get("kind") == "categorical"}


def _build_parameter_space(specs: list[dict]):
    """Construct the optimizer's ParameterSpace from the stored spec dicts (imports torch)."""
    from materials_synthesis_agent.optimize import ParameterSpace, ParameterSpec

    parsed = []
    for s in specs:
        parsed.append(ParameterSpec(
            name=s["name"], kind=s["kind"],
            bounds=tuple(s["bounds"]) if s.get("bounds") else None,
            categories=tuple(s["categories"]) if s.get("categories") else None,
        ))
    return ParameterSpace(parsed)


def _protocol_params_for_prior(candidate, specs: list[dict]) -> dict[str, Any]:
    """Parse a ProtocolCandidate's conditions into the SAME typed params the optimizer uses
    (`protocol_to_params`), so what the flywheel records matches what suggest_next later consumes.
    Returns {} if there is no space or the candidate can't be parsed -- the experiment is still
    saved; only the prior record is skipped, never corrupted with unparsable values."""
    if not specs:
        return {}
    try:
        from materials_synthesis_agent.cli.params import protocol_to_params

        space = _build_parameter_space(specs)
        return dict(protocol_to_params(candidate, space))
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

def h_setup_space(ctx: ToolContext, specs: list[dict]) -> dict:
    """Define / update the optimizer's search space for this campaign."""
    for s in specs:
        if "name" not in s or s.get("kind") not in ("continuous", "categorical"):
            raise ToolError(f"Bad parameter spec: {s!r}")
        if s["kind"] == "continuous" and not s.get("bounds"):
            raise ToolError(f"Continuous parameter '{s.get('name')}' needs bounds.")
        if s["kind"] == "categorical" and not s.get("categories"):
            raise ToolError(f"Categorical parameter '{s.get('name')}' needs categories.")
    ctx.campaign.space = {"specs": specs}
    ctx.store.save_campaign(ctx.campaign)
    return {"dim": len(specs), "parameters": [s["name"] for s in specs]}


def h_warm_prior(ctx: ToolContext) -> dict:
    """Query the cross-campaign prior for this campaign's chemistry + metric."""
    target = ctx.store.get_target(ctx.campaign.target_id) if ctx.campaign.target_id else None
    if target is None:
        raise ToolError("Campaign has no target yet.")
    specs = (ctx.campaign.space or {}).get("specs", [])
    wp = build_warm_prior(
        ctx.prior_store,
        linkage_chemistry=target.linkage_chemistry,
        metric_name=target.metric_name,
        maximize=target.maximize,
        required_params=_space_param_names(specs) or None,
        categorical_values=_space_categoricals(specs) or None,
    )
    return wp.model_dump()


def h_log_result(ctx: ToolContext, protocol_id: str, metrics: list[dict], notes: str = "") -> dict:
    """Record a bench result: save the Experiment, feed the cross-campaign prior (the flywheel),
    and create a CharacterizationResult. This is the 'measure' edge of the loop."""
    from materials_synthesis_agent.schema import Experiment, Metric

    target = ctx.store.get_target(ctx.campaign.target_id)
    if target is None:
        raise ToolError("Campaign has no target yet.")
    candidate = next(
        (c for c in ctx.store.list_protocol_candidates(target.id) if c.id == protocol_id), None
    )
    if candidate is None:
        raise ToolError(f"No protocol candidate {protocol_id} in this campaign.")

    metric_objs = [
        Metric(
            name=m["name"], value=float(m["value"]),
            uncertainty=(None if m.get("uncertainty") is None else float(m["uncertainty"])),
            measurement_method=m.get("measurement_method") or target.metric_measurement_method,
        )
        for m in metrics
    ]
    exp = Experiment(
        project_target_id=target.id, protocol_candidate_id=protocol_id, metrics=metric_objs
    )
    ctx.store.save_experiment(exp)

    # Feed the flywheel: record the primary metric against this chemistry's conditions.
    specs = (ctx.campaign.space or {}).get("specs", [])
    params = _protocol_params_for_prior(candidate, specs)
    primary = next((m for m in metric_objs if m.name == target.metric_name), None)
    if primary is not None and params:
        ctx.prior_store.record(
            linkage_chemistry=target.linkage_chemistry,
            metric_name=target.metric_name,
            params=params,
            metric_value=primary.value,
            campaign_id=ctx.campaign.id,
        )

    char = CharacterizationResult(
        experiment_id=exp.id,
        candidate_id=protocol_id,
        property_comparisons=[
            PropertyComparison.build(m.name, measured=m.value, measured_uncertainty=m.uncertainty)
            for m in metric_objs
        ],
        crystallinity_assessment=notes or None,
        outcome=SynthesisOutcome.UNKNOWN,
    )
    ctx.store.save_characterization(ctx.campaign.id, char)
    ctx.campaign.characterization_ids.append(char.id)
    ctx.store.save_campaign(ctx.campaign)

    return {"experiment_id": exp.id, "recorded_to_prior": primary is not None and bool(params),
            "n_metrics": len(metric_objs)}


def _build_observations(ctx: ToolContext, space, objective_name: str):
    """Observations for the optimizer -- literature-seeded first, then user-logged, exactly as the
    CLI's suggest-next does. The literature seeding (selected_experiments_to_observations) is what
    lets the GP propose a *first* protocol from the papers' real data before any bench result exists;
    the user-logged experiments are added as they come in on each rerun of the loop."""
    from materials_synthesis_agent.cli.params import protocol_to_params
    from materials_synthesis_agent.literature.outcomes import selected_experiments_to_observations
    from materials_synthesis_agent.optimize import Observation

    target_id = ctx.campaign.target_id
    candidates = {c.id: c for c in ctx.store.list_protocol_candidates(target_id)}

    obs = []
    # 1. Literature-seeded observations: the paper's real (conditions -> outcome) data points, for
    #    every extracted experiment the extractor marked as matching this objective.
    for cand in candidates.values():
        obs.extend(selected_experiments_to_observations(cand, space, objective_name))

    # 2. User-logged bench results (accumulate across reruns of the loop).
    for exp in ctx.store.list_experiments(target_id):
        cand = candidates.get(exp.protocol_candidate_id)
        metric = next((m for m in exp.metrics if m.name == objective_name), None)
        if cand is None or metric is None:
            continue
        try:
            params = protocol_to_params(cand, space)
        except ValueError:
            continue
        obs.append(Observation(params=params, value=metric.value, uncertainty=metric.uncertainty))
    return obs


def h_suggest_next(ctx: ToolContext) -> dict:
    """Propose the next experiment via BO. Blocked upstream by the calibration gate (raises
    CalibrationError, which the planner surfaces as a 409-style error) when overconfident."""
    from materials_synthesis_agent.optimize import SingleObjectiveOptimizer
    from materials_synthesis_agent.schema import BOSuggestion, Decision

    target = ctx.store.get_target(ctx.campaign.target_id)
    if target is None:
        raise ToolError("Campaign has no target yet.")
    specs = (ctx.campaign.space or {}).get("specs")
    if not specs:
        raise ToolError("No parameter space set. Call opt.setup_space first.")

    space = _build_parameter_space(specs)
    objective = target.all_objectives[0]
    obs = _build_observations(ctx, space, objective.name)
    if len(obs) < 2:
        raise ToolError("Need at least 2 logged results before the optimizer can suggest.")

    optimizer = SingleObjectiveOptimizer(space, maximize=target.maximize,
                                         variance_scale=ctx.campaign.variance_scale)
    # Gate: raises CalibrationError if the surrogate is overconfident.
    optimizer.check_calibration_or_raise(obs)

    # Warm start from prior campaigns on this chemistry.
    wp = build_warm_prior(
        ctx.prior_store, target.linkage_chemistry, target.metric_name, maximize=target.maximize,
        required_params=_space_param_names(specs), categorical_values=_space_categoricals(specs),
    )
    anchors = as_literature_anchors(wp)
    suggestion = optimizer.suggest_next(obs, literature_anchors=anchors or None)

    from materials_synthesis_agent.cli.params import params_to_new_candidate
    new_candidate = params_to_new_candidate(suggestion.params, target.id)
    ctx.store.save_protocol_candidate(new_candidate)
    ctx.store.save_bo_suggestion(BOSuggestion(
        target_id=target.id, protocol_candidate_id=new_candidate.id,
        expected_improvement=suggestion.expected_improvement,
        uncertainty=suggestion.predicted_uncertainty, rationale=suggestion.rationale,
    ))
    prev = ctx.store.latest_decision()
    ctx.store.save_decision(Decision(
        context=f"Suggest next experiment after {len(obs)} results.",
        options_considered=list({c.id for c in ctx.store.list_protocol_candidates(target.id)}),
        chosen_protocol_candidate_id=new_candidate.id, rationale=suggestion.rationale,
        expected_outcome=f"{objective.name} ~= {suggestion.predicted_value:.4g} "
                         f"(+/- {suggestion.predicted_uncertainty:.4g})",
        followed_from=prev.id if prev else None,
    ))
    return {
        "protocol_candidate_id": new_candidate.id,
        "params": suggestion.params,
        "predicted_value": suggestion.predicted_value,
        "predicted_uncertainty": suggestion.predicted_uncertainty,
        "expected_improvement": suggestion.expected_improvement,
        "n_warm_anchors": len(anchors),
        "rationale": suggestion.rationale,
    }


def h_calibration_report(ctx: ToolContext) -> dict:
    from materials_synthesis_agent.optimize import SingleObjectiveOptimizer

    target = ctx.store.get_target(ctx.campaign.target_id)
    specs = (ctx.campaign.space or {}).get("specs")
    if target is None or not specs:
        raise ToolError("Need a target and a parameter space first.")
    space = _build_parameter_space(specs)
    objective = target.all_objectives[0]
    obs = _build_observations(ctx, space, objective.name)
    optimizer = SingleObjectiveOptimizer(space, maximize=target.maximize)
    return optimizer.calibration_report(obs).model_dump()


def h_lit_search(ctx: ToolContext, query: str, limit: int = 10) -> dict:
    from materials_synthesis_agent.literature import search

    papers = search(query, limit=limit)
    return {"papers": [
        {"source_id": p.source_id, "title": p.title, "year": p.year,
         "open_access": bool(getattr(p, "oa_pdf_url", None))}
        for p in papers
    ]}


def h_extract_protocols(ctx: ToolContext, n: int = 5) -> dict:
    """The extractor: pull N citation-grounded candidate protocols (with their literature
    experiments) from the papers, save them, and mark the experiments whose outcome metric matches
    this campaign's objective as seeds for the optimizer -- so opt.suggest_next can propose a first
    protocol from the papers' real data with no bench result yet."""
    if ctx.llm is None:
        raise ToolError("Extraction needs an LLM client -- set ANTHROPIC_API_KEY so the harness "
                        "can run the extractor.")
    target = ctx.store.get_target(ctx.campaign.target_id) if ctx.campaign.target_id else None
    if target is None:
        raise ToolError("Campaign has no target yet.")

    from materials_synthesis_agent.literature import generate_protocols
    from materials_synthesis_agent.literature.outcomes import _metric_matches

    candidates = generate_protocols(
        target, n=n, client=ctx.llm, model=ctx.model, unpaywall_email=ctx.unpaywall_email,
    )
    seedable_total = 0
    summary = []
    for c in candidates:
        n_seed = 0
        for exp in c.literature_experiments:
            if _metric_matches(exp.outcome.metric_name, target.metric_name):
                exp.selected_for_seeding = True  # auto-select matching experiments for GP seeding
                n_seed += 1
        seedable_total += n_seed
        ctx.store.save_protocol_candidate(c)
        summary.append({
            "id": c.id,
            "source": getattr(c.source, "value", str(c.source)),
            "n_building_blocks": len(c.building_blocks or {}),
            "n_experiments": len(c.literature_experiments),
            "n_seedable": n_seed,
        })
    return {"n_candidates": len(candidates), "n_seedable_experiments": seedable_total,
            "candidates": summary}


def h_feasibility(ctx: ToolContext, protocol_id: str) -> dict:
    from materials_synthesis_agent.feasibility import check_protocol_candidate

    target = ctx.store.get_target(ctx.campaign.target_id)
    candidate = next(
        (c for c in ctx.store.list_protocol_candidates(target.id) if c.id == protocol_id), None
    )
    if candidate is None:
        raise ToolError(f"No protocol candidate {protocol_id}.")
    report = check_protocol_candidate(candidate)
    return report.model_dump() if hasattr(report, "model_dump") else dict(report)


def _design_unavailable(ctx: ToolContext, **_: Any) -> dict:
    raise ToolError(
        "Design-side tools require the inverse_design_agent package, which is not installed in this "
        "environment. Install it to enable design.generate_candidates / design.handoff."
    )


HANDLERS: dict[str, Callable[..., dict]] = {
    "opt.setup_space": h_setup_space,
    "opt.suggest_next": h_suggest_next,
    "opt.calibration_report": h_calibration_report,
    "data.warm_prior": h_warm_prior,
    "store.log_result": h_log_result,
    "lit.search": h_lit_search,
    "lit.extract_protocols": h_extract_protocols,
    "feas.check": h_feasibility,
    "design.generate_candidates": _design_unavailable,
    "design.handoff": _design_unavailable,
}


# --------------------------------------------------------------------------- #
# Anthropic tool-use definitions the planner sees
# --------------------------------------------------------------------------- #

TOOL_SPECS: list[dict] = [
    {"name": "opt.setup_space",
     "description": "Define the optimizer's search space (temperature, time, solvent, ...).",
     "input_schema": {"type": "object", "properties": {
         "specs": {"type": "array", "items": {"type": "object"}}}, "required": ["specs"]}},
    {"name": "opt.suggest_next",
     "description": "Propose the next experiment via Bayesian optimization over logged results. "
                    "BLOCKED if the surrogate's uncertainty is overconfident (calibration gate).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "opt.calibration_report",
     "description": "Leave-one-out calibration diagnostics for the current surrogate.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "data.warm_prior",
     "description": "What past experiments on this chemistry suggest as starting conditions.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "store.log_result",
     "description": "Record a bench result (measured metrics with uncertainty) for a protocol.",
     "input_schema": {"type": "object", "properties": {
         "protocol_id": {"type": "string"},
         "metrics": {"type": "array", "items": {"type": "object"}},
         "notes": {"type": "string"}}, "required": ["protocol_id", "metrics"]}},
    {"name": "lit.search",
     "description": "Search the literature for relevant COF synthesis papers (metadata only).",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}},
    {"name": "lit.extract_protocols",
     "description": "Extract N citation-grounded candidate protocols and their experiments from the "
                    "papers, and seed the optimizer with the experiments matching the objective. Run "
                    "this before opt.suggest_next on a fresh campaign so the GP has data to start from.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    {"name": "feas.check",
     "description": "Check building-block validity and purchasability for a protocol.",
     "input_schema": {"type": "object", "properties": {
         "protocol_id": {"type": "string"}}, "required": ["protocol_id"]}},
]


def run_tool(ctx: ToolContext, name: str, tool_input: dict) -> tuple[dict, Gate]:
    """Execute a handler and run its gate. Raises KeyError for an unknown tool; CalibrationError and
    ToolError propagate to the planner's dispatch, which turns them into error results."""
    handler = HANDLERS[name]
    output = handler(ctx, **(tool_input or {}))
    gate = run_gates(name, output)
    return output, gate
