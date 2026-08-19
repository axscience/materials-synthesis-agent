"""Minimal local web UI -- v0.2 (ROADMAP.md), an alternative to the CLI for one local project.

Still no auth, still single-user, still fully local (CLAUDE.md guardrail: nothing in this package
should assume a hosted context). Server-rendered HTML, no build step, no JS framework -- this is
meant to be a thin, honest view over the same Store/optimizer/literature modules the CLI uses, not
a second implementation of the product logic.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from materials_synthesis_agent.cli import project as proj
from materials_synthesis_agent.cli.params import params_to_new_candidate, protocol_to_params
from materials_synthesis_agent.feasibility import check_protocol_candidate
from materials_synthesis_agent.literature import build_query, estimate_generation_cost, generate_protocols, search
from materials_synthesis_agent.optimize import LiteratureAnchor, Observation, ParameterSpace, SingleObjectiveOptimizer
from materials_synthesis_agent.schema import BOSuggestion, Decision, Experiment, Metric
from materials_synthesis_agent.storage import Store

app = FastAPI(title="materials-synthesis-agent (local)")


def _project_name() -> str:
    name = os.environ.get("MATERIALS_AGENT_PROJECT")
    if not name:
        raise RuntimeError("MATERIALS_AGENT_PROJECT is not set -- launch via `materials-agent serve <project-name>`.")
    return name


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html><head><title>{title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
th {{ background: #f4f4f4; }}
.low-confidence {{ color: #a15c00; }}
.flag {{ color: #b00020; }}
form.inline {{ display: inline; }}
h2 {{ margin-top: 2rem; }}
</style></head>
<body><h1>{title}</h1>{body}<p><a href="/">&larr; dashboard</a></p></body></html>""")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    name = _project_name()
    store = Store(proj.db_path(name))
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)
    candidates = store.list_protocol_candidates(target_id)
    experiments = store.list_experiments(target_id)
    suggestions = store.list_bo_suggestions(target_id)
    trajectory = store.reconstruct_trajectory()
    store.close()

    objectives = target.all_objectives

    def _log_form(candidate_id: str) -> str:
        # One value+uncertainty input pair per objective, named by index (m0/u0, m1/u1, ...) so
        # objective names with spaces don't break form field names. Order matches all_objectives.
        inputs = "".join(
            f"<input name='m{i}' placeholder='{o.name}' size='6' required>"
            f"<input name='u{i}' placeholder='&plusmn; (opt)' size='5'>"
            for i, o in enumerate(objectives)
        )
        return (
            f"<form class='inline' method='post' action='/log-result'>"
            f"<input type='hidden' name='protocol_candidate_id' value='{candidate_id}'>"
            f"{inputs}<button type='submit'>log result</button></form>"
        )

    candidate_rows = "".join(
        f"<tr><td>{c.id[:8]}</td><td>{c.source.value}</td>"
        f"<td>{c.solvent.value if c.solvent else '-'}</td>"
        f"<td>{c.temperature_c.value if c.temperature_c else '-'}</td>"
        f"<td>{c.time_hours.value if c.time_hours else '-'}</td>"
        f"<td>{c.citation_coverage():.0%}</td>"
        f"<td class='flag'>{'; '.join(c.feasibility_flags) or ''}</td>"
        f"<td>{_log_form(c.id)}</td></tr>"
        for c in candidates
    )
    experiment_rows = "".join(
        f"<tr><td>{e.protocol_candidate_id[:8]}</td>"
        + "".join(
            f"<td class='{'low-confidence' if m.confidence.value == 'low_confidence' else ''}'>"
            f"{m.name}={m.value}{'' if m.uncertainty is None else f' &plusmn;{m.uncertainty}'}</td>"
            for m in e.metrics
        )
        + "</tr>"
        for e in experiments
    )
    if target.is_multi_objective:
        # Multi-objective: show predicted objective values per suggestion.
        suggestion_rows = "".join(
            f"<tr><td>{s.protocol_candidate_id[:8]}</td>"
            + "".join(f"<td>{s.predicted_values.get(o.name, float('nan')):.4g}</td>" for o in objectives)
            + f"<td>{'Pareto set ' + s.pareto_set_id[:8] if s.pareto_set_id else ''}</td></tr>"
            for s in suggestions
        )
        suggestion_header = (
            "<tr><th>candidate</th>" + "".join(f"<th>{o.name} ({o.direction.value})</th>" for o in objectives) + "<th>group</th></tr>"
        )
    else:
        suggestion_rows = "".join(
            f"<tr><td>{s.protocol_candidate_id[:8]}</td><td>{s.expected_improvement:.4g}</td>"
            f"<td>{s.uncertainty:.4g}</td><td>{s.rationale}</td></tr>"
            for s in suggestions
        )
        suggestion_header = "<tr><th>candidate</th><th>expected improvement</th><th>uncertainty</th><th>rationale</th></tr>"

    decision_rows = "".join(
        f"<tr><td>{d.context}</td><td>{d.expected_outcome}</td><td>{d.actual_outcome or '(pending)'}</td></tr>"
        for d in trajectory
    )

    objectives_line = ", ".join(f"<b>{o.direction.value} {o.name}</b> ({o.measurement_method})" for o in objectives)
    suggest_button_label = "Get Pareto set" if target.is_multi_objective else "Get BO suggestion"
    log_headers = "".join(f"<th>{o.name}</th>" for o in objectives)

    body = f"""
    <p><b>Target:</b> {target.application} &mdash; optimizing {objectives_line}</p>
    <p><b>Functional groups:</b> {", ".join(target.functional_groups)} &mdash;
    <b>Linkage chemistry:</b> {target.linkage_chemistry}</p>

    <h2>Protocol candidates ({len(candidates)})</h2>
    <form method="get" action="/suggest-protocols"><button type="submit">Search literature for more candidates</button></form>
    <table><tr><th>id</th><th>source</th><th>solvent</th><th>temp</th><th>time</th>
    <th>citations</th><th>feasibility flags</th><th>log result ({log_headers and 'per metric'})</th></tr>{candidate_rows}</table>

    <h2>Logged experiments ({len(experiments)})</h2>
    <table><tr><th>candidate</th><th>metrics</th></tr>{experiment_rows}</table>

    <h2>Next experiment{'s (Pareto set)' if target.is_multi_objective else ''}</h2>
    <form method="post" action="/suggest-next"><button type="submit">{suggest_button_label}</button></form>
    <table>{suggestion_header}{suggestion_rows}</table>

    <h2>Decision trajectory</h2>
    <table><tr><th>context</th><th>expected</th><th>actual</th></tr>{decision_rows}</table>
    """
    return _page(f"materials-synthesis-agent: {name}", body)


@app.get("/suggest-protocols", response_class=HTMLResponse)
def suggest_protocols_confirm():
    name = _project_name()
    store = Store(proj.db_path(name))
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)
    store.close()

    papers = search(build_query(target), limit=15)
    cost = estimate_generation_cost(target, papers)
    body = f"""<p>Found {len(papers)} candidate papers. Estimated extraction cost: <b>${cost:.4f}</b></p>
    <form method="post" action="/suggest-protocols/run"><button type="submit">Proceed and spend ~${cost:.4f}</button></form>"""
    return _page("Confirm literature search cost", body)


@app.post("/suggest-protocols/run")
def suggest_protocols_run():
    name = _project_name()
    store = Store(proj.db_path(name))
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)

    retrosynthesis_config = proj.get_global_retrosynthesis_config()
    candidates = generate_protocols(target, n=5)
    for c in candidates:
        c.feasibility_flags = check_protocol_candidate(c, retrosynthesis_config=retrosynthesis_config)
        store.save_protocol_candidate(c)
    store.close()
    return RedirectResponse("/", status_code=303)


@app.post("/log-result")
async def log_result(request: Request):
    # The form has a dynamic number of metric inputs (m0/u0, m1/u1, ... one per objective), so
    # read the raw form rather than declaring static Form(...) params.
    form = await request.form()
    protocol_candidate_id = form["protocol_candidate_id"]

    name = _project_name()
    store = Store(proj.db_path(name))
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)

    metrics = []
    for i, obj in enumerate(target.all_objectives):
        raw_value = form.get(f"m{i}", "")
        if not str(raw_value).strip():
            continue
        raw_unc = form.get(f"u{i}", "")
        unc = float(raw_unc) if str(raw_unc).strip() else None
        metrics.append(Metric(name=obj.name, value=float(raw_value), uncertainty=unc, measurement_method=obj.measurement_method))

    experiment = Experiment(project_target_id=target_id, protocol_candidate_id=protocol_candidate_id, metrics=metrics)
    store.save_experiment(experiment)
    store.close()
    return RedirectResponse("/", status_code=303)


@app.post("/suggest-next")
def suggest_next():
    name = _project_name()
    space = ParameterSpace(proj.load_parameter_space(name))
    store = Store(proj.db_path(name))
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)

    candidates = store.list_protocol_candidates(target_id)
    experiments = store.list_experiments(target_id)
    candidates_by_id = {c.id: c for c in candidates}

    if target.is_multi_objective:
        result = _web_suggest_multi(space, store, target, target_id, candidates, experiments, candidates_by_id)
    else:
        result = _web_suggest_single(space, store, target, target_id, candidates, experiments, candidates_by_id)
    return result


def _web_suggest_single(space, store, target, target_id, candidates, experiments, candidates_by_id):
    objective = target.all_objectives[0]
    observations = []
    for exp in experiments:
        candidate = candidates_by_id.get(exp.protocol_candidate_id)
        metric = next((m for m in exp.metrics if m.name == objective.name), None)
        if candidate is None or metric is None:
            continue
        try:
            params = protocol_to_params(candidate, space)
        except ValueError:
            continue
        observations.append(Observation(params=params, value=metric.value, uncertainty=metric.uncertainty))

    if len(observations) < 2:
        store.close()
        return _page("Not enough data", "<p>Need at least 2 logged results with values for every parameter-space field.</p>")

    anchors = []
    for c in candidates:
        try:
            anchors.append(LiteratureAnchor(params=protocol_to_params(c, space)))
        except ValueError:
            continue

    optimizer = SingleObjectiveOptimizer(space, maximize=target.maximize)
    suggestion = optimizer.suggest_next(observations, literature_anchors=anchors)

    new_candidate = params_to_new_candidate(suggestion.params, target_id)
    store.save_protocol_candidate(new_candidate)
    store.save_bo_suggestion(BOSuggestion(
        target_id=target_id, protocol_candidate_id=new_candidate.id,
        expected_improvement=suggestion.expected_improvement, uncertainty=suggestion.predicted_uncertainty,
        rationale=suggestion.rationale,
    ))
    previous = store.latest_decision()
    store.save_decision(Decision(
        context=f"Suggest next experiment for target '{target.application}' after {len(observations)} results.",
        options_considered=[c.id for c in candidates], chosen_protocol_candidate_id=new_candidate.id,
        rationale=suggestion.rationale,
        expected_outcome=f"{objective.name} ~= {suggestion.predicted_value:.4g} (+/- {suggestion.predicted_uncertainty:.4g})",
        followed_from=previous.id if previous else None,
    ))
    store.close()
    return RedirectResponse("/", status_code=303)


def _web_suggest_multi(space, store, target, target_id, candidates, experiments, candidates_by_id):
    from materials_synthesis_agent.optimize import MultiObjectiveOptimizer, MultiObservation, Objective
    import uuid

    objectives = target.all_objectives
    names = [o.name for o in objectives]
    observations = []
    for exp in experiments:
        candidate = candidates_by_id.get(exp.protocol_candidate_id)
        if candidate is None:
            continue
        by_name = {m.name: m for m in exp.metrics}
        if not all(n in by_name for n in names):
            continue
        try:
            params = protocol_to_params(candidate, space)
        except ValueError:
            continue
        observations.append(MultiObservation(
            params=params, values={n: by_name[n].value for n in names},
            uncertainties={n: by_name[n].uncertainty for n in names},
        ))

    if len(observations) < 3:
        store.close()
        return _page("Not enough data", f"<p>Need at least 3 results with all {len(objectives)} objectives measured.</p>")

    optimizer = MultiObjectiveOptimizer(space, [Objective(name=o.name, maximize=o.maximize) for o in objectives])
    pareto = optimizer.suggest_next(observations, n_suggestions=4)

    pareto_set_id = str(uuid.uuid4())
    for s in pareto:
        new_candidate = params_to_new_candidate(s.params, target_id)
        store.save_protocol_candidate(new_candidate)
        store.save_bo_suggestion(BOSuggestion(
            target_id=target_id, protocol_candidate_id=new_candidate.id, rationale=s.rationale,
            pareto_set_id=pareto_set_id, predicted_values=s.predicted_values,
        ))
    previous = store.latest_decision()
    store.save_decision(Decision(
        context=f"Suggest Pareto set for target '{target.application}' ({', '.join(names)}) after {len(observations)} results.",
        options_considered=[c.id for c in candidates], chosen_protocol_candidate_id="",
        rationale=f"{len(pareto)} Pareto-optimal candidates; researcher chooses the tradeoff.",
        expected_outcome="; ".join(names), followed_from=previous.id if previous else None,
    ))
    store.close()
    return RedirectResponse("/", status_code=303)
