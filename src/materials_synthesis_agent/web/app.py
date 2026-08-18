"""Minimal local web UI -- v0.2 (ROADMAP.md), an alternative to the CLI for one local project.

Still no auth, still single-user, still fully local (CLAUDE.md guardrail: nothing in this package
should assume a hosted context). Server-rendered HTML, no build step, no JS framework -- this is
meant to be a thin, honest view over the same Store/optimizer/literature modules the CLI uses, not
a second implementation of the product logic.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Form
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

    candidate_rows = "".join(
        f"<tr><td>{c.id[:8]}</td><td>{c.source.value}</td>"
        f"<td>{c.solvent.value if c.solvent else '-'}</td>"
        f"<td>{c.temperature_c.value if c.temperature_c else '-'}</td>"
        f"<td>{c.time_hours.value if c.time_hours else '-'}</td>"
        f"<td>{c.citation_coverage():.0%}</td>"
        f"<td class='flag'>{'; '.join(c.feasibility_flags) or ''}</td>"
        f"<td><form class='inline' method='post' action='/log-result'>"
        f"<input type='hidden' name='protocol_candidate_id' value='{c.id}'>"
        f"<input name='value' placeholder='{target.metric_name} value' size='6' required>"
        f"<input name='uncertainty' placeholder='+/- (optional)' size='6'>"
        f"<button type='submit'>log result</button></form></td></tr>"
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
    suggestion_rows = "".join(
        f"<tr><td>{s.protocol_candidate_id[:8]}</td><td>{s.expected_improvement:.4g}</td>"
        f"<td>{s.uncertainty:.4g}</td><td>{s.rationale}</td></tr>"
        for s in suggestions
    )
    decision_rows = "".join(
        f"<tr><td>{d.context}</td><td>{d.expected_outcome}</td><td>{d.actual_outcome or '(pending)'}</td></tr>"
        for d in trajectory
    )

    body = f"""
    <p><b>Target:</b> {target.application} &mdash; <b>{target.objective_direction.value}</b>
    <b>{target.metric_name}</b> ({target.metric_measurement_method})</p>
    <p><b>Functional groups:</b> {", ".join(target.functional_groups)} &mdash;
    <b>Linkage chemistry:</b> {target.linkage_chemistry}</p>

    <h2>Protocol candidates ({len(candidates)})</h2>
    <form method="get" action="/suggest-protocols"><button type="submit">Search literature for more candidates</button></form>
    <table><tr><th>id</th><th>source</th><th>solvent</th><th>temp</th><th>time</th>
    <th>citations</th><th>feasibility flags</th><th>log a result</th></tr>{candidate_rows}</table>

    <h2>Logged experiments ({len(experiments)})</h2>
    <table><tr><th>candidate</th><th>metrics</th></tr>{experiment_rows}</table>

    <h2>Next experiment</h2>
    <form method="post" action="/suggest-next"><button type="submit">Get BO suggestion</button></form>
    <table><tr><th>candidate</th><th>expected improvement</th><th>uncertainty</th><th>rationale</th></tr>{suggestion_rows}</table>

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
def log_result(protocol_candidate_id: str = Form(...), value: float = Form(...), uncertainty: str = Form("")):
    name = _project_name()
    store = Store(proj.db_path(name))
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)

    unc = float(uncertainty) if uncertainty.strip() else None
    metric = Metric(name=target.metric_name, value=value, uncertainty=unc, measurement_method=target.metric_measurement_method)
    experiment = Experiment(project_target_id=target_id, protocol_candidate_id=protocol_candidate_id, metrics=[metric])
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

    observations = []
    for exp in experiments:
        candidate = candidates_by_id.get(exp.protocol_candidate_id)
        metric = next((m for m in exp.metrics if m.name == target.metric_name), None)
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
    store.save_bo_suggestion(
        BOSuggestion(
            target_id=target_id,
            protocol_candidate_id=new_candidate.id,
            expected_improvement=suggestion.expected_improvement,
            uncertainty=suggestion.predicted_uncertainty,
            rationale=suggestion.rationale,
        )
    )
    previous = store.latest_decision()
    store.save_decision(
        Decision(
            context=f"Suggest next experiment for target '{target.application}' after {len(observations)} results.",
            options_considered=[c.id for c in candidates],
            chosen_protocol_candidate_id=new_candidate.id,
            rationale=suggestion.rationale,
            expected_outcome=f"{target.metric_name} ~= {suggestion.predicted_value:.4g} (+/- {suggestion.predicted_uncertainty:.4g})",
            followed_from=previous.id if previous else None,
        )
    )
    store.close()
    return RedirectResponse("/", status_code=303)
