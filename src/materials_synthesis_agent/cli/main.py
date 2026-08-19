"""CLI entrypoints: init, suggest-protocols, log-result, suggest-next.

Every command that spends money (suggest-protocols) shows a cost estimate and asks for
confirmation before running, unless --yes is passed -- CLAUDE.md guardrail on cost governance,
enforced locally the same way materials-copilot enforces it for hosted users.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from materials_synthesis_agent.cli import project as proj
from materials_synthesis_agent.cli.llm_config import get_configured_provider, save_llm_config
from materials_synthesis_agent.cli.params import params_to_new_candidate, protocol_to_params
from materials_synthesis_agent.feasibility import check_protocol_candidate
from materials_synthesis_agent.literature import build_query, estimate_generation_cost, generate_protocols, search
from materials_synthesis_agent.llm import PROVIDERS
from materials_synthesis_agent.optimize import LiteratureAnchor, Observation, ParameterSpace, SingleObjectiveOptimizer
from materials_synthesis_agent.schema import Decision, Experiment, Metric, ObjectiveDirection, Target, TargetObjective
from materials_synthesis_agent.storage import Store

app = typer.Typer(help="A literature-informed, Bayesian-optimization-driven agent for closed-loop materials synthesis.")
console = Console()


def _load_target(name: str, store: Store) -> Target:
    target_id = proj.target_path(name).read_text().strip()
    target = store.get_target(target_id)
    if target is None:
        console.print(f"[red]No target found for project '{name}' (id {target_id}).[/red]")
        raise typer.Exit(1)
    return target


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name -- becomes a local directory."),
    functional_groups: str = typer.Option(..., prompt="Functional groups (comma-separated)"),
    linkage_chemistry: str = typer.Option(..., prompt="Linkage chemistry (e.g. 'imine condensation')"),
    application: str = typer.Option(..., prompt="Target application"),
    metric_name: str = typer.Option(..., prompt="Metric to optimize (e.g. 'crystallinity')"),
    metric_measurement_method: str = typer.Option(..., prompt="How will you measure it? (e.g. 'PXRD peak area ratio')"),
    direction: str = typer.Option(
        "maximize",
        prompt="Do you want to 'maximize' or 'minimize' this metric?",
        help="'maximize' (e.g. yield, crystallinity) or 'minimize' (e.g. particle size, defect density, cost).",
    ),
):
    """Create a new local project and define its synthesis target."""

    def _parse_direction(raw: str) -> ObjectiveDirection:
        norm = raw.strip().lower()
        if norm not in (ObjectiveDirection.MAXIMIZE.value, ObjectiveDirection.MINIMIZE.value):
            console.print(f"[red]'{raw}' must be 'maximize' or 'minimize'.[/red]")
            raise typer.Exit(1)
        return ObjectiveDirection(norm)

    first_direction = _parse_direction(direction)

    # The first metric (prompted above) is objective #1. Optionally collect more, for a
    # multi-objective (Pareto) project -- e.g. maximize yield AND minimize cost simultaneously.
    objectives = [TargetObjective(name=metric_name, measurement_method=metric_measurement_method, direction=first_direction)]
    console.print("[dim]Optimize more than one metric at once? Add more objectives, or leave the name blank to finish.[/dim]")
    while True:
        extra_name = typer.prompt("Additional metric to optimize (blank to finish)", default="", show_default=False)
        if not extra_name.strip():
            break
        extra_method = typer.prompt(f"How will you measure '{extra_name.strip()}'?")
        extra_direction = _parse_direction(typer.prompt(f"'maximize' or 'minimize' '{extra_name.strip()}'?", default="maximize"))
        objectives.append(TargetObjective(name=extra_name.strip(), measurement_method=extra_method, direction=extra_direction))

    proj.project_dir(name).mkdir(parents=True, exist_ok=True)
    store = Store(proj.db_path(name))
    target = Target(
        functional_groups=[g.strip() for g in functional_groups.split(",") if g.strip()],
        linkage_chemistry=linkage_chemistry,
        application=application,
        metric_name=metric_name,
        metric_measurement_method=metric_measurement_method,
        objective_direction=first_direction,
        # Only store the list when there's genuinely more than one; a single objective stays on the
        # legacy fields so single-objective projects are byte-for-byte unchanged.
        objectives=objectives if len(objectives) > 1 else [],
    )
    store.save_target(target)
    proj.target_path(name).write_text(target.id)
    space_path = proj.write_example_parameter_space(name)
    store.close()

    console.print(f"[green]Project '{name}' created.[/green]")
    if len(objectives) > 1:
        console.print("Optimizing " + ", ".join(f"{o.name} ({o.direction.value})" for o in objectives) + " together (Pareto).")
    console.print(f"Edit [bold]{space_path}[/bold] to match this target's real synthesis parameters, then run:")
    console.print(f"  materials-agent suggest-protocols {name}")


@app.command()
def configure():
    """Choose an LLM provider and enter your API key -- one-time setup, shared across every local
    project. The key is stored locally at ~/.materials-agent/config.json (readable only by you)
    and is never sent anywhere except in requests to the provider you choose."""
    console.print("[bold]Choose an LLM provider:[/bold]")
    provider_keys = list(PROVIDERS.keys())
    for i, key in enumerate(provider_keys, start=1):
        config = PROVIDERS[key]
        console.print(f"  {i}. {config.display_name} (default model: {config.default_model})")

    choice = typer.prompt("Enter a number", type=int)
    if not (1 <= choice <= len(provider_keys)):
        console.print(f"[red]'{choice}' isn't one of the options above.[/red]")
        raise typer.Exit(1)
    provider = provider_keys[choice - 1]
    config = PROVIDERS[provider]

    console.print(
        f"\nEnter your {config.display_name} API key (input hidden). "
        f"You can also skip this step entirely and just set {config.env_var} in your shell."
    )
    api_key = typer.prompt("API key", hide_input=True)

    model = typer.prompt(
        f"Model to use (leave blank for the default, {config.default_model})", default="", show_default=False
    )

    save_llm_config(provider, api_key, model=model or None)
    console.print(
        f"\n[green]Configured.[/green] {config.display_name} "
        f"({model or config.default_model}) will be used for suggest-protocols from now on."
    )


@app.command()
def suggest_protocols(
    name: str,
    n: int = typer.Option(5, help="Number of candidate protocols to generate."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-estimate confirmation prompt."),
):
    """Search the literature and extract N citation-grounded candidate protocols."""
    if get_configured_provider() is None:
        console.print(
            "[red]No LLM provider configured.[/red] Run [bold]materials-agent configure[/bold] first, "
            "or set one of: " + ", ".join(c.env_var for c in PROVIDERS.values())
        )
        raise typer.Exit(1)

    store = Store(proj.db_path(name))
    target = _load_target(name, store)

    papers = search(build_query(target), limit=n * 3)
    estimated_cost = estimate_generation_cost(target, papers[: n * 3])
    console.print(f"Found {len(papers)} candidate papers. Estimated extraction cost: [bold]${estimated_cost:.4f}[/bold]")
    if not yes and not typer.confirm("Proceed?"):
        raise typer.Exit(0)

    retrosynthesis_config = proj.get_global_retrosynthesis_config()
    candidates = generate_protocols(target, n=n)
    for candidate in candidates:
        candidate.feasibility_flags = check_protocol_candidate(candidate, retrosynthesis_config=retrosynthesis_config)
        store.save_protocol_candidate(candidate)

    store.record_usage("llm_call", estimated_cost, idempotency_key=f"suggest-protocols:{target.id}:{len(candidates)}")
    store.close()

    table = Table(title=f"{len(candidates)} candidate protocols")
    table.add_column("id")
    table.add_column("solvent")
    table.add_column("temp (C)")
    table.add_column("time (h)")
    table.add_column("citation coverage")
    table.add_column("feasibility flags")
    for c in candidates:
        table.add_row(
            c.id[:8],
            c.solvent.value if c.solvent else "-",
            c.temperature_c.value if c.temperature_c else "-",
            c.time_hours.value if c.time_hours else "-",
            f"{c.citation_coverage():.0%}",
            "; ".join(c.feasibility_flags) or "none",
        )
    console.print(table)


@app.command()
def log_result(
    name: str,
    protocol_candidate_id: str,
    value: float = typer.Option(None, help="Single-objective: the measured metric value."),
    uncertainty: float = typer.Option(None, help="Single-objective: measurement uncertainty. Omitting it flags the result low-confidence."),
    metric: list[str] = typer.Option(
        [], "--metric", help="Multi-objective: name=value (repeat once per objective, e.g. --metric yield=0.7 --metric cost=12)."
    ),
    metric_uncertainty: list[str] = typer.Option(
        [], "--metric-uncertainty", help="Multi-objective: name=uncertainty (repeat; optional per objective)."
    ),
    deviation: list[str] = typer.Option([], help="key=value pairs describing how you deviated from the protocol."),
):
    """Log a lab result against a protocol candidate.

    Single-objective projects: pass --value (and optionally --uncertainty).
    Multi-objective projects: pass one --metric name=value per objective (and optional
    --metric-uncertainty name=value)."""
    store = Store(proj.db_path(name))
    target = _load_target(name, store)

    candidates = store.list_protocol_candidates(target.id)
    matches = [c for c in candidates if c.id.startswith(protocol_candidate_id)]
    if len(matches) != 1:
        console.print(f"[red]{len(matches)} protocol candidates match '{protocol_candidate_id}' -- need exactly 1.[/red]")
        raise typer.Exit(1)
    candidate = matches[0]

    def _parse_pairs(items: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for item in items:
            if "=" not in item:
                console.print(f"[red]'{item}' must be name=value.[/red]")
                raise typer.Exit(1)
            key, val = item.split("=", 1)
            out[key.strip()] = float(val)
        return out

    objective_names = {o.name for o in target.all_objectives}
    metrics: list[Metric] = []

    if metric:
        # Multi-objective path.
        values = _parse_pairs(metric)
        uncertainties = _parse_pairs(metric_uncertainty)
        missing = objective_names - set(values)
        if missing:
            console.print(f"[red]Missing --metric value(s) for: {', '.join(sorted(missing))}.[/red]")
            raise typer.Exit(1)
        for obj in target.all_objectives:
            metrics.append(Metric(
                name=obj.name, value=values[obj.name], uncertainty=uncertainties.get(obj.name),
                measurement_method=obj.measurement_method,
            ))
    else:
        # Single-objective path.
        if target.is_multi_objective:
            console.print(f"[red]This project has multiple objectives ({', '.join(sorted(objective_names))}). Use --metric name=value for each.[/red]")
            raise typer.Exit(1)
        if value is None:
            console.print("[red]--value is required for a single-objective project.[/red]")
            raise typer.Exit(1)
        obj = target.all_objectives[0]
        metrics.append(Metric(name=obj.name, value=value, uncertainty=uncertainty, measurement_method=obj.measurement_method))

    deviations = dict(d.split("=", 1) for d in deviation)
    experiment = Experiment(project_target_id=target.id, protocol_candidate_id=candidate.id, deviations=deviations, metrics=metrics)
    store.save_experiment(experiment)
    store.close()

    summary = ", ".join(
        f"{m.name}={m.value}" + ("" if m.uncertainty is not None else " (low-confidence)") for m in metrics
    )
    console.print(f"[green]Logged.[/green] {summary}")


@app.command()
def suggest_next(name: str):
    """Get the next recommended experiment(s) from the Bayesian optimizer.

    Single-objective projects get one recommendation. Multi-objective projects get a Pareto set --
    several experiments that trade the objectives off against each other, for you to choose from."""
    space = ParameterSpace(proj.load_parameter_space(name))
    store = Store(proj.db_path(name))
    target = _load_target(name, store)

    candidates = store.list_protocol_candidates(target.id)
    experiments = store.list_experiments(target.id)
    candidates_by_id = {c.id: c for c in candidates}

    if target.is_multi_objective:
        _suggest_next_multi(name, space, store, target, candidates, experiments, candidates_by_id)
    else:
        _suggest_next_single(name, space, store, target, candidates, experiments, candidates_by_id)


def _suggest_next_single(name, space, store, target, candidates, experiments, candidates_by_id):
    objective = target.all_objectives[0]
    observations = []
    for exp in experiments:
        candidate = candidates_by_id.get(exp.protocol_candidate_id)
        matching_metric = next((m for m in exp.metrics if m.name == objective.name), None)
        if candidate is None or matching_metric is None:
            continue
        try:
            params = protocol_to_params(candidate, space)
        except ValueError as exc:
            console.print(f"[yellow]Skipping experiment {exp.id[:8]}: {exc}[/yellow]")
            continue
        observations.append(Observation(params=params, value=matching_metric.value, uncertainty=matching_metric.uncertainty))

    if len(observations) < 2:
        console.print(
            f"[yellow]Only {len(observations)} usable result(s) logged -- need at least 2 for a "
            "meaningful recommendation.[/yellow] Try one of the un-tried literature protocols from "
            "`suggest-protocols` first, then log its result."
        )
        raise typer.Exit(0)

    anchors = []
    for c in candidates:
        try:
            anchors.append(LiteratureAnchor(params=protocol_to_params(c, space)))
        except ValueError:
            continue

    optimizer = SingleObjectiveOptimizer(space, maximize=target.maximize)
    suggestion = optimizer.suggest_next(observations, literature_anchors=anchors)

    new_candidate = params_to_new_candidate(suggestion.params, target.id)
    store.save_protocol_candidate(new_candidate)

    from materials_synthesis_agent.schema import BOSuggestion

    store.save_bo_suggestion(BOSuggestion(
        target_id=target.id, protocol_candidate_id=new_candidate.id,
        expected_improvement=suggestion.expected_improvement, uncertainty=suggestion.predicted_uncertainty,
        rationale=suggestion.rationale,
    ))
    previous_decision = store.latest_decision()
    store.save_decision(Decision(
        context=f"Suggest next experiment for target '{target.application}' after {len(observations)} results.",
        options_considered=[c.id for c in candidates], chosen_protocol_candidate_id=new_candidate.id,
        rationale=suggestion.rationale,
        expected_outcome=f"{objective.name} ~= {suggestion.predicted_value:.4g} (+/- {suggestion.predicted_uncertainty:.4g})",
        followed_from=previous_decision.id if previous_decision else None,
    ))
    store.close()

    console.print(f"[bold green]Next suggested experiment (candidate {new_candidate.id[:8]}):[/bold green]")
    for k, v in suggestion.params.items():
        console.print(f"  {k}: {v}")
    console.print(f"Expected {objective.name}: {suggestion.predicted_value:.4g} (+/- {suggestion.predicted_uncertainty:.4g})")
    console.print(f"[dim]{suggestion.rationale}[/dim]")


def _suggest_next_multi(name, space, store, target, candidates, experiments, candidates_by_id):
    from materials_synthesis_agent.optimize import MultiObjectiveOptimizer, MultiObservation, Objective
    from materials_synthesis_agent.schema import BOSuggestion

    objectives = target.all_objectives
    objective_names = [o.name for o in objectives]

    observations = []
    for exp in experiments:
        candidate = candidates_by_id.get(exp.protocol_candidate_id)
        if candidate is None:
            continue
        by_name = {m.name: m for m in exp.metrics}
        if not all(n in by_name for n in objective_names):
            continue  # need every objective measured for this experiment to use it
        try:
            params = protocol_to_params(candidate, space)
        except ValueError as exc:
            console.print(f"[yellow]Skipping experiment {exp.id[:8]}: {exc}[/yellow]")
            continue
        observations.append(MultiObservation(
            params=params,
            values={n: by_name[n].value for n in objective_names},
            uncertainties={n: by_name[n].uncertainty for n in objective_names},
        ))

    if len(observations) < 3:
        console.print(
            f"[yellow]Only {len(observations)} usable result(s) with all {len(objectives)} objectives measured -- "
            "need at least 3 for a multi-objective recommendation.[/yellow] Log more results (each with every "
            "metric) first."
        )
        raise typer.Exit(0)

    optimizer = MultiObjectiveOptimizer(space, [Objective(name=o.name, maximize=o.maximize) for o in objectives])
    pareto = optimizer.suggest_next(observations, n_suggestions=4)

    import uuid
    pareto_set_id = str(uuid.uuid4())
    previous_decision = store.latest_decision()
    rows = []
    for s in pareto:
        new_candidate = params_to_new_candidate(s.params, target.id)
        store.save_protocol_candidate(new_candidate)
        store.save_bo_suggestion(BOSuggestion(
            target_id=target.id, protocol_candidate_id=new_candidate.id, rationale=s.rationale,
            pareto_set_id=pareto_set_id, predicted_values=s.predicted_values,
        ))
        rows.append((new_candidate, s))

    store.save_decision(Decision(
        context=f"Suggest Pareto set for target '{target.application}' ({', '.join(objective_names)}) after {len(observations)} results.",
        options_considered=[c.id for c in candidates],
        chosen_protocol_candidate_id=rows[0][0].id if rows else "",
        rationale=f"{len(rows)} Pareto-optimal candidates; researcher chooses the tradeoff.",
        expected_outcome="; ".join(objective_names),
        followed_from=previous_decision.id if previous_decision else None,
    ))
    store.close()

    console.print(f"[bold green]{len(rows)} Pareto-optimal next experiments[/bold green] -- pick the tradeoff you want, run it, and log-result:")
    table = Table(title="Pareto set (predicted objective values)")
    table.add_column("candidate")
    for spec in space.specs:
        table.add_column(spec.name)
    for o in objectives:
        table.add_column(f"{o.name} ({o.direction.value})")
    for candidate, s in rows:
        param_cells = [str(round(s.params[spec.name], 2) if isinstance(s.params[spec.name], float) else s.params[spec.name]) for spec in space.specs]
        pred_cells = [f"{s.predicted_values[o.name]:.4g}" for o in objectives]
        table.add_row(candidate.id[:8], *param_cells, *pred_cells)
    console.print(table)


@app.command()
def serve(
    name: str,
    port: int = typer.Option(8000, help="Local port to serve on."),
    host: str = typer.Option("127.0.0.1", help="Bind address -- stays local by default."),
):
    """Launch the minimal local web UI for one project (v0.2, no auth, single-user, local-only)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]The web UI needs the 'web' extra: pip install 'materials-synthesis-agent[web]'[/red]")
        raise typer.Exit(1)

    if not proj.target_path(name).exists():
        console.print(f"[red]No project '{name}' found in the current directory. Run `materials-agent init {name}` first.[/red]")
        raise typer.Exit(1)

    os.environ["MATERIALS_AGENT_PROJECT"] = name
    console.print(f"Serving '{name}' at [bold]http://{host}:{port}[/bold] (local only, no auth)")
    uvicorn.run("materials_synthesis_agent.web.app:app", host=host, port=port)


# Real sizes of AiZynthFinder's public data, confirmed via HEAD/range requests against the actual
# Zenodo/figshare hosts -- not the "several GB" this project assumed before checking.
_RETROSYNTHESIS_DOWNLOAD_FILES = [
    ("uspto_model.onnx", "91.5 MB", "expansion policy"),
    ("uspto_templates.csv.gz", "3.3 MB", "expansion policy"),
    ("uspto_ringbreaker_model.onnx", "15.0 MB", "ringbreaker policy"),
    ("uspto_ringbreaker_templates.csv.gz", "0.4 MB", "ringbreaker policy"),
    ("uspto_filter_model.onnx", "16.8 MB", "filter policy"),
    ("zinc_stock.hdf5", "632.4 MB", "purchasable-stock database"),
]
_RETROSYNTHESIS_TOTAL_SIZE = "~759 MB"


@app.command()
def setup_retrosynthesis(
    data_dir: str = typer.Option(
        None, help="Where to store the downloaded data (default: ~/.materials-agent/retrosynthesis-data)"
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the download-size confirmation prompt."),
):
    """One-time download of AiZynthFinder's public policy/stock data, shared across all local
    projects. After this, suggest-protocols automatically uses retrosynthesis for building blocks
    that aren't confirmed purchasable."""
    try:
        import aizynthfinder  # noqa: F401
    except ImportError:
        console.print(
            "[red]Install the retrosynthesis extra first:[/red] "
            "pip install \"materials-synthesis-agent[retrosynthesis]\""
        )
        raise typer.Exit(1)

    target_dir = proj.default_retrosynthesis_data_dir() if data_dir is None else Path(data_dir)
    config_path = target_dir / "config.yml"

    if config_path.exists():
        console.print(f"Already set up at [bold]{config_path}[/bold].")
        if not yes and not typer.confirm("Re-download anyway?"):
            proj.set_global_retrosynthesis_config(str(config_path))
            raise typer.Exit(0)

    table = Table(title=f"Retrosynthesis public data ({_RETROSYNTHESIS_TOTAL_SIZE} total, from Zenodo + figshare)")
    table.add_column("file")
    table.add_column("size")
    table.add_column("used for")
    for filename, size, purpose in _RETROSYNTHESIS_DOWNLOAD_FILES:
        table.add_row(filename, size, purpose)
    console.print(table)

    if not yes and not typer.confirm(f"Download {_RETROSYNTHESIS_TOTAL_SIZE} to {target_dir}?"):
        raise typer.Exit(0)

    target_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Downloading to {target_dir} (this reuses AiZynthFinder's own download_public_data)...")
    result = subprocess.run(["download_public_data", str(target_dir)])
    if result.returncode != 0 or not config_path.exists():
        console.print("[red]Download failed -- see output above.[/red]")
        raise typer.Exit(1)

    proj.set_global_retrosynthesis_config(str(config_path))
    console.print(
        f"[bold green]Done.[/bold green] Retrosynthesis configured at {config_path}. "
        "suggest-protocols will use it automatically from now on."
    )


if __name__ == "__main__":
    app()
