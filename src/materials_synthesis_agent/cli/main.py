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
from materials_synthesis_agent.cli.params import params_to_new_candidate, protocol_to_params
from materials_synthesis_agent.feasibility import check_protocol_candidate
from materials_synthesis_agent.literature import build_query, estimate_generation_cost, generate_protocols, search
from materials_synthesis_agent.optimize import LiteratureAnchor, Observation, ParameterSpace, SingleObjectiveOptimizer
from materials_synthesis_agent.schema import Decision, Experiment, Metric, Target
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
):
    """Create a new local project and define its synthesis target."""
    proj.project_dir(name).mkdir(parents=True, exist_ok=True)
    store = Store(proj.db_path(name))
    target = Target(
        functional_groups=[g.strip() for g in functional_groups.split(",") if g.strip()],
        linkage_chemistry=linkage_chemistry,
        application=application,
        metric_name=metric_name,
        metric_measurement_method=metric_measurement_method,
    )
    store.save_target(target)
    proj.target_path(name).write_text(target.id)
    space_path = proj.write_example_parameter_space(name)
    store.close()

    console.print(f"[green]Project '{name}' created.[/green]")
    console.print(f"Edit [bold]{space_path}[/bold] to match this target's real synthesis parameters, then run:")
    console.print(f"  materials-agent suggest-protocols {name}")


@app.command()
def suggest_protocols(
    name: str,
    n: int = typer.Option(5, help="Number of candidate protocols to generate."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-estimate confirmation prompt."),
):
    """Search the literature and extract N citation-grounded candidate protocols."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set.[/red] This command calls Claude for extraction; set your key and retry.")
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
    value: float = typer.Option(..., help="The measured metric value."),
    uncertainty: float = typer.Option(None, help="Measurement uncertainty. Omitting this flags the result low-confidence."),
    deviation: list[str] = typer.Option([], help="key=value pairs describing how you deviated from the protocol."),
):
    """Log a lab result against a protocol candidate."""
    store = Store(proj.db_path(name))
    target = _load_target(name, store)

    candidates = store.list_protocol_candidates(target.id)
    matches = [c for c in candidates if c.id.startswith(protocol_candidate_id)]
    if len(matches) != 1:
        console.print(f"[red]{len(matches)} protocol candidates match '{protocol_candidate_id}' -- need exactly 1.[/red]")
        raise typer.Exit(1)
    candidate = matches[0]

    deviations = dict(d.split("=", 1) for d in deviation)
    metric = Metric(name=target.metric_name, value=value, uncertainty=uncertainty, measurement_method=target.metric_measurement_method)
    experiment = Experiment(project_target_id=target.id, protocol_candidate_id=candidate.id, deviations=deviations, metrics=[metric])
    store.save_experiment(experiment)
    store.close()

    confidence_note = "" if uncertainty is not None else " [yellow](low-confidence: no uncertainty given)[/yellow]"
    console.print(f"[green]Logged.[/green] {target.metric_name}={value}{confidence_note}")


@app.command()
def suggest_next(name: str):
    """Get the next recommended experiment from the Bayesian optimizer."""
    space = ParameterSpace(proj.load_parameter_space(name))
    store = Store(proj.db_path(name))
    target = _load_target(name, store)

    candidates = store.list_protocol_candidates(target.id)
    experiments = store.list_experiments(target.id)
    candidates_by_id = {c.id: c for c in candidates}

    observations = []
    for exp in experiments:
        candidate = candidates_by_id.get(exp.protocol_candidate_id)
        matching_metric = next((m for m in exp.metrics if m.name == target.metric_name), None)
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

    optimizer = SingleObjectiveOptimizer(space, maximize=True)
    suggestion = optimizer.suggest_next(observations, literature_anchors=anchors)

    new_candidate = params_to_new_candidate(suggestion.params, target.id)
    store.save_protocol_candidate(new_candidate)

    from materials_synthesis_agent.schema import BOSuggestion

    bo_suggestion = BOSuggestion(
        target_id=target.id,
        protocol_candidate_id=new_candidate.id,
        expected_improvement=suggestion.expected_improvement,
        uncertainty=suggestion.predicted_uncertainty,
        rationale=suggestion.rationale,
    )
    store.save_bo_suggestion(bo_suggestion)

    previous_decision = store.latest_decision()
    decision = Decision(
        context=f"Suggest next experiment for target '{target.application}' after {len(observations)} results.",
        options_considered=[c.id for c in candidates],
        chosen_protocol_candidate_id=new_candidate.id,
        rationale=suggestion.rationale,
        expected_outcome=f"{target.metric_name} ~= {suggestion.predicted_value:.4g} (+/- {suggestion.predicted_uncertainty:.4g})",
        followed_from=previous_decision.id if previous_decision else None,
    )
    store.save_decision(decision)
    store.close()

    console.print(f"[bold green]Next suggested experiment (candidate {new_candidate.id[:8]}):[/bold green]")
    for k, v in suggestion.params.items():
        console.print(f"  {k}: {v}")
    console.print(f"Expected {target.metric_name}: {suggestion.predicted_value:.4g} (+/- {suggestion.predicted_uncertainty:.4g})")
    console.print(f"[dim]{suggestion.rationale}[/dim]")


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
