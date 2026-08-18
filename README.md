# materials-synthesis-agent

A literature-informed, Bayesian-optimization-driven agent for closed-loop materials synthesis. Given
a target material (starting with covalent organic frameworks — COFs) and an application, it proposes
citation-grounded synthesis protocols from the literature, checks building-block feasibility, and —
as you report back lab results — recommends the next experiment to run.

Runs entirely on your own machine, against your own LLM API key. No account, no login, no data
leaves your machine except the literature/LLM API calls you explicitly trigger.

> **Status: v0.1 and v0.2 implemented and tested.** Literature retrieval + citation-grounded
> extraction, RDKit feasibility checking, single- and multi-objective Bayesian optimization, the
> CLI, and a minimal local web UI are all real, working code with a 42-test suite (`pytest`, all
> passing). Retrosynthesis (AiZynthFinder) is written but **not verified live** -- see the
> Known limitations section below before relying on it.

## Why

Materials synthesis optimization today is slow, trial-and-error-heavy, and poorly informed by prior
literature at the point of decision-making — nobody has time to read hundreds of papers before
choosing reaction conditions, and each physical experiment costs real days and reagents. Published
work has already shown the mechanism works: a 2025-Nobel-laureate-led group's literature-informed AI
agent achieved a 350% crystallinity improvement on a benchmark COF. This project makes that loop a
tool anyone can run, not a one-off research prototype tied to one lab's internal infrastructure.

## What it does

1. **Literature agent** — retrieves papers relevant to your target's functional groups/linkage
   chemistry (Semantic Scholar / arXiv / PubMed), extracts structured protocols (building blocks,
   stoichiometry, solvent, modulator, temperature, time, concentration), and cites its source for
   every field it asserts. Anything it infers rather than reads directly is flagged `inferred`, never
   presented as sourced fact.
2. **Feasibility checker** — validates building blocks with RDKit, checks commercial availability,
   and flags (not silently drops) protocols built on infeasible components.
3. **You run it in the lab** and report back what happened — including your metric's measurement
   uncertainty. A result without a stated uncertainty is accepted but flagged low-confidence
   everywhere it's used downstream.
4. **Bayesian optimizer** — a Gaussian-process model (BoTorch) over your protocol's mixed
   continuous/categorical parameter space, seeded with the literature protocols as informed priors
   (not a cold, space-filling search), recommends the next experiment with its expected improvement
   and uncertainty stated explicitly.
5. Repeat until your own convergence criterion (threshold, budget, or plateau) is met.

## What it deliberately does not do

- No lab automation or robotics — you execute the synthesis yourself. (A future integration with
  automated lab hardware is a plausible later direction, not this project's current scope.)
- No retrosynthesis planning in v0.1 — feasibility is validity + purchasability only; retrosynthesis
  (e.g. AiZynthFinder integration) is planned for a later release, see `ROADMAP.md`.
- No material classes beyond COFs yet, and no catalysis *computational* modeling (DFT/ML interatomic
  potentials) — see `ROADMAP.md` for what's planned versus explicitly out of scope.
- No hosted/multi-user mode — that's a separate, closed-source product built on top of this package;
  this repository is the standalone, self-hosted core.

## Safety

This tool proposes synthesis protocols extracted or inferred from literature and statistical
recommendations from an optimizer. **It does not verify chemical safety, and a suggested protocol is
not a substitute for your own chemist's judgment, institutional safety review, or standard lab
safety practices.** Treat every output as a starting point to evaluate, not an instruction to follow
blindly — especially for reagents, quantities, and conditions you haven't independently checked.

## Getting started

Not yet published to PyPI -- install from source. Requires **Python 3.12** (see Known limitations).

```bash
git clone <this-repo-url> && cd materials-synthesis-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[web]"

export ANTHROPIC_API_KEY=...   # your own key; calls go directly from your machine
materials-agent init my-cof-project              # prompts for your target, writes parameter_space.json
# edit my-cof-project/parameter_space.json to match your real synthesis parameters
materials-agent suggest-protocols my-cof-project # searches literature, shows cost estimate, confirms before spending
materials-agent log-result my-cof-project <candidate-id> --value 0.7 --uncertainty 0.05
materials-agent suggest-next my-cof-project      # Bayesian-optimization recommendation

# or run the local web UI instead of the CLI:
materials-agent serve my-cof-project             # http://127.0.0.1:8000, local only, no auth
```

Run the test suite with `pytest` (42 tests, no API key needed -- LLM calls are covered with a fake
client, see `tests/test_extraction.py`).

## Known limitations (found during development, not yet resolved)

- **Requires Python 3.12, not newer.** `torch` has no wheel for Python 3.14 in a standard pip index
  as of this writing; the venv setup above pins 3.12 deliberately.
- **`torch` is pinned `<2.3`, and `numpy`/`scipy` are pinned below their normal versions to match.**
  torch 2.2.x is built against the NumPy 1.x ABI; newer scipy (a botorch dependency) requires NumPy
  2. See the comment in `pyproject.toml` next to these pins -- revisit once a NumPy-2-compatible
  torch build is available in your install environment.
- **Retrosynthesis (`feasibility/retrosynthesis.py`) is unverified.** `pip install aizynthfinder`
  itself failed in this project's development environment (a `llvmlite` wheel build failure -- a
  real, observed failure, not hypothetical), and using it for real also needs a separate multi-GB
  `download_public_data` step regardless. The adapter is written against AiZynthFinder's documented
  API with a fully lazy import (nothing else breaks if it's not installed), but treat it as a
  starting point, not a working integration, until you've exercised it yourself.
- **The literature/extraction pipeline has not made a real LLM call.** No Anthropic API key was
  available in the environment this was built in. The extraction and cost-estimation logic is
  covered by tests against a fake client (`tests/test_extraction.py`) and the retrieval half is
  verified against the real Semantic Scholar/arXiv APIs, but the full literature-to-protocol path
  needs your own key to confirm end to end.

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — architecture guardrails and scope, for anyone (human or agent)
  developing this package
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — component design
- [`ROADMAP.md`](./ROADMAP.md) — what's planned, phased, and explicitly out of scope
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to contribute

## License

MIT — see [`LICENSE`](./LICENSE).
