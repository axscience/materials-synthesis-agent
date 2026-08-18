# materials-synthesis-agent

A literature-informed, Bayesian-optimization-driven agent for closed-loop materials synthesis. Given
a target material (starting with covalent organic frameworks — COFs) and an application, it proposes
citation-grounded synthesis protocols from the literature, checks building-block feasibility, and —
as you report back lab results — recommends the next experiment to run.

Runs entirely on your own machine, against your own LLM API key. No account, no login, no data
leaves your machine except the literature/LLM API calls you explicitly trigger.

> **Status: pre-implementation.** Scope and architecture are defined (see below); the code itself has
> not been written yet. This README describes the intended v0.1 behavior.

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

Not yet runnable — see `ROADMAP.md` for the v0.1 milestone. Once available:

```bash
pip install materials-synthesis-agent
export ANTHROPIC_API_KEY=...   # your own key; calls go directly from your machine
materials-agent init my-cof-project
materials-agent suggest-protocols
```

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — architecture guardrails and scope, for anyone (human or agent)
  developing this package
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — component design
- [`ROADMAP.md`](./ROADMAP.md) — what's planned, phased, and explicitly out of scope
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to contribute

## License

MIT — see [`LICENSE`](./LICENSE).
