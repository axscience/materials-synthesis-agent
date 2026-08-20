# materials-synthesis-agent

A literature-informed, Bayesian-optimization-driven agent for closed-loop materials synthesis. Given
a target material (starting with covalent organic frameworks — COFs) and an application, it proposes
citation-grounded synthesis protocols from the literature, checks building-block feasibility, and —
as you report back lab results — recommends the next experiment to run.

Runs entirely on your own machine, against your own LLM API key. No account, no login, no data
leaves your machine except the literature/LLM API calls you explicitly trigger.

> **Status: v0.1, v0.2, and v0.3 implemented, tested, and fully wired end to end.** Literature
> retrieval + citation-grounded extraction (including real open-access full-text extraction, not
> just abstracts), RDKit feasibility checking, single- and multi-objective Bayesian optimization,
> retrosynthesis (AiZynthFinder), a natural-language front door, CIF structure identification, the
> CLI, and a minimal local web UI are all real, working code with a 141-test suite (`pytest`, all
> passing). Retrosynthesis has been run for real against actual COF-relevant chemistry (see
> `tests/test_retrosynthesis.py::TestRealSearch`) -- it correctly proposed nitro-group reduction as
> the route to a diamine linker, the standard real synthesis for that kind of compound, not a random
> disconnection. Structure identification has likewise been run for real against the live
> CURATED-COFs database: 4 held-out fixtures each matched to exactly the right name and citation
> with zero cross-matches (`tests/test_structure_database.py`). Full-text extraction has been run
> against real open-access papers too: a real Nature Communications COF paper's Experimental Section
> was correctly located and yielded a genuine, quantitative synthesis procedure (exact reagent
> masses, mmol, solvent volumes, reflux time) that its abstract never mentioned.

## Why

Materials synthesis optimization today is slow, trial-and-error-heavy, and poorly informed by prior
literature at the point of decision-making — nobody has time to read hundreds of papers before
choosing reaction conditions, and each physical experiment costs real days and reagents. Published
work has already shown the mechanism works: a 2025-Nobel-laureate-led group's literature-informed AI
agent achieved a 350% crystallinity improvement on a benchmark COF. This project makes that loop a
tool anyone can run, not a one-off research prototype tied to one lab's internal infrastructure.

## What it does

0. **Natural-language front door** — describe what you want in your own words (`materials-agent
   ask "..."`). An LLM parses your request into a target, the same forced-tool-use discipline the
   literature agent uses for citations: anything it filled in by inference is shown as such, and
   anything essential it couldn't determine — a metric you never stated, for instance — comes back
   as a question, never a silent guess.

   Mention a CIF file and `ask` tries to identify the structure two ways: first against a local
   copy of the CURATED-COFs database (`materials-agent setup-structure-database`, ~2.6 MB) for an
   exact, citation-backed name match; if that comes back empty (the common case for a genuinely
   novel, hypothesized COF), it falls back to classifying linkage chemistry from the structure's
   real bond graph. That fallback covers **imine and boronate ester only** — the two chemistries
   verified against real, labeled structures — and says so plainly rather than guessing at
   hydrazone, imide, or triazine linkages it hasn't been validated against.
1. **Literature agent** — retrieves papers relevant to your target's functional groups/linkage
   chemistry (Semantic Scholar / OpenAlex / arXiv) — or, if you named a specific known COF, searches
   for that COF by name directly — extracts structured protocols (building blocks, stoichiometry,
   solvent, modulator, temperature, time, concentration), and cites its source for every field it
   asserts. Anything it infers rather than reads directly is flagged `inferred`, never presented as
   sourced fact. It searches **hierarchically by linkage chemistry**: the exact COF first, then
   COFs with the *same* linkage; if those come up short, it asks you before searching COFs with a
   *related* linkage (e.g. an imine target falling back to hydrazone, azine, …). Anything found from
   a different linkage is labeled a weaker prior, never silently mixed in as an exact precedent.
   `suggest-protocols --full-text` tries each paper's real, open-access **Supplementary Information
   and Experimental/Methods section** instead of just its abstract — that's where exact quantities,
   ratios, temperatures, and times usually live, and the SI is often open even when the article is
   paywalled. Falls back to abstract-only per paper when nothing open-access is found.
2. **Feasibility checker** — validates building blocks with RDKit, checks commercial availability,
   and flags (not silently drops) protocols built on infeasible components. Optionally, once you've
   run `materials-agent setup-retrosynthesis`, a non-purchasable block gets a real retrosynthesis
   search (AiZynthFinder) instead of just a dead-end flag — a candidate route to purchasable
   starting materials, if one exists.
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

# describe what you want in your own words -- an LLM parses it into a target, and asks instead of
# guessing if something essential (like what to optimize for) isn't stated:
materials-agent ask "Here's COF-5, tell me how to synthesize it to maximize crystallinity via PXRD peak ratio" --name my-cof-project

# or use the typed prompts directly:
materials-agent init my-cof-project              # prompts for your target, writes parameter_space.json
# edit my-cof-project/parameter_space.json to match your real synthesis parameters
materials-agent suggest-protocols my-cof-project # searches literature, shows cost estimate, confirms before spending
materials-agent suggest-protocols my-cof-project --full-text   # also tries each paper's real open-access full text
materials-agent log-result my-cof-project <candidate-id> --value 0.7 --uncertainty 0.05
materials-agent suggest-next my-cof-project      # Bayesian-optimization recommendation

# or run the local web UI instead of the CLI:
materials-agent serve my-cof-project             # http://127.0.0.1:8000, local only, no auth

# optional: one-time retrosynthesis setup (shared across all local projects)
pip install -e ".[retrosynthesis]"
materials-agent setup-retrosynthesis             # shows the ~759 MB / 6-file breakdown, confirms, downloads
# suggest-protocols now automatically uses it for non-purchasable building blocks

# optional: one-time structure-database setup, for identifying COFs from a CIF file
pip install -e ".[structure]"
materials-agent setup-structure-database         # downloads CURATED-COFs (~2.6 MB), builds a local index
# ask now identifies a CIF's name (if it's a known COF) or linkage chemistry automatically
```

Run the test suite with `pytest` (141 tests, no API key needed -- LLM calls are covered with a fake
client, see `tests/test_extraction.py` and `tests/test_nl_parser.py`; PDF fetching in
`tests/test_fulltext.py` is tested against a real, hand-built minimal PDF with the network mocked).
Retrosynthesis API-shape tests (`tests/test_retrosynthesis.py`) and structure tests
(`tests/test_cif.py`, `tests/test_linkage.py`, `tests/test_structure_database.py`) run
automatically if you've installed the corresponding extra, and skip cleanly if you haven't. The
real end-to-end retrosynthesis search tests (`tests/test_retrosynthesis.py::TestRealSearch`)
additionally need `setup-retrosynthesis` to have been run — expect each to take 80-100+ seconds,
since it's a genuine tree search against a real trained policy network, not a stub. Structure tests
against real CURATED-COFs fixtures
(`tests/test_cif.py`, `tests/test_linkage.py`, and most of `tests/test_structure_database.py`) run
offline against 4 bundled real CIFs; only `tests/test_structure_database.py::TestRealDownload`
needs `setup-structure-database` to have been run, and it's fast (a few MB, not hundreds).

## Known limitations (found during development)

- **Requires Python 3.12, not newer.** `torch` has no wheel for Python 3.14 in a standard pip index
  as of this writing; the venv setup above pins 3.12 deliberately.
- **`torch` is pinned `<2.3`, and `numpy`/`scipy` are pinned below their normal versions to match.**
  torch 2.2.x is built against the NumPy 1.x ABI; newer scipy (a botorch dependency) requires NumPy
  2. See the comment in `pyproject.toml` next to these pins -- revisit once a NumPy-2-compatible
  torch build is available in your install environment.
- **Retrosynthesis was resolved end to end during development -- now fully working, not a
  known-limitation entry anymore, kept here as the paper trail.** `pip install
  "materials-synthesis-agent[retrosynthesis]"` initially failed -- `aizynthfinder` pulls in `numba`
  unconstrained, which resolves to a version requiring an `llvmlite` release with no prebuilt wheel
  for Intel macOS, forcing a from-source build that needs the LLVM toolchain. Fixed by pinning
  `numba<0.61` in the `retrosynthesis` extra. `materials-agent setup-retrosynthesis` then downloads
  the real ~759 MB public data (6 files from Zenodo/figshare: USPTO expansion/ringbreaker/filter
  policy models + a ZINC purchasable-stock database) and wires it in automatically -- `stock:
  zinc` / `expansion: uspto` in the generated config.yml, confirmed to exactly match what
  `feasibility/retrosynthesis.py` already called. Verified with real searches
  (`tests/test_retrosynthesis.py::TestRealSearch`): correctly proposed nitro-group reduction as
  the route to benzidine (a real COF/MOF diamine linker), which is the actual standard synthesis
  for that class of compound -- not a plausible-looking guess. Each real search takes 80-100+
  seconds (genuine MCTS against a trained policy network), so those tests are gated behind the
  data being present and don't run in CI or on a fresh clone.
- **The literature/extraction pipeline has not made a real LLM call.** No Anthropic API key was
  available in the environment this was built in. The extraction and cost-estimation logic is
  covered by tests against a fake client (`tests/test_extraction.py`) and the retrieval half is
  verified against the real Semantic Scholar/arXiv APIs, but the full literature-to-protocol path
  needs your own key to confirm end to end. The same applies to `nl/parser.py` (`ask`'s parsing
  step) -- covered by `tests/test_nl_parser.py` against a fake client, not yet a real call.
- **The linkage-chemistry classifier (`structure/linkage.py`) covers exactly two chemistries: imine
  and boronate ester.** These are the only two verified against real, labeled CURATED-COFs
  structures with measured bond lengths (`tests/test_linkage.py`). Hydrazone, imide, and triazine
  linkages are not implemented -- adding bond-length thresholds for them without a labeled
  structure to check against would be an unverified guess wearing the same confidence dressing as
  the two chemistries that actually are verified, which is worse than just not having them yet.

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — architecture guardrails and scope, for anyone (human or agent)
  developing this package
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — component design
- [`ROADMAP.md`](./ROADMAP.md) — what's planned, phased, and explicitly out of scope
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to contribute

## License

MIT — see [`LICENSE`](./LICENSE).
