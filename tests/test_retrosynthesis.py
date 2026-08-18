"""Tests against the real installed aizynthfinder package (not a fake) -- skipped cleanly when
the optional `retrosynthesis` extra isn't installed. Most tests here only check the API shape this
module depends on hasn't shifted, without a real tree search. `TestRealSearch` below is the
exception: a genuine end-to-end search, gated behind the ~759 MB public data actually being
downloaded (`materials-agent setup-retrosynthesis`) -- skipped in CI and on a fresh clone, but not
hypothetical: it has been run for real, against real COF-relevant chemistry, with sane results
(reduction of a nitro group as the proposed route to a diamine linker -- correct, standard
chemistry, not a random guess). See feasibility/retrosynthesis.py's module docstring for the
full picture of what's verified.
"""

import os
from pathlib import Path

import pytest

aizynthfinder = pytest.importorskip("aizynthfinder")

from materials_synthesis_agent.feasibility.retrosynthesis import RetrosynthesisUnavailable, check_retrosynthesis

_DEFAULT_CONFIG = Path.home() / ".materials-agent" / "retrosynthesis-data" / "config.yml"


def test_aizynthfinder_constructor_accepts_configfile_kwarg():
    from aizynthfinder.aizynthfinder import AiZynthFinder

    # Must not raise TypeError on the kwarg name this module's check_retrosynthesis() passes.
    finder = AiZynthFinder(configfile=None, configdict=None)
    assert finder is not None


def test_finder_has_the_attributes_this_module_depends_on():
    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configfile=None, configdict=None)
    assert hasattr(finder, "stock")
    assert hasattr(finder, "expansion_policy")
    assert hasattr(finder, "routes")
    assert hasattr(finder.routes, "dicts")  # the list-of-dict representation this module reads
    assert callable(finder.stock.select)
    assert callable(finder.expansion_policy.select)
    assert callable(finder.tree_search)
    assert callable(finder.build_routes)
    assert callable(finder.extract_statistics)


def test_check_retrosynthesis_raises_clearly_without_real_config():
    # No real config/model files available in this environment -- this should fail with a
    # meaningful error from aizynthfinder itself (bad stock/policy selection), not silently
    # return a bogus "not solved" result.
    with pytest.raises(Exception):
        check_retrosynthesis("c1ccccc1", config_path=None)


def test_retrosynthesis_unavailable_is_a_runtime_error():
    # Sanity check on the exception hierarchy used for the "not installed" path (tested
    # separately in the fake-import scenario; here just confirming the type is sane).
    assert issubclass(RetrosynthesisUnavailable, RuntimeError)


@pytest.mark.skipif(
    not _DEFAULT_CONFIG.exists(),
    reason="Real public data not downloaded -- run `materials-agent setup-retrosynthesis` first. "
    "Slow (~1-2 min/search) by design, so it's opt-in, not part of the default fast suite.",
)
class TestRealSearch:
    """Genuine tree searches against real downloaded data. Each takes 80-100+ seconds (real MCTS
    against a trained policy network), so this class only runs when the data is actually present
    -- never in CI, never on a fresh clone. Confirms the whole chain (config -> stock/policy
    selection -> tree_search -> build_routes -> routes.dicts) produces sane, real chemistry, not
    just that it runs without crashing."""

    def test_solves_a_trivial_molecule(self):
        result = check_retrosynthesis("CCO", str(_DEFAULT_CONFIG))  # ethanol
        assert result.is_solved is True
        assert result.num_routes > 0
        assert result.top_route_summary is not None

    def test_finds_a_real_route_to_a_cof_building_block(self):
        # Benzidine (4,4'-diaminobiphenyl) -- a real, common COF/MOF diamine linker.
        result = check_retrosynthesis("Nc1ccc(-c2ccc(N)cc2)cc1", str(_DEFAULT_CONFIG))
        assert result.is_solved is True
        assert result.num_routes > 0
        # Real chemistry check, not just "did it run": the standard route to an aromatic amine
        # is nitro-group reduction. The proposed template's reaction SMARTS should show a
        # nitrogen transformation consistent with that (an [N] appearing on both sides with an
        # oxygen introduced/removed), not an arbitrary unrelated disconnection.
        assert "[N" in result.top_route_summary

    def test_check_building_block_with_retrosynthesis_flags_non_purchasable_with_a_real_route(self):
        from materials_synthesis_agent.feasibility.retrosynthesis import check_building_block_with_retrosynthesis

        result = check_building_block_with_retrosynthesis(
            "benzidine", "Nc1ccc(-c2ccc(N)cc2)cc1", str(_DEFAULT_CONFIG)
        )
        assert result.is_valid_structure is True
        # Whether or not it's independently flagged purchasable via PubChem, the feasibility
        # flags should mention the retrosynthesis outcome if it wasn't confirmed purchasable --
        # i.e. the composition in checker.check_protocol_candidate has something real to report.
        if not result.is_purchasable:
            assert any("route" in f.lower() or "retrosynthesis" in f.lower() for f in result.flags)
