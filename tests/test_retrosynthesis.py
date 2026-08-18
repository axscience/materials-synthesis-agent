"""Tests against the real installed aizynthfinder package (not a fake) -- skipped cleanly when
the optional `retrosynthesis` extra isn't installed. This only checks the API shape this module
depends on hasn't shifted; it does NOT run a real tree search (that needs `download_public_data`'s
multi-GB model files, not available in CI or most dev environments). See
feasibility/retrosynthesis.py's module docstring for what is and isn't verified.
"""

import pytest

aizynthfinder = pytest.importorskip("aizynthfinder")

from materials_synthesis_agent.feasibility.retrosynthesis import RetrosynthesisUnavailable, check_retrosynthesis


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
