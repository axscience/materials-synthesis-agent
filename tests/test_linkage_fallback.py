"""Tests for hierarchical linkage-chemistry fallback: the search tiers, and generate_protocols'
confirm-before-expanding + provenance-tagging behavior."""

import materials_synthesis_agent.literature.agent as agent
from materials_synthesis_agent.literature.linkage_fallback import (
    base_linkage,
    build_search_tiers,
    related_linkages,
)
from materials_synthesis_agent.schema import Citation, FieldValue, ProtocolCandidate, ProtocolSource, Target


def _target(name=None, linkage="imine condensation"):
    return Target(
        name=name, functional_groups=["imine"], linkage_chemistry=linkage,
        application="CO2 capture", metric_name="crystallinity", metric_measurement_method="PXRD",
    )


def test_base_linkage_normalizes_verbose_chemistry():
    assert base_linkage("imine condensation") == "imine"
    assert base_linkage("boronate ester formation") == "boronate ester"
    assert base_linkage("some exotic linkage") is None


def test_tiers_named_target_is_exact_then_same_then_related():
    tiers = build_search_tiers(_target(name="COF-LZU1"))
    assert "exact COF" in tiers[0].label and tiers[0].beyond_target_linkage is False
    assert "same linkage" in tiers[1].label and tiers[1].beyond_target_linkage is False
    assert tiers[2].beyond_target_linkage is True  # first related-linkage tier
    # related linkages are the imine family, most similar first
    assert related_linkages(_target()) == ["hydrazone", "azine", "beta-ketoenamine", "imide"]


def test_unnamed_target_has_no_exact_tier():
    tiers = build_search_tiers(_target(name=None))
    assert all("exact COF" not in t.label for t in tiers)
    assert tiers[0].beyond_target_linkage is False  # same-linkage tier first


def test_unknown_linkage_offers_no_related_tiers():
    tiers = build_search_tiers(_target(linkage="some exotic linkage"))
    assert all(t.beyond_target_linkage is False for t in tiers)
    assert related_linkages(_target(linkage="some exotic linkage")) == []


def _install_fakes(monkeypatch, hits_predicate):
    """Fake search returns one paper only for queries where hits_predicate(query) is True; fake
    extraction always yields a candidate."""
    from materials_synthesis_agent.literature.retrieval import Paper

    def fake_search(query, limit=10):
        if hits_predicate(query):
            return [Paper(source_id="10.x/y", title=f"paper for {query[:20]}", abstract="...", year=2024, url="http://x", source="openalex")]
        return []

    def fake_extract(target, paper, **kw):
        return ProtocolCandidate(
            target_id=target.id, source=ProtocolSource.LITERATURE,
            solvent=FieldValue(value="dioxane", citation=Citation(source_id=paper.source_id, title=paper.title)),
        )

    monkeypatch.setattr(agent, "search", fake_search)
    monkeypatch.setattr(agent, "extract_protocol", fake_extract)


def test_stays_within_linkage_when_expansion_declined(monkeypatch):
    # Only related-linkage queries have hits; target-linkage tiers are empty -> must ask, and on
    # decline return nothing rather than silently using the related-linkage papers.
    _install_fakes(monkeypatch, hits_predicate=lambda q: not q.startswith("imine") and "COF-X" not in q)
    got = agent.generate_protocols(_target(name="COF-X"), n=3, confirm_expand=lambda alts: False)
    assert got == []


def test_no_confirmer_never_expands(monkeypatch):
    _install_fakes(monkeypatch, hits_predicate=lambda q: not q.startswith("imine") and "COF-X" not in q)
    got = agent.generate_protocols(_target(name="COF-X"), n=3, confirm_expand=None)
    assert got == []


def test_expands_and_tags_provenance_when_confirmed(monkeypatch):
    _install_fakes(monkeypatch, hits_predicate=lambda q: not q.startswith("imine") and "COF-X" not in q)
    offered = {}
    got = agent.generate_protocols(
        _target(name="COF-X"), n=2, confirm_expand=lambda alts: offered.setdefault("alts", alts) or True
    )
    assert offered["alts"] == ["hydrazone", "azine", "beta-ketoenamine", "imide"]
    assert len(got) == 2
    assert all("DIFFERENT linkage" in c.provenance_note for c in got)


def test_same_linkage_hit_satisfying_n_never_triggers_expansion(monkeypatch):
    # Same-linkage tier yields enough to satisfy n -> confirm_expand must never be called, and the
    # candidate is not flagged a weaker prior. (n=1, and the fake same-linkage tier returns 1 paper.)
    _install_fakes(monkeypatch, hits_predicate=lambda q: q.startswith("imine"))
    called = {"expand": False}

    def confirm(alts):
        called["expand"] = True
        return True

    got = agent.generate_protocols(_target(name=None), n=1, confirm_expand=confirm)
    assert called["expand"] is False
    assert len(got) == 1
    assert "DIFFERENT linkage" not in (got[0].provenance_note or "")


def test_partial_same_linkage_asks_before_filling_from_related(monkeypatch):
    # 1 same-linkage paper but n=2 -> the agent legitimately asks to expand to fill the gap.
    _install_fakes(monkeypatch, hits_predicate=lambda q: True)  # every tier has a paper
    asked = {"n": 0}
    got = agent.generate_protocols(
        _target(name=None), n=2, confirm_expand=lambda alts: (asked.__setitem__("n", asked["n"] + 1), True)[1]
    )
    assert asked["n"] == 1  # asked exactly once, before the first related tier
    assert len(got) == 2
    # first from target linkage, second from a related one (weaker prior)
    assert "DIFFERENT linkage" not in (got[0].provenance_note or "")
    assert "DIFFERENT linkage" in got[1].provenance_note
