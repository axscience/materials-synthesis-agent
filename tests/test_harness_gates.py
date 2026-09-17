"""Gates that replace the Critic Agent -- deterministic, blocking vs advisory."""

import pytest

from materials_synthesis_agent.harness.gates import Gate, run_gates, validate_smiles


def test_suggestion_without_uncertainty_is_blocked():
    g = run_gates("opt.suggest_next", {"predicted_value": 1840.0, "predicted_uncertainty": None})
    assert g.blocked is True
    assert g.severity == "error"


def test_suggestion_with_zero_uncertainty_is_blocked():
    g = run_gates("opt.suggest_next", {"predicted_value": 1840.0, "predicted_uncertainty": 0.0})
    assert g.blocked is True


def test_suggestion_with_uncertainty_passes():
    g = run_gates("opt.suggest_next", {"predicted_value": 1840.0, "predicted_uncertainty": 55.0})
    assert g.blocked is False
    assert g.severity == "ok"


def test_unknown_tool_is_ungated():
    assert run_gates("lit.search", {"anything": True}).blocked is False


def test_feasibility_flags_are_advisory_not_blocking():
    g = run_gates("feas.check", {"flags": ["exotic_linker_smiles"]})
    assert g.blocked is False
    assert g.severity == "warning"
    assert "exotic_linker_smiles" in g.message


def test_extraction_valid_smiles_passes():
    pytest.importorskip("rdkit")
    out = {"building_blocks": {
        "TAPB": {"value": "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1"},
        "PDA": {"value": "O=Cc1ccc(C=O)cc1"},
    }}
    assert run_gates("lit.extract_protocol", out).blocked is False


def test_extraction_invalid_smiles_is_blocked():
    pytest.importorskip("rdkit")
    out = {"building_blocks": {"BAD": {"value": "this is not smiles ((("}}}
    g = run_gates("lit.extract_protocol", out)
    assert g.blocked is True
    assert "BAD" in g.detail["invalid_blocks"]


def test_validate_smiles_degrades_gracefully_without_rdkit(monkeypatch):
    # Simulate RDKit import failure -> validate_smiles must not block (returns True = "not checked").
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rdkit"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert validate_smiles("literally anything") is True
