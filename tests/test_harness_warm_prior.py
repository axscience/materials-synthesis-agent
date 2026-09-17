"""Cross-campaign warm prior -- the SQL flywheel. Torch-free, so it runs fast."""

import pytest

from materials_synthesis_agent.harness.warm_prior import (
    PriorStore,
    build_warm_prior,
)


def _seed(store):
    # imine / crystallinity experiments across two past campaigns
    store.record("imine", "crystallinity", {"temperature_c": 120, "time_hours": 72, "solvent": "mesitylene"}, 0.65, "camp-1")
    store.record("imine", "crystallinity", {"temperature_c": 130, "time_hours": 72, "solvent": "mesitylene"}, 0.82, "camp-1")
    store.record("imine", "crystallinity", {"temperature_c": 110, "time_hours": 48, "solvent": "dioxane"}, 0.40, "camp-2")
    # a different chemistry that must NOT leak into imine queries
    store.record("boronate_ester", "crystallinity", {"temperature_c": 85, "time_hours": 24, "solvent": "dioxane"}, 0.90, "camp-3")


def test_empty_prior(tmp_path):
    store = PriorStore(tmp_path / "priors.db")
    wp = build_warm_prior(store, "imine", "crystallinity")
    assert wp.is_empty
    assert wp.anchor_params == []
    store.close()


def test_best_conditions_first_and_isolated_by_chemistry(tmp_path):
    store = PriorStore(tmp_path / "priors.db")
    _seed(store)
    wp = build_warm_prior(store, "imine", "crystallinity", maximize=True)

    assert wp.n_prior_experiments == 3            # boronate row excluded
    # Highest crystallinity (0.82 at 130C) ranks first.
    assert wp.anchor_params[0]["temperature_c"] == 130
    assert "camp-3" not in wp.source_campaigns    # no cross-chemistry leak
    store.close()


def test_anchors_filtered_to_space_params(tmp_path):
    store = PriorStore(tmp_path / "priors.db")
    _seed(store)
    # A space that also needs 'modulator_equiv', which none of the past rows have -> no anchors,
    # but the rows still count toward n and the numeric priors.
    wp = build_warm_prior(
        store, "imine", "crystallinity",
        required_params={"temperature_c", "time_hours", "solvent", "modulator_equiv"},
    )
    assert wp.anchor_params == []
    assert wp.n_prior_experiments == 3
    assert "temperature_c" in wp.parameter_priors
    store.close()


def test_invalid_categorical_excluded_from_anchors(tmp_path):
    store = PriorStore(tmp_path / "priors.db")
    _seed(store)
    # Only 'dioxane' is a valid solvent in this space -> only the 110C/dioxane row can anchor.
    wp = build_warm_prior(
        store, "imine", "crystallinity",
        required_params={"temperature_c", "time_hours", "solvent"},
        categorical_values={"solvent": {"dioxane"}},
    )
    assert len(wp.anchor_params) == 1
    assert wp.anchor_params[0]["solvent"] == "dioxane"
    store.close()


def test_parameter_priors_mean_and_std(tmp_path):
    store = PriorStore(tmp_path / "priors.db")
    _seed(store)
    wp = build_warm_prior(store, "imine", "crystallinity")
    temp = wp.parameter_priors["temperature_c"]
    assert temp.n_observations == 3
    assert temp.mean == pytest.approx((120 + 130 + 110) / 3)
    assert temp.std > 0
    store.close()


def test_minimize_direction_ranks_low_first(tmp_path):
    store = PriorStore(tmp_path / "priors.db")
    store.record("imine", "particle_size", {"temperature_c": 120}, 300.0, "c1")
    store.record("imine", "particle_size", {"temperature_c": 90}, 120.0, "c1")
    wp = build_warm_prior(store, "imine", "particle_size", maximize=False)
    assert wp.anchor_params[0]["temperature_c"] == 90  # lowest particle size first
    store.close()
