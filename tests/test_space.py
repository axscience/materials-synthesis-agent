import pytest
import torch

from materials_synthesis_agent.optimize.space import ParameterSpace, ParameterSpec


@pytest.fixture
def space():
    return ParameterSpace(
        [
            ParameterSpec("temperature_c", "continuous", bounds=(20, 150)),
            ParameterSpec("time_hours", "continuous", bounds=(1, 96)),
            ParameterSpec("solvent", "categorical", categories=("dioxane", "mesitylene", "DMAc")),
        ]
    )


def test_dim_accounts_for_one_hot_categorical(space):
    assert space.dim == 2 + 3  # 2 continuous + 3 one-hot categories


def test_encode_decode_round_trip(space):
    p = {"temperature_c": 120, "time_hours": 72, "solvent": "mesitylene"}
    assert space.decode(space.encode(p)) == {"temperature_c": 120.0, "time_hours": 72.0, "solvent": "mesitylene"}


def test_encode_rejects_invalid_category(space):
    with pytest.raises(ValueError):
        space.encode({"temperature_c": 100, "time_hours": 10, "solvent": "not_a_real_solvent"})


def test_encode_rejects_missing_field(space):
    with pytest.raises(KeyError):
        space.encode({"temperature_c": 100, "time_hours": 10})


def test_bounds_are_zero_one_for_categorical_dims(space):
    bounds = space.bounds
    cat_idx = space.categorical_feature_indices
    for i in cat_idx:
        assert bounds[0, i].item() == 0.0
        assert bounds[1, i].item() == 1.0


def test_fixed_features_list_covers_every_category_combination(space):
    combos = space.fixed_features_list()
    assert len(combos) == 3  # only one categorical spec with 3 categories
    for combo in combos:
        cat_idx = space.categorical_feature_indices
        assert sum(combo[i] for i in cat_idx) == 1.0  # exactly one category "on" per combo


def test_fixed_features_list_empty_when_no_categorical():
    space = ParameterSpace([ParameterSpec("x", "continuous", bounds=(0, 1))])
    assert space.fixed_features_list() == [{}]


def test_parameter_spec_requires_bounds_for_continuous():
    with pytest.raises(ValueError):
        ParameterSpec("x", "continuous")


def test_parameter_spec_requires_categories_for_categorical():
    with pytest.raises(ValueError):
        ParameterSpec("x", "categorical")
