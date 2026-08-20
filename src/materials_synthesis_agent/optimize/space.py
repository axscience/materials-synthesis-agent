"""Parameter-space definition and encoding between the abstract protocol-parameter space and the
tensors BoTorch needs.

Deliberately reusable, not COF-specific: a ParameterSpec list is passed in
by the caller, nothing here hardcodes temperature/solvent/etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch

ParamValue = Union[float, str]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: Literal["continuous", "categorical"]
    bounds: tuple[float, float] | None = None  # required if continuous
    categories: tuple[str, ...] | None = None  # required if categorical

    def __post_init__(self):
        if self.kind == "continuous" and self.bounds is None:
            raise ValueError(f"Continuous parameter '{self.name}' needs bounds=(low, high).")
        if self.kind == "categorical" and not self.categories:
            raise ValueError(f"Categorical parameter '{self.name}' needs a non-empty categories tuple.")


class ParameterSpace:
    """Encodes a dict of {name: value} into a flat tensor (one-hot for categorical dimensions)
    and back. This is the mixed-variable encoding BoTorch's MixedSingleTaskGP expects: continuous
    dims stay as-is, categorical dims become one-hot blocks, and `categorical_feature_indices`
    tells the model which output columns are categorical.
    """

    def __init__(self, specs: list[ParameterSpec]):
        if not specs:
            raise ValueError("ParameterSpace needs at least one ParameterSpec.")
        self.specs = specs
        self._offsets: dict[str, int] = {}
        self._widths: dict[str, int] = {}
        offset = 0
        for spec in specs:
            self._offsets[spec.name] = offset
            width = 1 if spec.kind == "continuous" else len(spec.categories)
            self._widths[spec.name] = width
            offset += width
        self.dim = offset

    @property
    def categorical_feature_indices(self) -> list[int]:
        """Column indices of one-hot categorical blocks, required by MixedSingleTaskGP."""
        idxs = []
        for spec in self.specs:
            if spec.kind == "categorical":
                idxs.extend(range(self._offsets[spec.name], self._offsets[spec.name] + self._widths[spec.name]))
        return idxs

    @property
    def bounds(self) -> torch.Tensor:
        """2 x dim tensor of (lower, upper) bounds -- one-hot categorical dims are bounded [0, 1]."""
        lower = torch.zeros(self.dim, dtype=torch.double)
        upper = torch.ones(self.dim, dtype=torch.double)
        for spec in self.specs:
            if spec.kind == "continuous":
                lo, hi = spec.bounds
                lower[self._offsets[spec.name]] = lo
                upper[self._offsets[spec.name]] = hi
        return torch.stack([lower, upper])

    def encode(self, params: dict[str, ParamValue]) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.double)
        for spec in self.specs:
            value = params.get(spec.name)
            if value is None:
                raise KeyError(f"Missing value for parameter '{spec.name}'.")
            offset = self._offsets[spec.name]
            if spec.kind == "continuous":
                vec[offset] = float(value)
            else:
                if value not in spec.categories:
                    raise ValueError(f"'{value}' is not a valid category for '{spec.name}' (choices: {spec.categories}).")
                vec[offset + spec.categories.index(value)] = 1.0
        return vec

    def decode(self, vec: torch.Tensor) -> dict[str, ParamValue]:
        out: dict[str, ParamValue] = {}
        for spec in self.specs:
            offset = self._offsets[spec.name]
            if spec.kind == "continuous":
                out[spec.name] = float(vec[offset].item())
            else:
                block = vec[offset : offset + self._widths[spec.name]]
                out[spec.name] = spec.categories[int(torch.argmax(block).item())]
        return out

    def encode_batch(self, param_dicts: list[dict[str, ParamValue]]) -> torch.Tensor:
        return torch.stack([self.encode(p) for p in param_dicts])

    def fixed_features_list(self) -> list[dict[int, float]]:
        """One dict per combination of categorical values (Cartesian product across categorical
        specs), fixing every one-hot column for those specs -- the shape `optimize_acqf_mixed`
        needs to search categorical combinations exhaustively while continuous-optimizing within
        each. Returns [{}] (no fixed features) if there are no categorical parameters."""
        import itertools

        cat_specs = [s for s in self.specs if s.kind == "categorical"]
        if not cat_specs:
            return [{}]
        choices_per_spec = [range(len(s.categories)) for s in cat_specs]
        combos = []
        for choice_indices in itertools.product(*choices_per_spec):
            fixed: dict[int, float] = {}
            for spec, chosen_idx in zip(cat_specs, choice_indices):
                offset = self._offsets[spec.name]
                for i in range(len(spec.categories)):
                    fixed[offset + i] = 1.0 if i == chosen_idx else 0.0
            combos.append(fixed)
        return combos
