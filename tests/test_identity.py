from __future__ import annotations

import pickle

import numpy as np
import pytest

from pyscf_vscf._arrays import immutable_array
from pyscf_vscf._identity import (
    canonical_json,
    float64_array_identity,
    immutable_json_mapping,
    payload_fingerprint,
)


def test_canonical_json_and_payload_fingerprint_are_order_independent() -> None:
    left = {"b": [2, 3], "a": {"value": 1.0}}
    right = {"a": {"value": 1.0}, "b": (2, 3)}

    assert canonical_json(left) == canonical_json(right)
    assert payload_fingerprint(left) == payload_fingerprint(right)


def test_float64_array_identity_uses_exact_bytes_and_shape() -> None:
    values = np.array([[1.0, -0.0], [2.0, 3.0]], dtype=">f8")
    same_values = values.astype("<f8")
    changed = same_values.copy()
    changed[0, 1] = 0.0

    assert float64_array_identity(values) == float64_array_identity(same_values)
    assert float64_array_identity(changed) != float64_array_identity(same_values)
    assert float64_array_identity(same_values.reshape(4)) != float64_array_identity(same_values)


def test_immutable_arrays_and_json_survive_pickle() -> None:
    array = immutable_array(np.array([1.0, 2.0]))
    mapping = immutable_json_mapping({"nested": {"items": [1, 2]}})

    restored_array, restored_mapping = pickle.loads(pickle.dumps((array, mapping)))

    with pytest.raises(ValueError):
        restored_array.setflags(write=True)
    with pytest.raises(TypeError):
        restored_mapping["new"] = 3  # type: ignore[index]
    assert tuple(restored_mapping["nested"]["items"]) == (1, 2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_identity_values_fail_closed(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": value})
    with pytest.raises(ValueError):
        float64_array_identity([value])
