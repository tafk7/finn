# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An initializer's weights as an integer tensor value: stated from the array (shape,
range, digest), the same value as its nested integers, loaded only when read."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from finn.custom_op.kernels.base import integer_tensor, kernel_op
from finn.kernels.values.semantics import IntegerTensorValue
from kernel_ops.models import WEIGHTS, matmul_model


@pytest.mark.parametrize("dtype", (np.float32, np.int8, np.int64))
def test_the_array_states_the_value_of_its_integers(dtype: type) -> None:
    values: npt.NDArray[Any] = np.random.default_rng(0).integers(-7, 8, size=(5, 3)).astype(dtype)
    stated = integer_tensor(values)
    assert stated._integers is None  # nothing loaded yet
    nested = IntegerTensorValue.of(tuple(tuple(int(v) for v in row) for row in values))
    assert stated == nested and stated.digest == nested.digest
    assert stated.shape == (5, 3) and stated.range == nested.range
    # A memory image reads them as an int64 array: no Python int is made.
    array = stated.row_major
    assert isinstance(array, np.ndarray) and array.tolist() == list(nested.integers)
    assert stated._integers is None
    assert stated.integers == nested.integers


def test_integers_beyond_64_bits_are_the_same_value() -> None:
    values = np.array([[2.0**64, -1.0]])
    assert integer_tensor(values) == IntegerTensorValue.of(((2**64, -1),))


def test_equal_integers_stored_differently_are_one_value_under_two_binding_keys() -> None:
    """A stored operand's identity is its shape and integers (FINN's integer digest), so a
    float32 and an int8 initializer of the same integers are one value. The bind cache
    keys an initializer by QONNX's ``content_digest`` of the stored bytes, so each binds
    once."""
    floats = matmul_model(infer=False)
    integers = matmul_model(infer=False)
    integers.set_initializer("w", np.asarray(WEIGHTS, dtype=np.int8))
    facts = [kernel_op(model, model.graph.node[0]).facts() for model in (floats, integers)]
    assert facts[0].values()["w"] == facts[1].values()["w"]
    assert facts[0].key != facts[1].key
