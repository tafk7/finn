# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An initializer's weights as an integer tensor value: stated from the array (shape,
range, digest), the same value as its nested integers, loaded only when read."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from finn.custom_op.kernels.base import integer_tensor
from finn.kernels.values.semantics import IntegerTensorValue


@pytest.mark.parametrize("dtype", (np.float32, np.int8, np.int64))
def test_the_array_states_the_value_of_its_integers(dtype: type) -> None:
    values: npt.NDArray[Any] = np.random.default_rng(0).integers(-7, 8, size=(5, 3)).astype(dtype)
    stated = integer_tensor(values)
    assert stated._integers is None  # nothing loaded yet
    nested = IntegerTensorValue.of(tuple(tuple(int(v) for v in row) for row in values))
    assert stated == nested and stated.digest == nested.digest
    assert stated.shape == (5, 3) and stated.range == nested.range
    assert stated.integers == nested.integers


def test_integers_beyond_64_bits_are_the_same_value() -> None:
    values = np.array([[2.0**64, -1.0]])
    assert integer_tensor(values) == IntegerTensorValue.of(((2**64, -1),))
