# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An integer tensor is a value: walked once when it is made, not per configuration.

``INTEGER_TENSOR`` snapshots nested int tuples into an ``IntegerTensorValue``,
whose shape is checked at construction and whose integers and range are
computed once; recognizing it again is a type check. So configuring a point
that holds weights does not re-walk them.
"""

from __future__ import annotations

import copy
import pickle

import pytest
from kernels.helpers import matmul_point
from qonnx.core.datatype import DataType

from finn.core.space import inspection
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.datatypes import semantics
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    IntegerTensorValue,
    integer_range,
    integers,
)
from finn.kernels.target import DspBlock

WEIGHTS = ((1, -2, 3), (4, 5, -6))


def test_the_snapshot_is_a_value_made_once() -> None:
    value = INTEGER_TENSOR.freeze(WEIGHTS)
    assert type(value) is IntegerTensorValue and value == WEIGHTS
    assert value.shape == (2, 3)
    assert value.integers == (1, -2, 3, 4, 5, -6) and integers(value) is value.integers
    assert value.range == (-6, 5) and integer_range(value) == (-6, 5)
    # Freezing the value again, copying or deep-copying it, is the value itself.
    assert INTEGER_TENSOR.freeze(value) is value
    assert copy.copy(value) is value and copy.deepcopy(value) is value
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is IntegerTensorValue and restored == value
    assert INTEGER_TENSOR.values_equal(value, WEIGHTS)
    with pytest.raises(AttributeError, match="immutable"):
        value.shape = (6,)


@pytest.mark.parametrize("wrong", ((), ((1, 2), (3,)), [[1]], ((True,),), ((1.0,),), 3))
def test_what_is_no_integer_tensor_is_refused(wrong: object) -> None:
    assert not INTEGER_TENSOR.accepts(wrong)
    with pytest.raises(TypeError):
        IntegerTensorValue(wrong)


def test_a_new_configuration_does_not_walk_the_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """The replay cost of phase 1's record: 700 ms per configuration of a 512 x 512
    point, all of it recognizing the same weights again."""
    int3 = DataType["INT3"]
    weights = tuple(tuple((r + c) % 3 - 1 for c in range(4)) for r in range(4))
    base = matmul_point(
        m=3,
        n=4,
        k=4,
        activation_dtype=int3,
        weights_dtype=int3,
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
        weights=weights,
    )
    assert base.matmul.weight_tensor == Tensor((4, 4), ScalarEncoding(int3, (-1, 1)))
    handles = {item.key: item.reference for item in inspection.decisions(base)}
    walks: list[object] = []
    walk = semantics._shape
    monkeypatch.setattr(semantics, "_shape", lambda value: walks.append(value) or walk(value))
    point = base.with_choices({handles["matmul.compute"]: "packed"})
    point = point.with_choices({handles["w.source.memstream.ram_style"]: "block"})
    assert point.matmul.weight_tensor.element.value_range == (-1, 1)
    assert point.w.source.value_range == (-1, 1)
    assert walks == []
