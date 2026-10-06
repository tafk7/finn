# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An integer tensor is a value its producer states: shape, range and digest; its
integers are loaded when first read.

``INTEGER_TENSOR`` snapshots nested int tuples into an ``IntegerTensorValue`` by
one walk; a producer holding the integers in another form states the facts and
how to load them. Recognizing a value again is a type check, equality compares
digests, and configuring a point that holds weights reads only the stated facts:
the integers are loaded when a memory image is packed.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.space import inspection
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    IntegerTensorValue,
    integer_bytes,
    integer_digest,
    integer_range,
    integers,
    row_major,
)
from kernels.helpers import FULL_DSP48E2, matmul_point

WEIGHTS = ((1, -2, 3), (4, 5, -6))


def test_the_snapshot_is_a_value_made_once() -> None:
    value = INTEGER_TENSOR.freeze(WEIGHTS)
    assert type(value) is IntegerTensorValue and value == IntegerTensorValue.of(WEIGHTS)
    assert value.shape == (2, 3)
    assert value.integers == (1, -2, 3, 4, 5, -6) and integers(value) is value.integers
    assert value.range == (-6, 5) and integer_range(value) == (-6, 5)
    assert value.digest == integer_digest((2, 3), integer_bytes((1, -2, 3, 4, 5, -6)))
    # Freezing the value again, copying or deep-copying it, is the value itself.
    assert INTEGER_TENSOR.freeze(value) is value
    assert copy.copy(value) is value and copy.deepcopy(value) is value
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is IntegerTensorValue and restored == value
    assert INTEGER_TENSOR.values_equal(value, WEIGHTS)
    assert not INTEGER_TENSOR.values_equal(value, ((1, -2, 3), (4, 5, -7)))
    with pytest.raises(AttributeError, match="immutable"):
        value.shape = (6,)


def test_the_digest_is_of_shape_and_integers() -> None:
    assert IntegerTensorValue.of(((1, 2),)) != IntegerTensorValue.of(((1,), (2,)))
    assert IntegerTensorValue.flat((2, 1), (1, 2)) == IntegerTensorValue.of(((1,), (2,)))
    # Beyond 64 bits the integers are digested as text: still a value.
    wide = IntegerTensorValue.of(((2**70, -(2**70)),))
    assert wide.range == (-(2**70), 2**70) and wide.integers == (2**70, -(2**70))


@pytest.mark.parametrize("wrong", ((), ((1, 2), (3,)), [[1]], ((True,),), ((1.0,),), 3))
def test_what_is_no_integer_tensor_is_refused(wrong: object) -> None:
    assert not INTEGER_TENSOR.accepts(wrong)
    with pytest.raises(TypeError):
        IntegerTensorValue.of(wrong)


def test_a_stated_value_loads_its_integers_once_and_checks_them() -> None:
    loads: list[int] = []

    def load() -> tuple[int, ...]:
        loads.append(1)
        return (1, -2, 3, 4, 5, -6)

    digest = integer_digest((2, 3), integer_bytes((1, -2, 3, 4, 5, -6)))
    stated = IntegerTensorValue((2, 3), (-6, 5), digest, load)
    assert stated == IntegerTensorValue.of(WEIGHTS) and loads == []
    assert stated.integers == (1, -2, 3, 4, 5, -6) and stated.integers and loads == [1]
    wrong = IntegerTensorValue((2, 3), (-6, 6), digest, load)
    with pytest.raises(ValueError, match="loaded other integers"):
        wrong.integers


def test_a_memory_image_reads_a_read_only_int64_array_up_to_64_bits() -> None:
    """``row_major`` is what ``pack`` reads: an int64 array, loaded once and checked as
    ``integers`` is, the integers read from it; beyond 64 bits, ``integers``."""
    loads: list[int] = []

    def load() -> tuple[int, ...]:
        loads.append(1)
        return (1, -2, 3, 4, 5, -6)

    digest = integer_digest((2, 3), integer_bytes((1, -2, 3, 4, 5, -6)))
    stated = IntegerTensorValue((2, 3), (-6, 5), digest, load)
    array = row_major(stated)
    assert isinstance(array, np.ndarray) and array.dtype == np.int64 and array.shape == (6,)
    assert not array.flags.writeable and array.tolist() == [1, -2, 3, 4, 5, -6]
    assert stated.integers == (1, -2, 3, 4, 5, -6) and stated.row_major is array
    assert loads == [1] and all(type(item) is int for item in stated.integers)
    nested = row_major(WEIGHTS)
    assert isinstance(nested, np.ndarray) and nested.tolist() == [1, -2, 3, 4, 5, -6]
    edges = IntegerTensorValue.of(((2**63 - 1, -(2**63)),)).row_major
    assert isinstance(edges, np.ndarray) and edges.tolist() == [2**63 - 1, -(2**63)]
    wide = IntegerTensorValue.of(((2**63, -1),))
    assert wide.row_major is wide.integers
    misstated: tuple[tuple[object, ...], ...] = (
        (1, -2, 3, 4, 5),
        (1, -2, 3, 4, 5, -7),
        (1, -2, 3, 4, 2**63, -6),
        ((1, -2, 3), (4, 5, -6)),
    )
    for loaded in misstated:
        wrong = IntegerTensorValue((2, 3), (-6, 5), digest, _loading(loaded))
        with pytest.raises(ValueError, match="loaded other integers"):
            wrong.row_major


def _loading(found: tuple[object, ...]) -> Callable[[], tuple[int, ...]]:
    """A load that returns ``found``, whatever it holds."""
    return lambda: cast(tuple[int, ...], found)


def test_a_new_configuration_reads_only_the_stated_facts() -> None:
    """Configuring a point reads the weights' shape and range, and only the memory image
    their integers: at 512 x 512, recognizing the same weights again cost about 700 ms a
    configuration."""
    int3 = DataType["INT3"]
    nested = tuple(tuple((r + c) % 3 - 1 for c in range(4)) for r in range(4))
    loads: list[int] = []
    flat = IntegerTensorValue.of(nested).integers

    def load() -> tuple[int, ...]:
        loads.append(1)
        return flat

    weights = IntegerTensorValue((4, 4), (-1, 1), integer_digest((4, 4), integer_bytes(flat)), load)
    base = matmul_point(
        m=3,
        n=4,
        k=4,
        activation_dtype=int3,
        weights_dtype=int3,
        platform=FULL_DSP48E2,
        weights=weights,
    )
    assert base.matmul.weight_tensor == Tensor((4, 4), ScalarEncoding(int3, (-1, 1)))
    handles = {item.key: item.reference for item in inspection.decisions(base)}
    point = base.with_choices({handles["matmul.compute"]: "packed"})
    point = point.with_choices({handles["w.source.memstream.ram_style"]: "block"})
    assert point.matmul.weight_tensor.element.value_range == (-1, 1)
    assert point.w.source.value_range == (-1, 1)
    assert loads == []
    folded = point.with_choices(
        {handles["matmul.compute.packed.pe"]: 2, handles["matmul.compute.packed.simd"]: 2}
    )
    assert folded.w.source.image and loads == [1]
