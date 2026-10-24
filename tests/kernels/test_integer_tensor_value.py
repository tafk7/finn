# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An integer tensor is a value its producer states: shape, range and digest; its
integers are loaded when first read.

A fact or choice annotated ``IntegerTensorValue`` takes the value, never a nest:
nested int tuples make one by one walk (``IntegerTensorValue.of``); a producer
holding the integers in another form states the facts and how to load them.
Recognizing a value is a type check, equality compares digests, and configuring a
point that holds weights reads only the stated facts: the integers are loaded when
a memory image is packed.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.core.ranges import TensorRange
from support.helpers import FULL_DSP48E2, choose, matmul_point

from finn.core.space import DefinitionError, Param, RequestError, Space, derived, design_space
from finn.core.space.occurrence import held
from finn.core.space.semantics import canonical
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.values.semantics import (
    IntegerTensorValue,
    integer_bytes,
    integer_digest,
    integer_range,
    integers,
    row_major,
)

WEIGHTS = ((1, -2, 3), (4, 5, -6))


def test_a_nest_makes_a_value_once() -> None:
    value = IntegerTensorValue.of(WEIGHTS)
    assert type(value) is IntegerTensorValue and value == IntegerTensorValue.of(WEIGHTS)
    assert value.shape == (2, 3)
    assert value.integers == (1, -2, 3, 4, 5, -6) and integers(value) is value.integers
    assert value.range == (-6, 5) and integer_range(value) == (-6, 5)
    assert value.digest == integer_digest((2, 3), integer_bytes((1, -2, 3, 4, 5, -6)))
    # Its canonical form, a copy or a deep copy of it, is the value itself.
    assert canonical(value) is value
    assert copy.copy(value) is value and copy.deepcopy(value) is value
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is IntegerTensorValue and restored == value
    # Equal by digest to the value of the same nest, to no other, and never to a nest.
    assert value != IntegerTensorValue.of(((1, -2, 3), (4, 5, -7)))
    assert value != WEIGHTS
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


def test_a_value_keeps_its_column_sums() -> None:
    """Each column's positive and negative sums, over the next-to-last axis, per set."""
    value = IntegerTensorValue.of(((1, -2), (3, 4), (-5, 0)))
    assert value.columns == ((4, -5), (4, -2))
    assert value.columns is value.columns  # computed once
    sets = IntegerTensorValue.of((((1, -1),), ((-2, 2),)))  # two sets of (1, 2)
    assert sets.columns == ((1, 0), (0, -1), (0, -2), (2, 0))
    assert IntegerTensorValue.of((2, -3, 4)).columns == ((6, -3),)  # a vector: one column
    # Sums past 64 bits are exact too.
    assert IntegerTensorValue.of(((2**62, -1), (2**62, 3))).columns == ((2**63, 0), (3, -1))


def test_a_new_configuration_reads_only_the_stated_facts() -> None:
    """Configuring a point reads the weights' shape and range, and their integers once:
    for the column sums MatMul's result range derives from, which the value keeps,
    and the memory image reads the same load. At 512 x 512, recognizing the same weights
    again cost about 700 ms a configuration."""
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
    # The weight channel's tensor is stated over the value's range, in the encoding it
    # needs (INT2); MatMul reads INT3. Its contracts wait on MatMul's core and folding.
    assert base.member("matmul").weight_tensor == Tensor((4, 4), ScalarEncoding(int3))
    point = choose(base, {"matmul.compute": "packed"})
    folded = choose(point, {"matmul.pe": 2, "matmul.simd": 2})
    folded = choose(folded, {"w.source.ram_style": "block"})
    w = folded.member("w")
    assert w.tensor == Tensor((4, 4), ScalarEncoding(DataType["INT2"], TensorRange(-1, 1)))
    assert w.source.value_range == (-1, 1)
    # The result's range read the columns (over INT3 activations: [-11, 11]), once.
    matmul = folded.member("matmul")
    assert matmul.y_tensor.element == ScalarEncoding(DataType["INT5"], TensorRange(-11, 11))
    assert loads == [1]
    assert w.source.image and loads == [1]


class Held(Space):
    """An optional integer tensor (a ``T | None`` annotation): ``None``, or the value."""

    contents: IntegerTensorValue | None = Param(default=None)

    @derived
    def rows(self) -> int:
        return 0 if self.contents is None else integer_shape_of(self.contents)[0]


def integer_shape_of(contents: object) -> tuple[int, ...]:
    assert type(contents) is IntegerTensorValue
    return contents.shape


def test_an_optional_integer_tensor_left_out_given_none_and_given() -> None:
    left_out, given_none = design_space(Held()), design_space(Held(contents=None))
    assert left_out.contents is None and given_none.contents is None
    assert left_out.rows == 0 and given_none.rows == 0
    assert held(left_out).ds.layer is held(given_none).ds.layer
    assert held(left_out).ds == held(given_none).ds
    given = design_space(Held(contents=IntegerTensorValue.of(WEIGHTS)))
    assert given.contents == IntegerTensorValue.of(WEIGHTS) and given.rows == 2
    assert held(given).ds.layer is not held(left_out).ds.layer
    # Two values of one nest are one fact.
    again = design_space(Held(contents=IntegerTensorValue.of(WEIGHTS)))
    assert held(again).ds.layer is held(given).ds.layer


def test_a_nest_given_for_an_integer_tensor_is_refused() -> None:
    """The annotation is the check: a nest is no ``IntegerTensorValue``, given at the call
    or as a change of facts; ``IntegerTensorValue.of`` makes the value."""
    with pytest.raises(DefinitionError, match="IntegerTensorValue"):
        Held(contents=cast(IntegerTensorValue, WEIGHTS))
    with pytest.raises(RequestError, match="contents: expected value of nominal type"):
        design_space(Held()).with_facts({Held.contents: WEIGHTS})
