# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Mathematics and design choices, independent of a hardware implementation."""

from itertools import product

import pytest
from qonnx.core.datatype import DataType

from kernels.helpers import point_for, value
from finn.kernels._engine import Absent, RequestError, Unresolved
from finn.parked.dataflow.kernels.dot_product import DotProduct
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.kernels.space import Decision, Input
from finn.kernels.space.declarations import declared_members


def facts(**updates):
    return (
        dict(
            repetitions=2,
            matrix_width=4,
            matrix_height=4,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["INT3"],
        )
        | updates
    )


def test_definition_owns_choices_and_needs_no_separate_logical_view():
    members = dict(declared_members(DotProduct))
    assert {name for name, member in members.items() if isinstance(member, Input)} == {
        "repetitions",
        "matrix_width",
        "matrix_height",
        "activation_dtype",
        "weights_dtype",
    }
    assert {name for name, member in members.items() if isinstance(member, Decision)} == {
        "pe",
        "simd",
    }
    point = point_for(DotProduct, facts())
    assert point.capability_names() == ("contract",)
    assert not hasattr(DotProduct, "logical")
    assert not hasattr(DotProduct, "logical_result")
    assert isinstance(point.contract.accepted_answer, Unresolved)
    assert point.result_type == DataType["INT8"]
    assert point.reference([[1, 2, -1, 0], [-4, -4, -4, -4]], [[1, 1, 1, 1]] * 4) == (
        (2, 2, 2, 2),
        (-16, -16, -16, -16),
    )


@pytest.mark.parametrize("pe,simd", tuple(product((1, 2, 4), repeat=2)))
def test_fold_choices_preserve_identity_and_define_exact_relations(pe, simd):
    point = point_for(DotProduct, facts(), pe=pe, simd=simd)
    region = value(point.contract.accepted_answer)
    assert region.input("X").operand.shape == (2, 4)
    assert region.input("W").operand.shape == (4, 4)
    output = region.outputs[0]
    assert output.port.operand.shape == (2, 4)
    assert output.port.operand.element_type == DataType["INT8"]
    nf_count, sf_count = 4 // pe, 4 // simd
    activation, weight = region.inputs
    for beat, (r, nf, sf) in enumerate(product(range(2), range(nf_count), range(sf_count))):
        xs = tuple((r, sf * simd + s) for s in range(simd))
        ws = tuple((nf * pe + p, sf * simd + s) for p in range(pe) for s in range(simd))
        assert activation.port.beat_sequence.beat(beat) == xs
        assert weight.port.beat_sequence.beat(beat) == ws
        for pos in product(range(2), range(4)):
            assert activation.requirements.required((r, nf, sf), pos) == int(pos in xs)
        for pos in product(range(4), repeat=2):
            assert weight.requirements.required((r, nf, sf), pos) == int(pos in ws)
    for beat, (r, nf) in enumerate(product(range(2), range(nf_count))):
        ys = tuple((r, nf * pe + p) for p in range(pe))
        assert output.port.beat_sequence.beat(beat) == ys
        for pos in ys:
            assert output.availability.available_at(pos) == (r, nf, sf_count - 1)
    # The repeated presentation names the SAME logical X positions.
    assert activation.requirements.occurrence_count == 2 * 4 * nf_count


@pytest.mark.parametrize(
    "left,right", tuple(product(("INT2", "UINT2", "INT1", "BINARY"), repeat=2))
)
def test_exact_arithmetic_and_minimal_signed_dtype_for_all_small_vectors(left, right):
    x_type, w_type = DataType[left], DataType[right]
    point = point_for(
        DotProduct, facts(matrix_width=2, activation_dtype=x_type, weights_dtype=w_type)
    )
    arithmetic = point.arithmetic
    xs = tuple(product(range(int(x_type.min()), int(x_type.max()) + 1), repeat=2))
    ws = tuple(product(range(int(w_type.min()), int(w_type.max()) + 1), repeat=2))
    results = []
    for x, w in product(xs, ws):
        expected = x[0] * w[0] + x[1] * w[1]
        assert arithmetic.evaluate(x, w) == expected
        results.append(expected)
    assert (arithmetic.result_range.minimum, arithmetic.result_range.maximum) == (
        min(results),
        max(results),
    )
    bits = point.result_type.bitwidth()
    assert -(1 << (bits - 1)) <= min(results) <= max(results) < (1 << (bits - 1))
    assert bits == 1 or min(results) < -(1 << (bits - 2)) or max(results) >= (1 << (bits - 2))
    with pytest.raises(ValueError, match="ranges"):
        arithmetic.evaluate([int(x_type.max()) + 1, 0], [0, 0])
    with pytest.raises(ValueError, match="length"):
        arithmetic.evaluate([0], [0])


@pytest.mark.parametrize("dtype", ("BIPOLAR", "TERNARY", "FLOAT32"))
def test_special_encodings_are_not_silently_reinterpreted(dtype):
    point = point_for(DotProduct, facts(activation_dtype=DataType[dtype]), pe=2, simd=2)
    assert isinstance(point.contract.accepted_answer, Absent)
    assert isinstance(point.answer(DotProduct.arithmetic), Absent)
    assert point.activation_dtype == DataType[dtype]


def test_large_integer_bounds_and_contract_construction_do_not_enumerate(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("contract construction enumerated the tensor")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", forbidden)
    point = point_for(
        DotProduct,
        facts(matrix_width=1_000_000, activation_dtype=DataType["INT80"]),
        pe=2,
        simd=4,
    )
    region = value(point.contract.accepted_answer)
    assert point.arithmetic.result_range.maximum == 1_000_000 * (1 << 79) * 4
    assert region.input("X").operand.element_type == DataType["INT80"]


@pytest.mark.parametrize(
    "updates", ({"repetitions": 0}, {"matrix_width": 0}, {"matrix_height": -1})
)
def test_empty_or_negative_tensor_extents_are_rejected(updates):
    point = point_for(DotProduct, facts(**updates), pe=1, simd=1)
    assert isinstance(point.contract.accepted_answer, Absent)


def test_nondividing_choices_are_rejected_by_space():
    with pytest.raises(RequestError) as error:
        point_for(DotProduct, facts(), pe=3, simd=2)
    assert {finding.code for finding in error.value.findings} == {"candidate-outside-domain"}
