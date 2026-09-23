# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit dotp traffic agrees with its contract and the historical oracle."""

import pytest
from qonnx.core.datatype import DataType

from finn.dataflow.kernels.dotp_axi_region import construct_dotp_region
from finn.dataflow.kernels.matmul.regions import construct_dot_product_region
from finn.dataflow.model.logical.region import RegionRefused
from finn.dataflow.model.logical.region_validation import validate_region


def facts(**overrides):
    return (
        dict(
            repetitions=2,
            matrix_width=6,
            matrix_height=4,
            activation_element_type=DataType["INT3"],
            weight_element_type=DataType["INT3"],
            output_element_type=DataType["INT16"],
            pe=2,
            simd=3,
        )
        | overrides
    )


@pytest.mark.parametrize(
    "rep,width,height,pe,simd",
    [(1, 1, 1, 1, 1), (2, 6, 4, 2, 3), (3, 8, 6, 3, 4), (1, 8, 4, 4, 8)],
)
def test_standard_schedule_and_ports_match_the_historical_oracle(rep, width, height, pe, simd):
    supplied = facts(repetitions=rep, matrix_width=width, matrix_height=height, pe=pe, simd=simd)
    actual = construct_dotp_region(**supplied)
    assert actual == construct_dot_product_region(**supplied)
    assert not validate_region(actual).issues


def test_replayed_activations_weight_tiles_and_output_completion_are_explicit():
    region = construct_dotp_region(**facts())
    activation = region.input_interface("activation")
    weights = region.input_interface("weight")
    output = region.output_interface("output")
    assert region.schedule.extents == (2, 2, 2)
    assert activation.port.operand.shape == (4, 6)
    assert activation.port.beat_sequence.beat(0) == ((0, 0), (0, 1), (0, 2))
    assert activation.port.beat_sequence.beat(2) == ((1, 0), (1, 1), (1, 2))
    assert activation.port.beat_sequence.beat(4) == ((2, 0), (2, 1), (2, 2))
    assert weights.port.beat_sequence.beat(0) == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
    assert weights.port.beat_sequence.beat(2) == ((2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2))
    assert weights.port.beat_sequence.beat(4) == weights.port.beat_sequence.beat(0)
    assert activation.requirements.required((1, 1, 1), (3, 5)) == 1
    assert weights.requirements.required((1, 1, 1), (3, 5)) == 1
    assert output.port.beat_sequence.beat(0) == ((0, 0), (0, 1))
    assert output.port.beat_sequence.beat(3) == ((1, 2), (1, 3))
    assert output.availability.available_at((0, 0)) == (0, 0, 1)
    assert output.availability.available_at((1, 3)) == (1, 1, 1)


@pytest.mark.parametrize("overrides", [{"pe": 0}, {"simd": 4}, {"matrix_height": 3}])
def test_incomplete_folding_is_refused(overrides):
    with pytest.raises(RegionRefused):
        construct_dotp_region(**facts(**overrides))
