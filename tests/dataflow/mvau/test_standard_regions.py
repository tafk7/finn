# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from typing import NamedTuple

from finn.dataflow.ops.mvau.regions import (
    MVAUWeightInterface,
    construct_mvau_compute_region,
    construct_standard_embedded_mvau_region,
    construct_standard_streamed_mvau_region,
    construct_streamed_weight_mvau_region,
)
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.region import NumericElementType
from finn.dataflow.region_validation import RegionValidationReport, validate_region

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


class _Arguments(NamedTuple):
    repetitions: int
    matrix_width: int
    matrix_height: int
    activation_type: NumericElementType
    weight_type: NumericElementType
    output_type: NumericElementType
    pe: int
    simd: int


def _arguments(pe: int = 2, simd: int = 2) -> _Arguments:
    return _Arguments(2, 4, 4, INT8, INT8, INT16, pe, simd)


def test_standard_constructors_preserve_the_compatibility_facades() -> None:
    arguments = _arguments()
    streamed = construct_standard_streamed_mvau_region(*arguments)
    embedded = construct_standard_embedded_mvau_region(*arguments)

    assert construct_streamed_weight_mvau_region(*arguments) == streamed
    assert construct_mvau_compute_region(*arguments, MVAUWeightInterface.STREAMED) == streamed
    assert construct_mvau_compute_region(*arguments, MVAUWeightInterface.EMBEDDED) == embedded


def test_standard_regions_have_exact_schedule_requirements_availability_and_beats() -> None:
    streamed = construct_standard_streamed_mvau_region(*_arguments())
    activation = streamed.input_interface("activation")
    weight = streamed.input_interface("weight")
    output = streamed.output_interface("output")

    assert streamed.schedule.level_names == ("rep", "nf", "sf")
    assert streamed.schedule.extents == (2, 2, 2)
    assert activation.requirements.required((0, 0, 0), (0, 0)) == 1
    assert activation.requirements.required((0, 1, 0), (0, 0)) == 1
    assert activation.requirements.required((0, 0, 0), (0, 2)) == 0
    assert activation.port.beat_sequence.beats == (
        ((0, 0), (0, 1)),
        ((0, 2), (0, 3)),
        ((1, 0), (1, 1)),
        ((1, 2), (1, 3)),
    )
    assert weight.requirements.required((0, 1, 0), (2, 0)) == 1
    assert weight.requirements.required((0, 1, 0), (0, 0)) == 0
    assert weight.port.beat_sequence.beat(2) == ((2, 0), (2, 1), (3, 0), (3, 1))
    assert output.availability.available_at((1, 2)) == (1, 1, 1)
    assert output.port.beat_sequence.beats == (
        ((0, 0), (0, 1)),
        ((0, 2), (0, 3)),
        ((1, 0), (1, 1)),
        ((1, 2), (1, 3)),
    )
    assert validate_region(streamed) == RegionValidationReport()


@pytest.mark.parametrize("pe,simd", [(1, 1), (1, 4), (4, 1), (4, 4)])
def test_standard_declarations_are_structurally_valid_at_divisor_edges(pe: int, simd: int) -> None:
    for constructor in (
        construct_standard_embedded_mvau_region,
        construct_standard_streamed_mvau_region,
    ):
        assert validate_region(constructor(*_arguments(pe=pe, simd=simd))) == (
            RegionValidationReport()
        )


def test_embedded_and_streamed_are_distinct_only_at_the_weight_interface() -> None:
    embedded = construct_standard_embedded_mvau_region(*_arguments())
    streamed = construct_standard_streamed_mvau_region(*_arguments())

    assert embedded != streamed
    assert embedded.schedule == streamed.schedule
    assert embedded.input_interface("activation") == streamed.input_interface("activation")
    assert embedded.outputs == streamed.outputs
    assert tuple(interface.port.id for interface in embedded.inputs) == ("activation",)
    assert tuple(interface.port.id for interface in streamed.inputs) == ("activation", "weight")


@pytest.mark.parametrize("pe,simd", [(3, 2), (2, 3)])
def test_invalid_standard_divisibility_is_rejected(pe: int, simd: int) -> None:
    with pytest.raises(ValueError):
        construct_standard_streamed_mvau_region(*_arguments(pe=pe, simd=simd))
