# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.dataflow.ops.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_streamed_mvau_region,
)
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import RegionValidationReport, validate_region

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


def _interleaved(
    repetitions: int = 6,
    matrix_width: int = 4,
    matrix_height: int = 6,
    pe: int = 3,
    simd: int = 2,
    interleave: int = 3,
) -> DataflowRegion:
    return construct_batch_interleaved_streamed_mvau_region(
        repetitions,
        matrix_width,
        matrix_height,
        INT8,
        INT8,
        INT16,
        pe,
        simd,
        interleave,
    )


def test_representative_interleaved_region_has_exact_schedule_and_relations() -> None:
    region = _interleaved()
    activation = region.input_interface("activation")
    weight = region.input_interface("weight")
    output = region.output_interface("output")

    assert validate_region(region) == RegionValidationReport()
    assert region.schedule.level_names == ("batch", "nf", "sf", "t")
    assert region.schedule.extents == (2, 2, 2, 3)
    assert activation.requirements.required((1, 0, 1, 2), (5, 2)) == 1
    assert activation.requirements.required((1, 0, 1, 2), (4, 2)) == 0
    assert weight.requirements.required((1, 1, 0, 2), (5, 1)) == 1
    assert weight.requirements.required((1, 1, 0, 2), (2, 1)) == 0
    assert output.availability.available_at((5, 5)) == (1, 1, 1, 2)
    assert output.availability.domain == output.port.beat_sequence.image


def test_interleaved_weight_chunks_have_exact_order_and_repeat_by_batch() -> None:
    weight_sequence = _interleaved().input_interface("weight").port.beat_sequence

    assert weight_sequence.elements_per_beat == 2
    assert weight_sequence.beat_count == 24
    assert weight_sequence.beats[:6] == (
        ((0, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((2, 0), (2, 1)),
        ((0, 2), (0, 3)),
        ((1, 2), (1, 3)),
        ((2, 2), (2, 3)),
    )
    assert weight_sequence.beats[12:18] == weight_sequence.beats[:6]
    for first in range(0, weight_sequence.beat_count, 3):
        tile = tuple(
            position for beat in weight_sequence.beats[first : first + 3] for position in beat
        )
        assert len(tile) == 6
        assert len(set(tile)) == 6


@pytest.mark.parametrize(
    "repetitions,matrix_width,matrix_height,pe,simd,interleave",
    [(4, 8, 4, 2, 2, 2), (8, 4, 8, 4, 2, 4), (6, 12, 6, 3, 3, 3)],
)
def test_interleaved_output_domain_and_sequence_are_exact_over_multiple_configs(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    pe: int,
    simd: int,
    interleave: int,
) -> None:
    region = _interleaved(repetitions, matrix_width, matrix_height, pe, simd, interleave)
    output = region.output_interface("output")
    assert validate_region(region) == RegionValidationReport()
    assert output.availability.domain == output.port.beat_sequence.image
    assert output.port.beat_sequence.beat_count == repetitions * (matrix_height // pe)


def test_interleaving_changes_schedule_and_weight_sequence_but_preserves_x_and_y() -> None:
    interleaved = _interleaved()
    standard = construct_standard_streamed_mvau_region(6, 4, 6, INT8, INT8, INT16, 3, 2)

    assert interleaved != standard
    assert interleaved.schedule != standard.schedule
    assert (
        interleaved.input_interface("weight").port.beat_sequence
        != standard.input_interface("weight").port.beat_sequence
    )
    assert (
        interleaved.input_interface("activation").port.beat_sequence
        == standard.input_interface("activation").port.beat_sequence
    )
    assert (
        interleaved.output_interface("output").port.beat_sequence
        == standard.output_interface("output").port.beat_sequence
    )


@pytest.mark.parametrize(
    "repetitions,pe,simd,interleave,match",
    [(5, 3, 2, 3, "repetitions"), (6, 2, 2, 3, r"PE \* SIMD"), (5, 2, 2, 3, None)],
)
def test_invalid_interleave_is_rejected_by_the_constructor(
    repetitions: int, pe: int, simd: int, interleave: int, match: str | None
) -> None:
    with pytest.raises(ValueError, match=match):
        _interleaved(repetitions, 4, 6, pe, simd, interleave)
