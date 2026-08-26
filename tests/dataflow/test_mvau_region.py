# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.dataflow.mvau_design import construct_streamed_weight_mvau_region
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    NumericElementType,
    OutputInterface,
    Port,
    ScheduledOutputAvailability,
)
from finn.dataflow.region_validation import validate_region


def _mvau_region(
    repetitions,
    matrix_width,
    matrix_height,
    simd,
    pe,
    *,
    weight_field_order=None,
    output_field_order=None,
):
    element_type = NumericElementType("int", 8)
    region = construct_streamed_weight_mvau_region(
        repetitions,
        matrix_width,
        matrix_height,
        element_type,
        element_type,
        element_type,
        pe,
        simd,
    )
    if weight_field_order is None and output_field_order is None:
        return region

    input_folds = matrix_width // simd
    output_folds = matrix_height // pe
    weight = region.input_interface("weight")
    output = region.output_interface("output")
    if weight_field_order is None:
        weight_field_order = tuple(
            (pe_index, lane) for pe_index in range(pe) for lane in range(simd)
        )
    if output_field_order is None:
        output_field_order = tuple(range(pe))
    weight_beats = tuple(
        tuple((nf * pe + pe_index, sf * simd + lane) for pe_index, lane in weight_field_order)
        for _rep in range(repetitions)
        for nf in range(output_folds)
        for sf in range(input_folds)
    )
    output_beats = tuple(
        tuple((rep, nf * pe + pe_index) for pe_index in output_field_order)
        for rep in range(repetitions)
        for nf in range(output_folds)
    )
    reordered_weight = InputInterface(
        Port(
            "weight",
            weight.port.operand,
            BeatSequence(pe * simd, weight_beats),
        ),
        weight.requirements,
    )
    reordered_output = OutputInterface(
        Port("output", output.port.operand, BeatSequence(pe, output_beats)),
        output.availability,
    )
    return DataflowRegion(
        region.schedule,
        (region.input_interface("activation"), reordered_weight),
        (reordered_output,),
    )


def test_small_streamed_weight_mvau_region_matches_authoring_semantics():
    repetitions, matrix_width, matrix_height, simd, pe = 2, 4, 4, 2, 2
    input_folds = matrix_width // simd
    output_folds = matrix_height // pe
    region = _mvau_region(repetitions, matrix_width, matrix_height, simd, pe)
    activation = region.input_interface("activation")
    weight = region.input_interface("weight")
    output = region.output_interface("output")

    assert validate_region(region).issues == ()
    assert region.schedule.level_names == ("rep", "nf", "sf")
    assert region.schedule.rank((1, 0, 1)) == 5
    assert activation.requirements.occurrence_count == (
        repetitions * output_folds * input_folds * simd
    )
    assert activation.port.beat_sequence.delivered_field_count == (repetitions * input_folds * simd)
    assert (
        activation.requirements.occurrence_count
        == output_folds * activation.port.beat_sequence.delivered_field_count
    )
    assert weight.requirements.occurrence_count == (
        repetitions * output_folds * input_folds * pe * simd
    )

    for iteration in region.schedule.iter_points():
        ordinal = region.schedule.rank(iteration)
        rep, nf, sf = iteration
        expected_weight_beat = tuple(
            (nf * pe + pe_index, sf * simd + lane) for pe_index in range(pe) for lane in range(simd)
        )
        assert weight.port.beat_sequence.beat(ordinal) == expected_weight_beat
        assert all(
            weight.requirements.required(iteration, position) == 1
            for position in expected_weight_beat
        )
        for pe_index in range(pe):
            position = (rep, nf * pe + pe_index)
            if sf == input_folds - 1:
                assert output.availability.available_at(position) == iteration
            else:
                assert output.availability.available_at(position) != iteration

    assert output.availability.domain == output.port.beat_sequence.image


def test_mvau_field_bijections_change_sequences_without_changing_widths_or_shapes():
    original = _mvau_region(2, 4, 4, 2, 2)
    reordered = _mvau_region(
        2,
        4,
        4,
        2,
        2,
        weight_field_order=((1, 1), (1, 0), (0, 1), (0, 0)),
        output_field_order=(1, 0),
    )
    original_weight = original.input_interface("weight").port
    reordered_weight = reordered.input_interface("weight").port
    original_output = original.output_interface("output").port
    reordered_output = reordered.output_interface("output").port

    assert validate_region(reordered).issues == ()
    assert original_weight.operand.shape == reordered_weight.operand.shape
    assert original_weight.logical_beat_bits == reordered_weight.logical_beat_bits
    assert original_weight.beat_sequence != reordered_weight.beat_sequence
    assert original_output.operand.shape == reordered_output.operand.shape
    assert original_output.logical_beat_bits == reordered_output.logical_beat_bits
    assert original_output.beat_sequence != reordered_output.beat_sequence


def test_corrupting_mvau_output_availability_is_a_structural_failure():
    region = _mvau_region(2, 4, 4, 2, 2)
    output = region.output_interface("output")
    corrupt = OutputInterface(
        output.port,
        ScheduledOutputAvailability(dict(output.availability.entries[:-1])),
    )
    malformed = DataflowRegion(region.schedule, region.inputs, (corrupt,))

    assert "output.domain_image_mismatch" in tuple(
        issue.code for issue in validate_region(malformed)
    )


@pytest.mark.parametrize(
    "repetitions,matrix_width,matrix_height,simd,pe,expected_activation,expected_weight",
    [
        (1, 4, 4, 2, 2, 8, 16),
        (1, 64, 64, 8, 8, 512, 4096),
    ],
)
def test_mvau_concrete_storage_scales_with_occurrences(
    repetitions,
    matrix_width,
    matrix_height,
    simd,
    pe,
    expected_activation,
    expected_weight,
):
    region = _mvau_region(repetitions, matrix_width, matrix_height, simd, pe)

    assert validate_region(region).issues == ()
    assert len(region.input_interface("activation").requirements.entries) == expected_activation
    assert len(region.input_interface("weight").requirements.entries) == expected_weight
