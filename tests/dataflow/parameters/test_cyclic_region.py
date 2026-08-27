# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from finn.dataflow.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.parameters.cyclic.region import (
    construct_chunked_cyclic_parameter_region,
    construct_full_tile_cyclic_parameter_region,
)
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.region_validation import RegionValidationReport, validate_region

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def test_full_tile_delivery_exactly_preserves_standard_weight_sequence() -> None:
    compute = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    expected = compute.input_interface("weight").port
    delivery = construct_full_tile_cyclic_parameter_region(expected)

    assert delivery.schedule.levels == ()
    assert delivery.schedule.iteration_points == ((),)
    assert delivery.output_interface("weight").port == expected
    assert set(delivery.output_interface("weight").availability.entries) == {
        (position, ()) for position in expected.beat_sequence.image
    }
    assert validate_region(delivery) == RegionValidationReport()


def test_chunked_delivery_exactly_preserves_interleaved_weight_sequence() -> None:
    compute = construct_batch_interleaved_streamed_mvau_region(6, 4, 6, INT8, INT8, INT16, 3, 2, 3)
    expected = compute.input_interface("weight").port
    delivery = construct_chunked_cyclic_parameter_region(expected)

    assert delivery.output_interface("weight").port == expected
    assert validate_region(delivery) == RegionValidationReport()


def test_rank_zero_delivery_keeps_beat_order_independent_of_availability() -> None:
    operand = Operand("W", INT8, (1, 4))
    port = Port(
        "requested",
        operand,
        BeatSequence(2, (((0, 2), (0, 3)), ((0, 0), (0, 1)))),
    )
    region = construct_full_tile_cyclic_parameter_region(port)
    output = region.output_interface("weight")

    assert output.availability.entries == (
        ((0, 0), ()),
        ((0, 1), ()),
        ((0, 2), ()),
        ((0, 3), ()),
    )
    assert output.port.beat_sequence.beats == port.beat_sequence.beats


def test_repeated_output_positions_remain_structurally_consistent() -> None:
    operand = Operand("W", INT8, (1, 2))
    port = Port("requested", operand, BeatSequence(1, (((0, 0),), ((0, 1),), ((0, 0),))))
    region = construct_full_tile_cyclic_parameter_region(port)

    assert region.output_interface("weight").availability.domain == frozenset({(0, 0), (0, 1)})
    assert validate_region(region) == RegionValidationReport()


def test_rank_zero_and_finite_delivery_schedules_are_both_structural_candidates() -> None:
    compute = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    port = compute.input_interface("weight").port
    rank_zero = construct_full_tile_cyclic_parameter_region(port)
    finite_schedule = LogicalSchedule((ScheduleLevel("delivery", port.beat_sequence.beat_count),))
    first_ordinal = {
        position: next(
            ordinal for ordinal, beat in enumerate(port.beat_sequence.beats) if position in beat
        )
        for position in port.beat_sequence.image
    }
    finite = DataflowRegion(
        finite_schedule,
        (),
        (
            OutputInterface(
                Port("weight", port.operand, port.beat_sequence),
                ScheduledOutputAvailability(
                    {position: (ordinal,) for position, ordinal in first_ordinal.items()}
                ),
            ),
        ),
    )

    assert validate_region(rank_zero) == RegionValidationReport()
    assert validate_region(finite) == RegionValidationReport()
    assert rank_zero.output_interface("weight").port == finite.output_interface("weight").port
    assert rank_zero.schedule.iteration_count == 1
    assert finite.schedule.iteration_count == port.beat_sequence.beat_count
