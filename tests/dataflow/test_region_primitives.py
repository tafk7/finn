# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataclasses import FrozenInstanceError

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    Port,
    ScheduledInputRequirements,
)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 2, 3)])
def test_operand_position_enumeration_and_rank_round_trip(shape):
    operand = Operand("x", DataType["INT8"], shape)

    assert len(operand.positions) == operand.position_count
    for expected_rank, position in enumerate(operand.positions):
        assert operand.position_rank(position) == expected_rank
        assert operand.position_at_rank(expected_rank) == position


@pytest.mark.parametrize("levels", [(), (("x", 3),), (("x", 2), ("y", 3))])
def test_schedule_enumeration_and_rank_round_trip(levels):
    schedule = LogicalSchedule(levels)

    assert len(schedule.iteration_points) == schedule.iteration_count
    for expected_rank, point in enumerate(schedule.iteration_points):
        assert schedule.rank(point) == expected_rank
        assert schedule.point_at_rank(expected_rank) == point


def test_rank_zero_operand_and_schedule_have_one_empty_coordinate():
    operand = Operand("scalar", DataType["FIXED<12,6>"], ())
    schedule = LogicalSchedule(())

    assert operand.positions == ((),)
    assert schedule.iteration_points == ((),)


def test_beat_field_order_and_repeated_positions_are_preserved():
    sequence = BeatSequence(3, (((2,), (0,), (2,)), ((1,), (3,), (1,))))

    assert sequence.beat_count == 2
    assert sequence.field_ordinals == (0, 1, 2)
    assert sequence.beat(0) == ((2,), (0,), (2,))
    assert sequence.position_at(1, 1) == (3,)
    assert sequence.delivered_field_count == 6
    assert sequence.image == frozenset({(0,), (1,), (2,), (3,)})


@pytest.mark.parametrize(
    "lookup,error",
    [
        (lambda sequence: sequence.beat(-1), ValueError),
        (lambda sequence: sequence.beat(2), ValueError),
        (lambda sequence: sequence.beat(True), TypeError),
        (lambda sequence: sequence.position_at(0, -1), ValueError),
        (lambda sequence: sequence.position_at(0, 3), ValueError),
        (lambda sequence: sequence.position_at(0, False), TypeError),
    ],
)
def test_beat_access_rejects_indices_outside_canonical_domains(lookup, error):
    sequence = BeatSequence(3, (((0,), (1,), (2,)), ((3,), (4,), (5,))))

    with pytest.raises(error):
        lookup(sequence)


def test_requirement_multiplicity_and_occurrences_are_exact():
    requirements = ScheduledInputRequirements(
        {
            ((1,), (0,)): 1,
            ((0,), (1,)): 3,
        }
    )

    assert requirements.required((0,), (0,)) == 0
    assert requirements.required((0,), (1,)) == 3
    assert requirements.occurrences == (
        ((0,), (1,), 0),
        ((0,), (1,), 1),
        ((0,), (1,), 2),
        ((1,), (0,), 0),
    )
    assert requirements.occurrence_count == 4


def test_equal_normalized_values_ignore_mapping_declaration_order():
    first = ScheduledInputRequirements(
        [
            (((1,), (1,)), 2),
            (((0,), (0,)), 1),
        ]
    )
    second = ScheduledInputRequirements(
        [
            (((0,), (0,)), 1),
            (((1,), (1,)), 2),
        ]
    )

    assert first == second
    assert hash(first) == hash(second)


def test_explicit_zero_requirement_is_rejected_as_noncanonical_sparse_syntax():
    with pytest.raises(ValueError, match="explicit zero multiplicity"):
        ScheduledInputRequirements({((0,), (0,)): 0})


def test_mutable_inputs_are_snapshotted_and_values_are_frozen():
    shape = [2]
    beats = [[(0,)], [(1,)]]
    requirement_map = {((0,), (0,)): 1}
    operand = Operand("x", DataType["UINT4"], shape)
    sequence = BeatSequence(1, beats)
    requirements = ScheduledInputRequirements(requirement_map)

    shape[0] = 99
    beats[0][0] = (99,)
    requirement_map[((0,), (0,))] = 7

    assert operand.shape == (2,)
    assert sequence.beats == (((0,),), ((1,),))
    assert requirements.required((0,), (0,)) == 1
    with pytest.raises(FrozenInstanceError):
        operand.shape = (3,)


def test_region_interface_sets_have_deterministic_order_and_equality():
    operand = Operand("x", DataType["INT8"], (2,))
    requirements = ScheduledInputRequirements()
    first = InputInterface(Port("b", operand, BeatSequence(1, (((0,),),))), requirements)
    second = InputInterface(Port("a", operand, BeatSequence(1, (((1,),),))), requirements)

    left = DataflowRegion(LogicalSchedule(()), (first, second), ())
    right = DataflowRegion(LogicalSchedule(()), (second, first), ())

    assert left == right
    assert tuple(interface.port.id for interface in left.inputs) == ("a", "b")
