# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataclasses import FrozenInstanceError

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    Port,
    ScheduledInputRequirements,
    InternalInput,
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


def test_ported_input_order_is_the_port_order_even_when_operands_sort_the_other_way():
    """Ported inputs keep their port-id order, deliberately.

    The MVAU compute region's ports are ``activation`` before ``weight`` and its
    operands are ``W`` before ``X``, so a single key over operand ids would have
    reordered every region that already exists and stopped two regions meaning
    the same thing from comparing equal.  The names below sort the two ways
    round on purpose, so this is a decision with a test rather than an accident
    of the current MVAU vocabulary.
    """

    early_port_late_operand = InputInterface(
        Port("a_port", Operand("z_operand", DataType["INT8"], (1,)), BeatSequence(1, (((0,),),))),
        ScheduledInputRequirements(),
    )
    late_port_early_operand = InputInterface(
        Port("z_port", Operand("a_operand", DataType["INT8"], (1,)), BeatSequence(1, (((0,),),))),
        ScheduledInputRequirements(),
    )

    region = DataflowRegion(
        LogicalSchedule(()), (late_port_early_operand, early_port_late_operand), ()
    )

    assert tuple(item.port.id for item in region.inputs) == ("a_port", "z_port")
    assert tuple(item.operand.id for item in region.inputs) == ("z_operand", "a_operand")


def test_internal_inputs_follow_the_ported_ones_in_operand_order():
    """Grouped, so a ported-only region's value is exactly what it was.

    ``inputs[:n]`` equals ``input_interfaces`` for every region, and for a
    region that has no internal input the whole tuple is unchanged from before
    the sum type existed.
    """

    operand = Operand("x", DataType["INT8"], (2,))
    ported = InputInterface(
        Port("z_port", operand, BeatSequence(1, (((0,),),))), ScheduledInputRequirements()
    )
    first_internal = InternalInput(
        Operand("b", DataType["INT8"], (1,)), ScheduledInputRequirements()
    )
    second_internal = InternalInput(
        Operand("a", DataType["INT8"], (1,)), ScheduledInputRequirements()
    )

    left = DataflowRegion(LogicalSchedule(()), (first_internal, ported, second_internal), ())
    right = DataflowRegion(LogicalSchedule(()), (ported, second_internal, first_internal), ())

    assert left == right
    assert left.inputs == (ported, second_internal, first_internal)
    assert left.input_interfaces == (ported,)
    assert left.internal_inputs == (second_internal, first_internal)
    assert left.inputs[: len(left.input_interfaces)] == left.input_interfaces


def test_a_ported_input_takes_its_operand_from_its_port():
    """No second field, so the two can never disagree."""

    operand = Operand("x", DataType["INT8"], (2,))
    interface = InputInterface(
        Port("in", operand, BeatSequence(1, (((0,),),))), ScheduledInputRequirements()
    )

    assert interface.operand is operand


def test_a_region_input_is_reachable_by_operand_and_a_port_by_its_id():
    operand = Operand("x", DataType["INT8"], (2,))
    ported = InputInterface(
        Port("in", operand, BeatSequence(1, (((0,),),))), ScheduledInputRequirements()
    )
    internal = InternalInput(Operand("w", DataType["INT8"], (1,)), ScheduledInputRequirements())
    region = DataflowRegion(LogicalSchedule(()), (ported, internal), ())

    assert region.input("x") is ported
    assert region.input("w") is internal
    assert region.input_interface("in") is ported
    assert region.interfaces == (ported,)
    with pytest.raises(KeyError):
        region.input_interface("w")
    with pytest.raises(KeyError):
        region.input("absent")


def test_required_positions_collapse_iterations_and_multiplicity():
    requirements = ScheduledInputRequirements(
        {((step, visit), (step,)): 1 for step in range(2) for visit in range(3)}
    )

    assert requirements.occurrence_count == 6
    assert requirements.required_positions == frozenset({(0,), (1,)})
