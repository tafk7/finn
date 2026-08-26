# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)
from finn.dataflow.region_validation import validate_region

ELEMENT_TYPE = NumericElementType("int", 8)


def _input(port_id, operand, beats, requirements, elements_per_beat=1):
    return InputInterface(
        Port(port_id, operand, BeatSequence(elements_per_beat, beats)),
        ScheduledInputRequirements(requirements),
    )


def _output(port_id, operand, beats, availability, elements_per_beat=1):
    return OutputInterface(
        Port(port_id, operand, BeatSequence(elements_per_beat, beats)),
        ScheduledOutputAvailability(availability),
    )


def _codes(region):
    return tuple(issue.code for issue in validate_region(region))


def test_s1_rank_zero_scalar_pass_is_structurally_valid():
    scalar_in = Operand("x", ELEMENT_TYPE, ())
    scalar_out = Operand("y", ELEMENT_TYPE, ())
    region = DataflowRegion(
        LogicalSchedule(()),
        (_input("in", scalar_in, (((),),), {((), ()): 1}),),
        (_output("out", scalar_out, (((),),), {(): ()}),),
    )

    assert region.schedule.iteration_points == ((),)
    assert validate_region(region) == ()


def test_s2_replay_shaped_input_is_structurally_valid():
    operand = Operand("x", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("step", 2),)),
        (_input("in", operand, (((0,),),), {((0,), (0,)): 1, ((1,), (0,)): 1}),),
        (),
    )

    interface = region.input_interface("in")
    assert interface.requirements.occurrence_count == 2
    assert interface.port.beat_sequence.delivered_field_count == 1
    assert validate_region(region) == ()


def test_s3_repeated_boundary_input_is_structurally_valid():
    operand = Operand("x", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (_input("in", operand, (((0,),), ((0,),)), {((0,), (0,)): 1}),),
        (),
    )

    assert validate_region(region) == ()


def test_s4_locally_supplied_input_position_is_structurally_valid():
    operand = Operand("parameter", ELEMENT_TYPE, (2,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (_input("in", operand, (((0,),),), {((0,), (1,)): 1}),),
        (),
    )

    assert (1,) not in region.input_interface("in").port.beat_sequence.image
    assert validate_region(region) == ()


def test_s5_repeated_output_position_is_structurally_valid():
    operand = Operand("y", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (),
        (_output("out", operand, (((0,),), ((0,),)), {(0,): (0,)}),),
    )

    assert validate_region(region) == ()


def test_s6_output_position_missing_availability_fails_condition_8():
    operand = Operand("y", ELEMENT_TYPE, (2,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (),
        (_output("out", operand, (((0,),), ((1,),)), {(0,): (0,)}),),
    )

    assert _codes(region) == ("output.domain_image_mismatch",)


def test_s7_available_output_omitted_from_sequence_fails_condition_8():
    operand = Operand("y", ELEMENT_TYPE, (2,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (),
        (_output("out", operand, (((0,),),), {(0,): (0,), (1,): (0,)}),),
    )

    assert _codes(region) == ("output.domain_image_mismatch",)


def test_s8_same_width_with_wrong_field_order_compares_unequal():
    ascending = BeatSequence(2, (((0,), (1,)), ((2,), (3,))))
    descending = BeatSequence(2, (((1,), (0,)), ((3,), (2,))))

    assert ascending.beat_count == descending.beat_count
    assert ascending.elements_per_beat == descending.elements_per_beat
    assert ascending != descending


def test_s9_aggregate_failures_are_reported_in_stable_order():
    operand = Operand("y", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("dup", 1), ("dup", 1))),
        (),
        (_output("out", operand, (((9,),),), {(0,): (7, 0)}),),
    )

    assert _codes(region) == (
        "schedule.level_name_duplicate",
        "availability.iteration_out_of_domain",
        "beat.position_out_of_domain",
        "output.domain_image_mismatch",
    )
    assert validate_region(region) == validate_region(region)


def test_condition_1_schedule_extent_is_positive():
    assert "schedule.extent_not_positive" in _codes(
        DataflowRegion(LogicalSchedule((("x", 0),)), (), ())
    )


def test_condition_2_port_identity_is_unique_across_directions():
    operand = Operand("x", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (_input("same", operand, (((0,),),), {}),),
        (_output("same", operand, (((0,),),), {(0,): (0,)}),),
    )
    assert "port.id_duplicate" in _codes(region)


def test_condition_3_operand_shape_type_and_identity_are_consistent():
    invalid = Operand("shared", NumericElementType("int", 0), (0,))
    conflicting = Operand("shared", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (_input("a", invalid, (), {}), _input("b", conflicting, (((0,),),), {})),
        (),
    )
    codes = _codes(region)
    assert "operand.bit_width_not_positive" in codes
    assert "operand.extent_not_positive" in codes
    assert "operand.identity_conflict" in codes


def test_condition_4_elements_per_beat_is_positive():
    operand = Operand("x", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule(()), (_input("in", operand, (), {}, elements_per_beat=0),), ()
    )
    assert "beat.elements_per_beat_not_positive" in _codes(region)


def test_condition_5_requirement_entries_are_in_domain_and_non_negative():
    operand = Operand("x", ELEMENT_TYPE, (1,))
    requirements = {((2,), (4,)): -1}
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)), (_input("in", operand, (), requirements),), ()
    )
    codes = _codes(region)
    assert "requirement.iteration_out_of_domain" in codes
    assert "requirement.position_out_of_domain" in codes
    assert "requirement.multiplicity_negative" in codes


def test_condition_6_availability_entries_are_in_both_domains():
    operand = Operand("y", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule((("step", 1),)),
        (),
        (_output("out", operand, (), {(4,): (2,)}),),
    )
    codes = _codes(region)
    assert "availability.position_out_of_domain" in codes
    assert "availability.iteration_out_of_domain" in codes


def test_condition_7_beat_fields_are_total_and_positions_valid():
    operand = Operand("x", ELEMENT_TYPE, (1,))
    region = DataflowRegion(
        LogicalSchedule(()),
        (_input("in", operand, (((0,),), ((4,), (0,))), {}, elements_per_beat=2),),
        (),
    )
    codes = _codes(region)
    assert "beat.field_count_mismatch" in codes
    assert "beat.position_out_of_domain" in codes
