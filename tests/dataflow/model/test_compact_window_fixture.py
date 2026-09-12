# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)


def _window_regions(count: int = 4, length: int = 3):
    input_length = count + length - 1
    schedule = LogicalSchedule((("o", count), ("k", length)))
    source_operand = Operand("X", DataType["INT8"], (input_length,))
    window_operand = Operand("W", DataType["INT8"], (count, length))
    grouped_operand = Operand("V", DataType["INT8"], (count, length))

    source_beats = BeatSequence.affine(
        source_operand.position_domain,
        elements_per_beat=1,
        beat_count=input_length,
        view_extents=(input_length, 1),
        offset=0,
        coefficients=(1, 0),
    )
    scalar_window_beats = BeatSequence.affine(
        window_operand.position_domain,
        elements_per_beat=1,
        beat_count=count * length,
        view_extents=(count, length, 1),
        offset=0,
        coefficients=(length, 1, 0),
    )
    grouped_window_beats = BeatSequence.affine(
        grouped_operand.position_domain,
        elements_per_beat=length,
        beat_count=count,
        view_extents=(count, length),
        offset=0,
        coefficients=(length, 1),
    )
    identity_requirements = ScheduledInputRequirements.affine(
        schedule.iteration_domain,
        window_operand.position_domain,
        base=(0, 0),
        iteration_coefficients=((1, 0), (0, 1)),
    )
    grouped_requirements = ScheduledInputRequirements.affine(
        schedule.iteration_domain,
        grouped_operand.position_domain,
        base=(0, 0),
        iteration_coefficients=((1, 0), (0, 1)),
    )

    generator = DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("x_in", source_operand, source_beats),
                ScheduledInputRequirements.affine(
                    schedule.iteration_domain,
                    source_operand.position_domain,
                    base=(0,),
                    iteration_coefficients=((1, 1),),
                ),
            ),
        ),
        (
            OutputInterface(
                Port("w_out", window_operand, scalar_window_beats),
                ScheduledOutputAvailability.affine(
                    window_operand.position_domain,
                    schedule.iteration_domain,
                    view_extents=(count, length),
                    offset=0,
                    coefficients=(length, 1),
                ),
            ),
        ),
    )
    dwc = DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("w_in", window_operand, scalar_window_beats),
                identity_requirements,
            ),
        ),
        (
            OutputInterface(
                Port("v_out", grouped_operand, grouped_window_beats),
                ScheduledOutputAvailability.affine(
                    grouped_operand.position_domain,
                    schedule.iteration_domain,
                    view_extents=(count, length),
                    offset=length - 1,
                    coefficients=(length, 0),
                ),
            ),
        ),
    )
    consumer = DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("v_in", grouped_operand, grouped_window_beats),
                grouped_requirements,
            ),
        ),
        (),
    )
    return generator, dwc, consumer


def test_window_fixture_uses_a_dwc_between_scalar_and_grouped_presentations():
    generator, dwc, consumer = _window_regions()
    window_set = generator.output_interface("w_out").port.beat_sequence.image_set
    grouped_set = dwc.output_interface("v_out").port.beat_sequence.image_set
    network = DataflowNetwork(
        (
            NetworkNode("generator", generator),
            NetworkNode("dwc", dwc),
            NetworkNode("consumer", consumer),
        ),
        (
            Edge(
                "scalar",
                RegionEndpoint("generator", "w_out"),
                (
                    SinkContract(
                        RegionEndpoint("dwc", "w_in"),
                        PositionMap.identity(window_set),
                    ),
                ),
            ),
            Edge(
                "grouped",
                RegionEndpoint("dwc", "v_out"),
                (
                    SinkContract(
                        RegionEndpoint("consumer", "v_in"),
                        PositionMap.identity(grouped_set),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "input",
                RegionEndpoint("generator", "x_in"),
                generator.input_interface("x_in").port.beat_sequence,
            ),
        ),
    )

    assert validate_network(network) == NetworkValidationReport()
    requirements = generator.input("X").requirements
    expected = tuple((offset + field,) for offset in range(4) for field in range(3))
    actual = tuple(
        position
        for iteration in generator.schedule.iter_points()
        for position in generator.input("X").operand.iter_positions()
        if requirements.required(iteration, position)
    )
    assert actual == expected


def test_direct_scalar_to_grouped_window_edge_is_the_negative_control():
    generator, _dwc, consumer = _window_regions()
    position_set = generator.output_interface("w_out").port.beat_sequence.image_set
    network = DataflowNetwork(
        (NetworkNode("generator", generator), NetworkNode("consumer", consumer)),
        (
            Edge(
                "wrong",
                RegionEndpoint("generator", "w_out"),
                (
                    SinkContract(
                        RegionEndpoint("consumer", "v_in"),
                        PositionMap.identity(position_set),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "input",
                RegionEndpoint("generator", "x_in"),
                generator.input_interface("x_in").port.beat_sequence,
            ),
        ),
    )

    codes = {issue.code for issue in validate_network(network)}
    assert {"edge.beat_count_mismatch", "edge.element_count_mismatch"} <= codes
