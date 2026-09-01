# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test-only evidence for the deferred weight-sequence adapter Region."""

from __future__ import annotations

from collections import Counter

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.mvau.weight_adapter_region import (
    construct_weight_sequence_adapter_region,
    weight_sequence_adapter_applicable,
)
from finn.dataflow.network import (
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import validate_network
from finn.dataflow.ops.mvau.regions import (
    construct_batch_interleaved_mvau_weight_port,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Port,
    ScheduledInputRequirements,
)
from finn.dataflow.region_validation import validate_region

INT8 = DataType["INT8"]


def _ports() -> tuple[Port, Port]:
    return (
        construct_standard_mvau_weight_port(2, 4, 4, INT8, 2, 2),
        construct_batch_interleaved_mvau_weight_port(2, 4, 4, INT8, 2, 2, 2),
    )


def test_full_tile_to_chunked_adapter_has_exact_schedule_and_relations() -> None:
    full, chunked = _ports()
    adapter = construct_weight_sequence_adapter_region(full, chunked)

    assert not validate_region(adapter)
    assert adapter.schedule.level_names == ("output_beat",)
    assert adapter.schedule.extents == (chunked.beat_sequence.beat_count,)
    assert adapter.input_interface("weight_in").port.beat_sequence == full.beat_sequence
    assert adapter.output_interface("weight_out").port.beat_sequence == chunked.beat_sequence
    requirements = adapter.input_interface("weight_in").requirements
    for ordinal, beat in enumerate(chunked.beat_sequence.beats):
        expected = Counter(beat)
        for position in chunked.beat_sequence.image:
            assert requirements.required((ordinal,), position) == expected[position]


def test_chunked_to_full_adapter_preserves_exact_boundary_sequences() -> None:
    full, chunked = _ports()
    adapter = construct_weight_sequence_adapter_region(chunked, full)

    assert adapter.input_interface("weight_in").port.beat_sequence == chunked.beat_sequence
    assert adapter.output_interface("weight_out").port.beat_sequence == full.beat_sequence
    assert adapter.input_interface("weight_in").port.operand == (
        adapter.output_interface("weight_out").port.operand
    )


def test_equal_width_with_different_field_order_is_not_directly_equal() -> None:
    full, _chunked = _ports()
    reordered = Port(
        "weight",
        full.operand,
        BeatSequence(
            full.beat_sequence.elements_per_beat,
            tuple(tuple(reversed(beat)) for beat in full.beat_sequence.beats),
        ),
    )

    assert full.beat_type.logical_bit_width == reordered.beat_type.logical_bit_width
    assert full.beat_sequence != reordered.beat_sequence
    assert weight_sequence_adapter_applicable(full, reordered)
    source = construct_cyclic_parameter_region(full)
    sink = DataflowRegion(
        LogicalSchedule(()),
        (InputInterface(reordered, ScheduledInputRequirements()),),
        (),
    )
    report = validate_network(
        DataflowNetwork(
            (NetworkNode("source", source), NetworkNode("sink", sink)),
            (
                Edge(
                    "weight",
                    RegionEndpoint("source", "weight"),
                    (
                        SinkContract(
                            RegionEndpoint("sink", "weight"),
                            PositionMap.identity(full.beat_sequence.image),
                        ),
                    ),
                ),
            ),
            (),
        )
    )
    assert "edge.beat_sequence_mismatch" in {issue.code for issue in report.issues}


def test_adapter_rejects_different_tensor_images() -> None:
    full, chunked = _ports()
    invalid = Port(
        "weight",
        chunked.operand,
        BeatSequence(
            chunked.beat_sequence.elements_per_beat,
            chunked.beat_sequence.beats[:-1],
        ),
    )

    assert not weight_sequence_adapter_applicable(full, invalid)
    with pytest.raises(ValueError):
        construct_weight_sequence_adapter_region(full, invalid)
