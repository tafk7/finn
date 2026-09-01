# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import fields

from finn.dataflow.ops.mvau.regions import construct_standard_streamed_mvau_region
from finn.dataflow.network import ChannelSpec, OrderedChannel
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]


def test_ordered_channel_has_no_physical_capacity_or_timing_fields() -> None:
    channel = OrderedChannel(ChannelSpec("weights"))
    assert tuple(field.name for field in fields(channel.specification)) == (
        "channel_id",
        "preserve_order",
        "pass_correspondence",
    )
    assert not hasattr(channel.specification, "depth")
    assert not hasattr(channel.specification, "latency")


def test_position_map_is_not_inferred_from_equal_widths() -> None:
    element_type = DataType["INT8"]
    region = construct_standard_streamed_mvau_region(
        1, 4, 4, element_type, element_type, element_type, 2, 2
    )
    weight = region.input_interface("weight").port
    assert weight.logical_beat_bits == 32
    assert weight.beat_sequence.beat(0) != tuple(reversed(weight.beat_sequence.beat(0)))
