# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Detached AXI stream packing and scalar datatype snapshots."""

import pytest
from qonnx.core.datatype import DataType
from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.layout import LanePlacement, UnusedBitPolicy, UnusedBitRange


@pytest.mark.parametrize("endpoint", tuple(Endpoint))
def test_subbyte_scalars_pack_tightly_and_padding_follows_direction(endpoint):
    stream = AxiStream("data", DataType["INT3"], 2, endpoint=endpoint)
    assert stream.data_width == 8
    assert stream.payload.lanes == (LanePlacement(0, 0, 3), LanePlacement(1, 3, 3))
    assert stream.payload.unused == (
        UnusedBitRange(
            6,
            2,
            UnusedBitPolicy.IGNORE_ON_RECEIVE
            if endpoint is Endpoint.TARGET
            else UnusedBitPolicy.UNSPECIFIED,
        ),
    )
    directions = dict(stream.bus().member_directions())
    assert directions["data_tdata"] is (
        Direction.IN if endpoint is Endpoint.TARGET else Direction.OUT
    )
    assert directions["data_tready"] is (
        Direction.OUT if endpoint is Endpoint.TARGET else Direction.IN
    )


def test_dtype_is_a_snapshot_not_a_mutable_caller_reference():
    dtype = DataType["INT3"]
    stream = AxiStream("data", dtype, 2, endpoint=Endpoint.TARGET)
    dtype._bitwidth = 8
    stream.dtype._bitwidth = 16
    assert stream.dtype == DataType["INT3"]
    assert stream.data_width == 8


@pytest.mark.parametrize("lanes", (0, -1, True))
def test_invalid_beat_size_refuses(lanes):
    with pytest.raises(ValueError, match="positive integer"):
        AxiStream("data", DataType["INT3"], lanes, endpoint=Endpoint.TARGET)
