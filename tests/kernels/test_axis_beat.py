# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Detached AXIS beat packing and scalar datatype snapshots."""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.transport import AxisBeat


@pytest.mark.parametrize("endpoint", tuple(Endpoint))
def test_subbyte_scalars_pack_tightly_into_a_byte_and_directions_follow_the_endpoint(endpoint):
    beat = AxisBeat("data", DataType["INT3"], 2, endpoint=endpoint)
    assert beat.payload_bits == 6 and beat.data_width == 8
    directions = dict(beat.bus().member_directions())
    assert directions["data_tdata"] is (
        Direction.IN if endpoint is Endpoint.TARGET else Direction.OUT
    )
    assert directions["data_tready"] is (
        Direction.OUT if endpoint is Endpoint.TARGET else Direction.IN
    )


def test_dtype_is_the_immutable_value_the_caller_gave():
    dtype = DataType["INT3"]
    beat = AxisBeat("data", dtype, 2, endpoint=Endpoint.TARGET)
    assert beat.dtype is dtype
    with pytest.raises(AttributeError, match="immutable datatype value"):
        dtype._bitwidth = 8
    assert beat.dtype == DataType["INT3"]
    assert beat.data_width == 8


@pytest.mark.parametrize("lanes", (0, -1, True))
def test_invalid_beat_size_refuses(lanes):
    with pytest.raises(ValueError, match="positive integer"):
        AxisBeat("data", DataType["INT3"], lanes, endpoint=Endpoint.TARGET)
