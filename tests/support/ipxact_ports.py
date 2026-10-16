# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ports of a module with a derived clock and an AXI-Lite bus, as its ABI declares them."""

from __future__ import annotations

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)


def stream(name: str, width: int, endpoint: Endpoint) -> Bus:
    members = [Member("tdata", f"{name}_tdata", width)]
    members += [Member(logical, f"{name}_{logical}") for logical in ("tvalid", "tready")]
    return Bus(
        name,
        StandardProtocol.AXIS,
        members,
        endpoint,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


AXILITE = Bus(
    "s_axilite",
    StandardProtocol.AXILITE,
    [
        Member(logical, f"s_axilite_{logical.upper()}", width)
        for logical, width in (("awaddr", 5), ("awvalid", 1), ("wdata", 32), ("rdata", 32))
    ],
    Endpoint.TARGET,
    associated_clock="ap_clk",
    associated_reset="ap_rst_n",
)
PORTS = (
    Signal("ap_clk", Direction.IN, 1, Clock(Free())),
    Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
    Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk", "ap_clk2x"))),
    stream("s_axis_0", 12, Endpoint.TARGET),
    stream("m_axis_0", 8, Endpoint.INITIATOR),
    AXILITE,
)
