# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The interface description: a module's pins with what its streams carry, as JSON.

Pins with a derived clock and an AXI-Lite bus (``test_ipxact``'s ``PORTS``), so that
every section is stated. ``tests/kernel_ops/test_ip_outputs.py`` reads a packaged
partition's description back against its module's ABI.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.artifacts.interface import InterfaceError, describe_interface

from .test_ipxact import PORTS

#: x's order, row-major: six beats of three lanes.
ROW_MAJOR = {
    "row_major": True,
    "passes": 1,
    "shape": [6, 3],
    "beat_loops": [[6, 3]],
    "lane_loops": [[3, 1]],
}
#: y's order: a (3, 2) matrix column by column, one element a beat.
TRANSPOSED = {
    "row_major": False,
    "passes": 1,
    "shape": [3, 2],
    "beat_loops": [[2, 1], [3, 2]],
    "lane_loops": [],
}

FACTS: dict[str, dict[str, Any]] = {
    "s_axis_0": {
        "port": "s_axis_0",
        "tensor": "x",
        "shape": [1, 6, 3],
        "element": "INT4",
        "range": [-8, 7],
        "lanes": 3,
        "beats": 6,
        "tdata": 12,
        "order": ROW_MAJOR,
    },
    "m_axis_0": {
        "port": "m_axis_0",
        "tensor": "y",
        "shape": [1, 6],
        "element": "UINT8",
        "range": None,
        "lanes": 1,
        "beats": 6,
        "tdata": 8,
        "order": TRANSPOSED,
    },
}


def test_every_pin_is_described_with_what_it_carries() -> None:
    described = describe_interface(PORTS, FACTS, part="xczu3eg-sbva484-1-e", period_ns=5.0)
    assert json.loads(json.dumps(described)) == described
    assert (described["part"], described["period_ns"]) == ("xczu3eg-sbva484-1-e", 5.0)
    # FREQ_HZ at the period; the 2x clock at twice it, aligned to ap_clk.
    assert described["clocks"] == [
        {"name": "ap_clk", "freq_hz": 200_000_000},
        {"name": "ap_clk2x", "freq_hz": 400_000_000, "derived": {"of": "ap_clk", "ratio": 2}},
    ]
    assert described["resets"] == [
        {"name": "ap_rst_n", "polarity": "ACTIVE_LOW", "synchronous_to": ["ap_clk", "ap_clk2x"]}
    ]
    assert described["streams"] == [
        {
            "name": "s_axis_0",
            "direction": "in",
            "tdata": 12,
            "clock": "ap_clk",
            "tensor": "x",
            "element": "INT4",
            "range": [-8, 7],
            "lanes": 3,
            "beats": 6,
            "shape": [1, 6, 3],
            "order": ROW_MAJOR,
        },
        {
            "name": "m_axis_0",
            "direction": "out",
            "tdata": 8,
            "clock": "ap_clk",
            "tensor": "y",
            "element": "UINT8",
            "range": None,
            "lanes": 1,
            "beats": 6,
            "shape": [1, 6],
            "order": TRANSPOSED,
        },
    ]
    # The bus's map is the IP-XACT's: one register block, its window at least 4 KiB.
    assert described["axilite"] == [
        {
            "name": "s_axilite",
            "address_width": 5,
            "data_width": 32,
            "clock": "ap_clk",
            "register_map": [
                {"name": "Reg0", "base": 0, "range": 4096, "width": 32, "usage": "register"}
            ],
        }
    ]
    assert described["aximm"] == []


def test_facts_that_contradict_the_pins_are_refused() -> None:
    def describe(streams: dict[str, dict[str, Any]]) -> None:
        describe_interface(PORTS, streams, part="xczu3eg-sbva484-1-e", period_ns=5.0)

    with pytest.raises(InterfaceError, match="m_axis_0: a stream with no facts"):
        describe({"s_axis_0": FACTS["s_axis_0"]})
    with pytest.raises(InterfaceError, match="s_axis_0: its facts state tdata 16, its pin 12"):
        describe({**FACTS, "s_axis_0": {**FACTS["s_axis_0"], "tdata": 16}})
    with pytest.raises(InterfaceError, match=r"streams the module does not have: \['s_axis_1'\]"):
        describe({**FACTS, "s_axis_1": FACTS["s_axis_0"]})


def test_a_pin_outside_every_interface_is_refused() -> None:
    loose = Signal("debug", Direction.OUT, 1)
    with pytest.raises(InterfaceError, match="debug: a pin outside every interface"):
        describe_interface((*PORTS, loose), FACTS, part="xczu3eg-sbva484-1-e", period_ns=5.0)
