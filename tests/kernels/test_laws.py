# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The library's interaction laws and the defaults they fix, read from one module: each
transport's pin names and a kernel's overrides, the two clock domains and the doubled
clock, and a kernel whose ports' names collide refused by name."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import Available, Rejected, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Clocking, Endpoint
from finn.kernels.base import Kernel
from finn.kernels.eltwise import EltwiseKernel, c
from finn.kernels.laws import (
    AXIS,
    AXIS_CLOCKING,
    LAWS,
    NATIVE,
    NATIVE_CLOCKING,
    Axis,
    AxisSpelling,
    Native,
    axis_carrier,
    axis_pin,
    doubled,
)
from finn.kernels.port_model import IndexedPort

TARGET, INITIATOR = Endpoint.TARGET, Endpoint.INITIATOR


def test_the_laws_are_ten_each_named_once_in_the_order_of_their_ids() -> None:
    assert [law.id for law in LAWS] == [f"L{number}" for number in range(1, 11)]
    assert len({law.topic for law in LAWS}) == len(LAWS)
    assert all(law.default and law.exception for law in LAWS)


def test_an_axis_port_s_bus_is_named_by_its_role_and_name_unless_its_kernel_states_one() -> None:
    assert AXIS.bus_name("set", TARGET) == "s_axis_set"
    assert AXIS.bus_name("output", INITIATOR) == "m_axis_output"
    assert Axis(bus="s_axis_input").bus_name("x", TARGET) == "s_axis_input"
    assert axis_pin("s_axis", "tdata", AxisSpelling.LOWER) == "s_axis_tdata"
    assert axis_pin("s_axis", "tlast", AxisSpelling.UPPER) == "s_axis_TLAST"
    assert [axis_carrier(bits) for bits in (1, 8, 9, 12)] == [8, 8, 16, 16]
    with pytest.raises(ValueError, match="nonempty"):
        Axis(bus="")


def test_a_native_port_s_pins_take_its_role_s_prefix_unless_its_kernel_states_one() -> None:
    assert NATIVE.pins(TARGET) == ("idat", "ivld", "irdy")
    assert NATIVE.pins(INITIATOR) == ("odat", "ovld", "ordy")
    assert NATIVE.loop_ends(INITIATOR) == "olst"
    stated = Native(prefix="a")
    assert stated.pins(TARGET) == stated.pins(INITIATOR) == ("adat", "avld", "ardy")
    assert stated.loop_ends(TARGET) == "alst"
    named = Native(names=("s_axis_0_tdata", "s_axis_0_tvalid", "s_axis_0_tready"))
    assert named.pins(TARGET) == ("s_axis_0_tdata", "s_axis_0_tvalid", "s_axis_0_tready")
    assert named.loop_ends(TARGET) is None


@pytest.mark.parametrize(
    "stated",
    [
        dict(prefix="a", names=("d", "v", "r")),
        dict(prefix=""),
        dict(names=("d", "v")),
        dict(names=("d", "d", "r")),
    ],
)
def test_a_native_port_states_a_prefix_or_three_distinct_names(stated: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Native(**stated)  # type: ignore[arg-type]


def test_the_two_clock_domains_and_the_doubled_clock_named_after_its_clock() -> None:
    assert AXIS_CLOCKING == Clocking(clock="ap_clk", reset="ap_rst_n", active_low=True)
    assert NATIVE_CLOCKING == Clocking(clock="clk", reset="rst", active_low=False)
    assert Kernel.clock_domain == AXIS_CLOCKING
    pumped = doubled(AXIS_CLOCKING, True)
    assert (pumped.doubled, pumped.doubling) == ("ap_clk2x", True)
    held = doubled(NATIVE_CLOCKING, False)
    assert (held.doubled, held.doubling) == ("clk2x", False)
    assert (held.clock, held.reset, held.active_low) == ("clk", "rst", False)


class Undistinguished(EltwiseKernel):
    """An eltwise whose two inputs take the native default: both ``idat``, ``ivld``,
    ``irdy``."""

    id = "test.laws.undistinguished"
    lhs = IndexedPort(
        label="lhs",
        transport=NATIVE,
        endpoint=TARGET,
        tensor=EltwiseKernel.lhs_tensor,
        schedule=EltwiseKernel.schedule,
        index=EltwiseKernel.indices,
        lanes=(c,),
        clocking=EltwiseKernel.port_clocking,
    )
    rhs = IndexedPort(
        label="rhs",
        transport=NATIVE,
        endpoint=TARGET,
        tensor=EltwiseKernel.rhs_tensor,
        schedule=EltwiseKernel.schedule,
        index=EltwiseKernel.rhs_indices,
        lanes=(c,),
        clocking=EltwiseKernel.port_clocking,
    )


def placed(kind: type[EltwiseKernel]) -> EltwiseKernel:
    lhs = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
    rhs = Tensor((4,), lhs.element)
    space = kind(
        operation="ADD", lhs_tensor=lhs, rhs_tensor=rhs, b_scale=1.0, platform=FULL_DSP48E2
    )
    return design_space(space).with_choices(pe=2)


def test_ports_whose_defaults_give_the_same_pins_are_refused_by_name() -> None:
    assert isinstance(placed(EltwiseKernel).query(EltwiseKernel.netlist), Available)
    found = placed(Undistinguished).query(Undistinguished.netlist)
    assert isinstance(found, Rejected)
    ((finding,),) = [found.findings]
    assert finding.code == "port-pins-collide"
    assert "['idat', 'irdy', 'ivld']" in finding.message
