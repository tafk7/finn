# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's module: a leaf binds one FinnLib module and accounts for its pins at creation.

Every input among a leaf's other pins is held or presented as a control bus
(``pins_accounted``), and every port runs on its clock (``clocked``); both are
refused when the kernel is created, not when it is composed. Its ``module`` is
the ``Leaf``; a kernel with children merges its members' netlists into one
``Composed`` module.
"""

from __future__ import annotations

from collections.abc import Mapping

from qonnx.core.datatype import DataType

from finn.core.space import Available, Param, Rejected, Space, derived, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, Signal
from finn.kernels.artifacts.module import Held, Leaf
from finn.kernels.base import Clocking, Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort
from kernels.helpers import FULL_DSP48E2

INT4 = DataType["INT4"]
TENSOR = Tensor((4,), ScalarEncoding(INT4))


def codes(result: object) -> set[str]:
    assert isinstance(result, Rejected), result
    return {finding.code for finding in result.findings}


class Probe(Kernel):
    """A model-only leaf: one input port, a mode pin, and a clocking of its choice."""

    id = "test.probe"
    rtl_module = "probe"

    stream: Channel = Param(required=False)
    hold_mode: bool = Param(default=True)
    port_clock: str = Param(default="ap_clk")

    x = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        stream=stream,
        sequence=BeatSequence(vector_major((4,), 1)),
        clock=port_clock,
    )

    @derived
    def clocking(self) -> Clocking:
        return Clocking(doubled="ap_clk2x", doubling=True)

    def other_pins(self) -> tuple[Signal | Bus, ...]:
        return (Signal("mode", Direction.IN, 2), Signal("busy", Direction.OUT, 1))

    def held(self) -> Held:
        return Held((("mode", 1),)) if self.hold_mode else Held()

    def parameters(self) -> Mapping[str, int | str]:
        return {}


def probe(**facts: object) -> Probe:
    class Placed(Space):
        edge = Channel(tensor=TENSOR, port="in0_V", platform=FULL_DSP48E2)
        kernel = Probe(stream=edge, **facts)  # type: ignore[arg-type]

    return design_space(Placed()).kernel


def test_a_leaf_holds_or_presents_every_input_that_is_no_ports() -> None:
    held = probe().module
    assert isinstance(held, Leaf) and held.held.inputs == (("mode", 1),)
    # The output is no one's to drive: left open, not accounted.
    assert codes(probe(hold_mode=False).query(Kernel.module)) == {"kernel-pins"}


def test_a_port_runs_on_its_kernels_clock() -> None:
    assert codes(probe(port_clock="ap_clk2x").query(Kernel.module)) == {"kernel-clock"}


def test_a_held_bus_is_accounted_for() -> None:
    point = design_space(
        MemStreamKernel(
            platform=FULL_DSP48E2, dtype=INT4, form=vector_major((4,), 2), contents=(1, 2, 3, 4)
        )
    )
    memory = commit(point, {"ram_style": "auto", "pumped_memory": False})
    assert memory.query(Kernel.pins_accounted) == Available(True)
