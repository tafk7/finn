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

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, Param, Rejected, Space, derived, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, Signal
from finn.kernels.artifacts.module import Held, Leaf
from finn.kernels.base import Clocking, Kernel
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.streams import Stream
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.target import DspBlock
from finn.kernels.vpc import VpcKernel
from kernels.helpers import placed_dotp

INT4 = DataType["INT4"]
TENSOR = Tensor((4,), ScalarEncoding(INT4))


def codes(result: object) -> set[str]:
    assert isinstance(result, Rejected), result
    return {finding.code for finding in result.findings}


class Probe(Kernel):
    """A model-only leaf: one input port, a mode pin, and a clocking of its choice."""

    id = "test.probe"
    rtl_module = "probe"

    stream: Stream = Param(required=False)
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
        edge = Stream(tensor=TENSOR, port="in0_V")
        kernel = Probe(stream=edge, **facts)  # type: ignore[arg-type]

    return design_space(Placed()).kernel


def test_a_leaf_holds_or_presents_every_input_that_is_no_ports() -> None:
    held = probe().module
    assert isinstance(held, Leaf) and held.held.inputs == (("mode", 1),)
    # The output is no one's to drive: left open, not accounted.
    assert codes(probe(hold_mode=False).query(Kernel.module)) == {"kernel-pins"}


def test_a_port_runs_on_its_kernels_clock() -> None:
    assert codes(probe(port_clock="ap_clk2x").query(Kernel.module)) == {"kernel-clock"}


def test_a_presented_bus_is_accounted_for_by_its_control_node() -> None:
    def memory(writable: bool) -> MemStreamKernel:
        point = design_space(
            MemStreamKernel(
                dtype=INT4, form=vector_major((4,), 2), contents=(1, 2, 3, 4), writable=writable
            )
        )
        return commit(point, {"ram_style": "auto", "pumped_memory": False})

    # Read-only: the bus is held; writable, it is presented, and refused without a node.
    assert memory(False).query(Kernel.pins_accounted) == Available(True)
    assert codes(memory(True).query(Kernel.module)) == {"memstream-control"}


LEAVES = {
    "fifo": lambda: commit(design_space(FifoKernel(word_bits=8, depth=4)), {"ram_style": "auto"}),
    "vpc": lambda: design_space(VpcKernel(element_bits=4, lanes_in=2, lanes_out=3)),
    "input_gen": lambda: commit(
        design_space(InputGeneratorKernel(word_bits=8, frame_words=4, dims=(2, 2), strides=(0, 1))),
        {"ram_style": "auto"},
    ),
    "memstream": lambda: commit(
        design_space(
            MemStreamKernel(dtype=INT4, form=vector_major((4,), 2), contents=(1, 2, 3, 4))
        ),
        {"ram_style": "auto", "pumped_memory": True},
    ),
    "thresholding": lambda: commit(
        design_space(
            ThresholdingAxiKernel(
                input_dtype=INT4,
                threshold_dtype=INT4,
                thresholds=(((-2, 0, 2), (-1, 1, 3)),),
                bias=0,
                depth_trigger_bram=0,
                depth_trigger_uram=0,
            )
        ),
        {"pe": 1, "use_axilite": False, "deep_pipeline": False},
    ),
    "dotp": lambda: placed_dotp(
        PackedDotpKernel,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        result_dtype=DataType["INT8"],
        pe=2,
        simd=2,
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
    ),
}


@pytest.mark.parametrize("name", sorted(LEAVES))
def test_a_leafs_module_is_what_its_requirements_were(name: str) -> None:
    point = LEAVES[name]()
    leaf, requirements = point.module, point.build_requirements
    assert isinstance(leaf, Leaf)
    assert leaf.name == requirements.abi.entry_point.value
    assert (leaf.implementation_id, leaf.implementation_version) == (
        requirements.implementation_id,
        requirements.implementation_version,
    )
    assert leaf.parameters == requirements.parameters
    assert leaf.pins.ports == requirements.abi.ports
    assert leaf.pins.parameters == requirements.abi.parameters
    assert leaf.pins.clock_alignments == requirements.abi.clock_alignments
    assert (*leaf.sources, *leaf.data) == requirements.contributions
    assert (leaf.held.inputs, leaf.held.unused) == (point.tieoffs.inputs, point.tieoffs.unused)
