# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell's admission (``finn.platform.shells.budgets``): what the module presents,
within what its shell's row takes.

The Zynq shell (``pynq``) takes nine AXI-Lite buses, the partition's and its ends'
together (each ``IODMA_hls`` end presents one), gives a partition no memory port of its
own and supplies no doubled clock; the ``ip`` shell bounds neither count and supplies
the clock. Each refusal is named: ``interface-budget-exceeded``, ``clock-unavailable``.

No KernelOp names a control port, so a partition of KernelOps presents no AXI-Lite bus:
the buses are counted here on a network built from the kernels directly, AXI-Lite
thresholdings each given its port, its module's counts (``bus_counts``) admitted by the
same ``budgets`` and rows a partition's network reads. No kernel initiates a memory port
(no bus protocol states one), so that count is always zero.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from kernels.helpers import built, choose, with_adapter_memories, with_direct_transports
from qonnx.core.datatype import DataType

import finn.custom_op.kernels.network as graph_network
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.reporting.findings import Finding
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network
from finn.custom_op.kernels.shell import configured_network
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.network import Interface, Network, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.platform.shells import IP_ROW, ShellRow, budgets, bus_counts, shell_row
from finn.transformation.kernels import explore_kernel_choices
from kernel_ops.models import MATMUL, configure_partition, kernel_model
from kernel_ops.tfc import ULTRA96

PYNQ_ROW = shell_row("pynq", "Ultra96")

UINT2 = DataType["UINT2"]
TENSOR = Tensor((3, 2), ScalarEncoding(UINT2))
"""Each channel's tensor: three rows of two UINT2 channels."""
TABLE = (((0, 1, 2), (0, 1, 2)),)
"""Each thresholding's table: two channels of three thresholds over UINT2, to UINT2."""


def thresholdings(count: int, row: ShellRow) -> Network:
    """A network for ``row`` of ``count`` AXI-Lite thresholdings in a row, each
    presenting its bus (``t<i>_s_axilite``), configured: every thresholding
    runtime-writable, at its baseline otherwise, every transport direct."""
    platform = ULTRA96.platform
    offered: dict[str, Any] = {"end_offer": row.ends} if row.ends else {}
    # Each channel takes the ends of the thresholdings on either side of it.
    names = ["x", *(f"c{index}" for index in range(count - 1)), "y"]
    channels: dict[str, Any] = {}
    for index, name in enumerate(names):
        sides: dict[str, Any] = {}
        if index > 0:
            sides["producer"] = Interface(f"t{index - 1}", "output_end")
        if index < count:
            sides["consumer"] = Interface(f"t{index}", "input_end")
        boundary = {"port": "s_axis_0" if index == 0 else "m_axis_0", **offered}
        channels[name] = wire(
            tensor=TENSOR,
            platform=platform,
            **sides,
            **(boundary if name in ("x", "y") else {}),
        )
    members: dict[str, Any] = {name: channels[name] for name in names[:-1]}
    for index in range(count):
        members[f"t{index}"] = ThresholdingAxiKernel(
            threshold_dtype=UINT2,
            thresholds=TABLE,
            bias=0,
            input_tensor=TENSOR,
            control=f"t{index}_s_axilite",
            platform=platform,
        )
    members["y"] = channels["y"]
    choices: dict[str, object] = {}
    for index in range(count):
        held = {"use_axilite": True, "pe": 1, "deep_pipeline": False, "ram_style": "auto"}
        choices |= {f"t{index}.{key}": value for key, value in held.items()}
        choices[f"t{index}.ultra_stages"] = 0
    point = with_direct_transports(choose(Network(members), choices))
    return with_adapter_memories(point)


def ends_of(point: Network) -> tuple[Any, ...]:
    """The ends the shell places on the network's boundary channels."""
    return tuple(point.member("x").placed_end) + tuple(point.member("y").placed_end)


def admission(point: Network, row: ShellRow) -> tuple[Finding, ...]:
    """``row``'s budgets of ``point``'s module and of its ends' AXI-Lite buses."""
    ends = sum(end.control_buses for end in ends_of(point))
    return budgets(row, bus_counts(built(point, "shell").abi), ends)


def test_eight_axi_lite_thresholdings_and_two_ends_exceed_the_zynq_shells_nine() -> None:
    point = thresholdings(8, PYNQ_ROW)
    assert [end.control_buses for end in ends_of(point)] == [1, 1]
    (found,) = admission(point, PYNQ_ROW)
    assert str(found) == (
        "budgets: interface-budget-exceeded: the 'pynq' shell takes 9 AXI-Lite buses; "
        "the partition presents 8 and its ends 2"
    )


def test_seven_and_two_ends_fill_the_budget() -> None:
    assert admission(thresholdings(7, PYNQ_ROW), PYNQ_ROW) == ()


def test_the_ip_shell_admits_the_same_partition_presenting_every_bus() -> None:
    point = thresholdings(8, IP_ROW)
    assert admission(point, IP_ROW) == ()
    assert tuple(point.member("x").placed_end) == ()
    ports = [pin.name for pin in built(point, "shell").abi.pins if getattr(pin, "protocol", None)]
    assert [port for port in ports if port.endswith("_s_axilite")] == [
        f"t{index}_s_axilite" for index in range(8)
    ]


def test_a_partition_of_kernel_ops_counts_its_ends() -> None:
    """The Chain in the Zynq shell: no AXI-Lite bus of its own and two ends, within nine;
    on a row taking one bus, its ends alone exceed it."""
    model = kernel_model()
    write_target(model, ULTRA96)
    explore_kernel_choices(model)
    configured_network(model, "chain")
    narrow = replace(PYNQ_ROW, control_budget=1)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(graph_network, "shell_row", lambda *_: narrow)
        bound = network(model, model.graph.node, name="chain")
        assert bound.row == narrow
        with pytest.raises(KernelOpError, match="refused by the 'pynq' shell") as explored:
            explore_kernel_choices(model)
        with pytest.raises(
            KernelOpError, match="takes 1 AXI-Lite buses; the partition presents 0"
        ) as packaged:
            configured_network(model, "chain")
    # Each refusal carries the shell's findings, by code.
    for refused in (explored, packaged):
        assert [f.code for f in refused.value.findings] == ["interface-budget-exceeded"]


def test_a_doubled_clock_the_row_does_not_supply_is_refused() -> None:
    """The shell row owns the doubled clock (SZ11 (e)): the Chain, its first MatMul's
    compute pumped, which its kernel offers on any platform, is admitted by the ip row,
    which supplies ap_clk2x, and refused by Ultra96's pynq row, which does not:
    ``clock-unavailable``."""
    model = kernel_model()
    pumped = {**MATMUL, "compute.packed.compute_pumping": True}
    kernel_op(model, model.graph.node[0]).save(pumped)
    configure_partition(model)
    point, _ = configured_network(model, "chain")
    assert point.row is IP_ROW and IP_ROW.clk2x
    assert point.module.abi.clock_alignments
    write_target(model, ULTRA96)
    assert not network(model, model.graph.node, name="chain").row.clk2x
    with pytest.raises(
        KernelOpError,
        match="clock-unavailable: the partition takes an aligned doubled clock "
        r"\(ap_clk2x\), which the 'pynq' shell does not supply",
    ):
        configured_network(model, "chain")


def test_the_shells_netlist_presents_each_bus_with_its_writes() -> None:
    """A network's netlist presents each bus its thresholdings present, with the writes
    each table takes, under each member's node."""
    point = thresholdings(2, IP_ROW)
    exports = built(point, "shell").fragment.exports
    assert [(item.instance, item.port, item.registers) for item in exports] == [
        (f"t{index}", f"t{index}_s_axilite", point.member(f"t{index}").register_map)
        for index in range(2)
    ]
    assert all(len(item.registers.writes) == 6 for item in exports)
