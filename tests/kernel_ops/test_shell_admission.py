# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell root's admission (``Shell.interfaces``): what the module presents, within
what its shell's row takes.

The Zynq shell (``pynq``) takes nine AXI-Lite buses, the partition's and its ends'
together (each ``IODMA_hls`` end presents one), gives a partition no memory port of its
own and supplies no doubled clock; the ``ip`` shell bounds neither count and supplies
the clock. Each refusal is named: ``interface-budget-exceeded``, ``clock-unavailable``.

No KernelOp places a control bus, so a partition of KernelOps presents no AXI-Lite bus:
the buses are counted here on a shell root built from the kernels directly, AXI-Lite
thresholdings each on its ``ControlBus``, through the same ``Shell`` class and rows the
shell root reads. No kernel initiates a memory port (no bus protocol
states one), so that count is always zero.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from kernels.helpers import with_adapter_memories, with_direct_transports
from qonnx.core.datatype import DataType

import finn.custom_op.kernels.shell as shell
from finn.core.space import composite, design_space
from finn.core.space.errors import ValueUnavailableError
from finn.custom_op.kernels.base import KernelOpError, kernel_op, write_target
from finn.custom_op.kernels.shell import Shell, admission_refusal, shell_root
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel
from finn.kernels.configure import commit, describe
from finn.kernels.control import ControlBus
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.platform import IP_ROW, ShellRow, shell_row
from finn.transformation.kernels import explore_kernel_choices
from finn.transformation.kernels.package import configured_root
from kernel_ops.models import MATMUL, kernel_model
from kernel_ops.tfc import ULTRA96

PYNQ_ROW = shell_row("pynq", "Ultra96")

UINT2 = DataType["UINT2"]
TENSOR = Tensor((3, 2), ScalarEncoding(UINT2))
"""Each channel's tensor: three rows of two UINT2 channels."""
TABLE = (((0, 1, 2), (0, 1, 2)),)
"""Each thresholding's table: two channels of three thresholds over UINT2, to UINT2."""


def thresholdings(count: int, row: ShellRow) -> Any:
    """A shell root for ``row`` of ``count`` AXI-Lite thresholdings in a row, each
    presenting its bus (``t<i>_s_axilite``), configured: every thresholding
    runtime-writable, at its baseline otherwise, every transport direct."""
    platform = ULTRA96.platform
    offered: dict[str, Any] = {"end_offer": row.ends} if row.ends else {}
    x = Channel(tensor=TENSOR, platform=platform, port="s_axis_0", **offered)
    y = Channel(tensor=TENSOR, platform=platform, port="m_axis_0", **offered)
    inner = [Channel(tensor=TENSOR, platform=platform) for _ in range(count - 1)]
    sides = [x, *inner, y]
    members: dict[str, object] = {"x": x, **{f"c{index}": each for index, each in enumerate(inner)}}
    for index in range(count):
        bus = ControlBus(port=f"t{index}_s_axilite")
        members[f"bus{index}"] = bus
        members[f"t{index}"] = ThresholdingAxiKernel(
            input_dtype=UINT2,
            threshold_dtype=UINT2,
            thresholds=TABLE,
            bias=0,
            input_channel=sides[index],
            output_channel=sides[index + 1],
            control=bus,
            platform=platform,
        )
    root: Any = composite(f"shell{count}", {**members, "y": y}, base=Shell)
    choices: dict[str, object] = {}
    for index in range(count):
        held = {"use_axilite": True, "pe": 1, "deep_pipeline": False, "ram_style": "auto"}
        choices |= {f"t{index}.{key}": value for key, value in held.items()}
        choices[f"t{index}.ultra_stages"] = 0
    point = commit(with_direct_transports(design_space(root(row=row))), choices)
    return with_adapter_memories(point)


def test_eight_axi_lite_thresholdings_and_two_ends_exceed_the_zynq_shells_nine() -> None:
    point = thresholdings(8, PYNQ_ROW)
    assert [end.control_buses for end in point.x.placed_end + point.y.placed_end] == [1, 1]
    assert admission_refusal(point) == (
        "interfaces: interface-budget-exceeded: the 'pynq' shell takes 9 AXI-Lite buses; "
        "the partition presents 8 and its ends 2"
    )
    # Its module is refused by the same admission.
    with pytest.raises(ValueUnavailableError) as error:
        _ = point.module
    assert "interface-budget-exceeded" in describe([error.value.result])


def test_seven_and_two_ends_fill_the_budget() -> None:
    assert admission_refusal(thresholdings(7, PYNQ_ROW)) is None


def test_the_ip_shell_admits_the_same_partition_presenting_every_bus() -> None:
    point = thresholdings(8, IP_ROW)
    assert admission_refusal(point) is None
    assert point.x.placed_end == ()
    ports = [pin.name for pin in point.module.abi.pins if getattr(pin, "protocol", None)]
    assert [port for port in ports if port.endswith("_s_axilite")] == [
        f"t{index}_s_axilite" for index in range(8)
    ]


def test_a_shell_root_of_kernel_ops_counts_its_ends() -> None:
    """The Chain in the Zynq shell: no AXI-Lite bus of its own and two ends, within nine;
    on a row taking one bus, its ends alone exceed it."""
    model = kernel_model()
    write_target(model, ULTRA96)
    explore_kernel_choices(model, [])
    configured_root(model, "chain")
    narrow = replace(PYNQ_ROW, control_budget=1)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(shell, "shell_row", lambda *_: narrow)
        root = shell_root(model, model.graph.node, name="chain")
        assert root.row == narrow
        with pytest.raises(KernelOpError, match="refused by the 'pynq' shell"):
            explore_kernel_choices(model, [])
        with pytest.raises(KernelOpError, match="takes 1 AXI-Lite buses; the partition presents 0"):
            configured_root(model, "chain")


def test_a_doubled_clock_the_row_does_not_supply_is_refused() -> None:
    """The shell row owns the doubled clock (SZ11 (e)): the Chain, its first MatMul's
    compute pumped, which its kernel offers on any platform, is admitted by the ip row,
    which supplies ap_clk2x, and refused by Ultra96's pynq row, which does not:
    ``clock-unavailable``."""
    model = kernel_model()
    pumped = {**MATMUL, "compute.packed.compute_pumping": True}
    kernel_op(model, model.graph.node[0]).save(pumped)
    point, _ = configured_root(model, "chain")
    assert point.row is IP_ROW and IP_ROW.clk2x
    assert point.module.abi.clock_alignments
    write_target(model, ULTRA96)
    assert not shell_root(model, model.graph.node, name="chain").row.clk2x
    with pytest.raises(
        KernelOpError,
        match="clock-unavailable: the partition takes an aligned doubled clock "
        r"\(ap_clk2x\), which the 'pynq' shell does not supply",
    ):
        configured_root(model, "chain")


def test_the_shells_netlist_presents_each_bus_with_its_writes() -> None:
    """The shell root's netlist is ``Kernel``'s composition of its members, each bus it
    presents with the writes its thresholding's table takes."""
    point = thresholdings(2, IP_ROW)
    assert point.fragment == point.composed_fragment()
    assert [(item.port, item.registers) for item in point.fragment.exports] == [
        (f"t{index}_s_axilite", getattr(point, f"t{index}").register_map) for index in range(2)
    ]
    assert all(len(item.registers.writes) == 6 for item in point.fragment.exports)
