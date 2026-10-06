# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A root's clocks: instances are driven by pin role, never by pin name.

A root is one composed module with ``ap_clk`` and ``ap_rst_n``, plus
``ap_clk2x`` only when an instance declares a clock at twice ``ap_clk``. An
unpumped dotp holds its 2x clock input low, so an unpumped MatMul's root has
no ``ap_clk2x`` pin. Each instance clock and reset pin is driven by its role
when the module is emitted (``finn.kernels.artifacts.build``); the role rules
on hand-built values are tested in ``artifacts/test_netlist.py``.
"""

from qonnx.core.datatype import DataType

from finn.kernels.artifacts.abi import (
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Free,
    Reset,
    Signal,
)
from finn.kernels.artifacts.build import netlist
from kernels.helpers import FULL_DSP58, matmul_assembly, placed

MATMUL_FACTS = dict(
    m=2,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    pe=2,
    simd=2,
    platform=FULL_DSP58,
    core="packed",
)


def assigns(module) -> set[str]:
    text = netlist(module, "top")
    return {line.strip() for line in text.splitlines() if line.strip().startswith("assign ")}


def test_an_unpumped_matmul_has_one_clock_and_holds_the_2x_input():
    built = matmul_assembly(**MATMUL_FACTS)
    top = {port.name: port for port in built.module.abi.pins}
    assert "ap_clk2x" not in top
    assert top["ap_clk"] == Signal("ap_clk", Direction.IN, 1, Clock(Free()))
    assert top["ap_rst_n"].role == Reset(True, True, ("ap_clk",))
    assert placed(built.module, "matmul.compute.packed").held.inputs == (("ap_clk2x", 0),)
    driven = assigns(built.module)
    compute, replay = "n__u_matmul_compute_packed", "n__u_x_adapter_input_gen_input_gen"
    assert {
        f"assign {compute}__ap_clk = ap_clk;",
        f"assign {compute}__ap_rst_n = ap_rst_n;",
        f"assign {compute}__ap_clk2x = 1'h0;",
        f"assign {replay}__clk = ap_clk;",
        # The active-high adapter reset is driven inverted from the active-low top reset.
        f"assign {replay}__rst = !ap_rst_n;",
    } <= driven


def test_a_pumped_matmul_adds_the_2x_clock_and_its_alignment():
    built = matmul_assembly(**{**MATMUL_FACTS, "compute_pumping": True})
    abi = built.module.abi
    ports = {port.name: port for port in abi.pins}
    assert [port.name for port in abi.pins][:3] == ["ap_clk", "ap_clk2x", "ap_rst_n"]
    assert ports["ap_clk2x"].role == Clock(Derived("ap_clk", 2))
    assert ports["ap_rst_n"].role == Reset(True, True, ("ap_clk", "ap_clk2x"))
    assert abi.clock_alignments == (ClockAlignment("ap_clk", "ap_clk2x"),)
    assert "assign n__u_matmul_compute_packed__ap_clk2x = ap_clk2x;" in assigns(built.module)
