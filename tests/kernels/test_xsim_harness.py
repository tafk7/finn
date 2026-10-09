# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim testbench (``finn.core.executors.xsim.rtl``): the stream testbench drives
every pin from what the module declares, refusing a stream the module does not present
and an input it declares no value for before anything is built; its stimulus and
expectations are
``$readmemh`` files; its watchdog is derived from the run's beats and names the stream
that stopped; it runs the simulator through FINN's toolchain.

In XSim: a design whose expected words outrun what it produces stops at the watchdog,
which names the output and its beat; and a control bus written something its kernel
does not declare (another table) computes with what was written, so the writes reach
the memories. The conformance cases with pumped compute and memory, and with AXI-Lite,
are ``tests/kernels/test_conformance.py``'s.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.pacing import FREE, STALLED, Pacing
from finn.core.executors.xsim.rtl import (
    LATENCY_ALLOWANCE,
    RESET_CYCLES,
    WATCHDOG_MARGIN,
    WORDS_DIFFER,
    SimulationFailed,
    Undriven,
    Words,
    WordsDiffer,
    simulate,
    stream_bench,
    stream_through,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import pack as pack_beats
from finn.dataflow.traversal import vector_major
from finn.harness.toolchain import SIMULATOR_TOOLS, vivado_simulator
from finn.kernels.artifacts.abi import ClockAlignment
from finn.kernels.artifacts.module import Held, RegisterMap, declared_registers
from finn.kernels.dotp import Int8Dsp58DotpKernel
from finn.kernels.matmul import datatype_range
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.util.toolchain import Selection, Toolchain
from kernels.chain import chain
from kernels.conformance import Sample, place
from kernels.helpers import FULL_DSP48E2, full_platform, point_for
from kernels.sweeps.threshold_numeric import CHANNELS, ELEMENT, PIXELS, VALUES, Case, placed
from kernels.xsim import requires_xsim

CHAIN_IN: dict[str, Words] = {"s_axis_0": ([1, 2, 3], 6)}
CHAIN_OUT: dict[str, Words] = {"m_axis_0": ([4, 5], 16)}


@pytest.mark.parametrize(
    "inputs, outputs, complaint",
    [
        # A name the module does not present (its ports take the shells' names).
        (
            {"in0_V": ([1], 6)},
            {"m_axis_0": ([1], 16)},
            "no input stream in0_V; its inputs: s_axis_0",
        ),
        # A name on the wrong side.
        (
            {"s_axis_0": ([1], 6)},
            {"s_axis_0": ([1], 6)},
            "no output stream s_axis_0; its outputs: m_axis_0",
        ),
    ],
)
def test_a_stream_the_module_lacks_is_refused_naming_its_ports(
    tmp_path: Path, inputs: dict[str, Words], outputs: dict[str, Words], complaint: str
) -> None:
    with pytest.raises(ValueError, match=complaint):
        stream_through(chain().module, tmp_path, inputs=inputs, outputs=outputs)
    assert not any(tmp_path.iterdir())  # refused before anything was built


def test_a_stream_given_no_words_is_an_undriven_input(tmp_path: Path) -> None:
    with pytest.raises(Undriven, match=r"m_axis_0_tready \(stream m_axis_0, given no words\)"):
        stream_bench(chain().module, tmp_path, inputs=CHAIN_IN, outputs={})
    assert not any(tmp_path.iterdir())


def _flat_thresholds() -> Any:
    """thresholding_axi alone, a leaf module: unplaced, it holds its streams and its
    AXI-Lite bus idle."""
    facts = dict(
        input_dtype=DataType["INT4"],
        threshold_dtype=DataType["INT4"],
        thresholds=(((-2, 0, 3),),),
        bias=0,
        platform=FULL_DSP48E2,
    )
    choices = dict(use_axilite=False, deep_pipeline=False, pe=1, ram_style="auto")
    return point_for(ThresholdingAxiKernel, facts, ultra_stages=0, **choices)


def test_a_leaf_s_held_inputs_are_driven_to_their_declared_values(tmp_path: Path) -> None:
    leaf = _flat_thresholds().module
    held = dict(leaf.held.inputs)
    assert held["s_axilite_AWVALID"] == 0 and "s_axis_set_tvalid" in held
    bench = stream_bench(leaf, tmp_path, inputs={}, outputs={})
    for pin, value in held.items():
        assert re.search(rf"logic (\[\d+:0\] )?{pin} = \d+'h{value:x};", bench.text), pin
    with pytest.raises(ValueError, match="the module holds the streams s_axis"):
        stream_bench(leaf, tmp_path / "held", inputs={"s_axis": ([1], 8)}, outputs={})


def test_an_input_the_module_declares_no_value_for_is_refused_naming_it(tmp_path: Path) -> None:
    """Held nothing, the leaf's set stream, given no words, has no value: its AXI-Lite bus is
    the testbench's to write (no write declared) and its other streams are given words."""
    leaf = _flat_thresholds().module
    unheld = replace(leaf, held=Held())
    with pytest.raises(Undriven) as refused:
        stream_bench(unheld, tmp_path, inputs={"s_axis": ([1], 8)}, outputs={"m_axis": ([0], 8)})
    assert str(refused.value).startswith(
        "the testbench has no value for the inputs s_axis_set_tdata (stream s_axis_set, given "
        "no words), s_axis_set_tvalid (stream s_axis_set, given no words): "
    )
    assert not any(tmp_path.iterdir())


def test_stimulus_and_expectations_are_files_beside_the_stream_bench(tmp_path: Path) -> None:
    words = list(range(1000))
    bench = stream_bench(
        chain().module, tmp_path, inputs={"s_axis_0": (words, 6)}, outputs=CHAIN_OUT
    )
    assert (tmp_path / "s_axis_0.input.mem").read_text().split() == [f"{w:02x}" for w in words]
    assert (tmp_path / "m_axis_0.output.mem").read_text().split() == ["0004", "0005"]
    assert '$readmemh("s_axis_0.input.mem", s_axis_0_words)' in bench.text
    assert "3e7" not in bench.text  # no word inlined: the text does not grow with them


def test_the_stream_bench_keeps_each_outputs_words_for_a_decoding(tmp_path: Path) -> None:
    """A word that differs is displayed and counted, not fatal at once: the run fails when
    every output has presented its words, which it writes out first."""
    bench = stream_bench(chain().module, tmp_path, inputs=CHAIN_IN, outputs=CHAIN_OUT)
    assert "m_axis_0_got[m_axis_0_beats] <= m_axis_0_tdata[15:0];" in bench.text
    assert '$writememh("m_axis_0.received.mem", m_axis_0_got);' in bench.text
    assert f'$fatal(1, "{WORDS_DIFFER}: %0d output words", differing);' in bench.text
    assert 'if (!differing) $display("m_axis_0 word %0d: %h != %h"' in bench.text


@pytest.mark.parametrize("pacing", [FREE, STALLED])
def test_the_watchdog_is_derived_from_the_paced_beats_and_names_each_stream(
    tmp_path: Path, pacing: Pacing
) -> None:
    bench = stream_bench(
        chain().module, tmp_path, inputs=CHAIN_IN, outputs=CHAIN_OUT, pacing=pacing, cycles=12
    )
    paced = pacing.input(0).cycles(3) + pacing.output(0).cycles(2)
    budget = RESET_CYCLES + WATCHDOG_MARGIN * (paced + 12) + LATENCY_ALLOWANCE
    assert bench.budget == budget
    assert f"if (cycle == {budget})" in bench.text
    for port, total in (("s_axis_0", 3), ("m_axis_0", 2)):
        assert f'"watchdog: {port} stopped at beat %0d of {total}' in bench.text


def test_a_pumped_module_gets_its_doubled_clock_aligned(tmp_path: Path) -> None:
    """INT8 dotp on DSP58 with its compute pumped, between boundary channels: the root takes
    ``ap_clk2x``, aligned with ``ap_clk``, and the testbench drives it at half the period."""
    inputs = {
        "x_channel": Tensor((2, 6), ScalarEncoding(DataType["INT8"])),
        "w_channel": Tensor((6, 4), ScalarEncoding(DataType["INT8"])),
    }
    point = place(
        Int8Dsp58DotpKernel,
        Sample("pumped", {"pe": 2, "simd": 3}),
        inputs,
        {"y_channel": (2, 4)},
        choices={"compute_pumping": True},
        facts={
            "platform": full_platform(DspBlock.DSP58),
            "form": Form.DENSE,
            "result_range": datatype_range(6, DataType["INT8"], DataType["INT8"]),
        },
    )
    module = point.module
    assert module.abi.clock_alignments == (ClockAlignment("ap_clk", "ap_clk2x"),)
    words: dict[str, Words] = {name: ([0], 8) for name in inputs}
    bench = stream_bench(module, tmp_path, inputs=words, outputs={"y_channel": ([0], 8)})
    # Half the clock's period; each rising edge of ap_clk is one of ap_clk2x.
    assert (
        "always #2.5 begin\n        ap_clk2x = !ap_clk2x;\n        if (ap_clk2x) ap_clk = !ap_clk;"
        in bench.text
    )
    assert "always @(posedge ap_clk) #1 begin" in bench.text  # never on an ap_clk2x edge


def _written(initial: tuple[tuple[int, ...], ...], written: tuple[tuple[int, ...], ...]) -> Any:
    return placed(Case("written", initial, written))


def test_a_control_bus_is_written_what_its_kernel_declares(tmp_path: Path) -> None:
    point = _written(((-3, 0, 2),), ((-6, -1, 5),))
    declared = declared_registers(point.module)
    assert declared == {"s_axilite": point.activate.register_map}
    # One 32-bit word a threshold of the shared row (C = 1, N = 3): the word select alone.
    assert declared["s_axilite"] == RegisterMap(((0, 0xD), (4, 0x0), (8, 0x2)))
    stimulus = {"s_axis_0": ([0] * (PIXELS * CHANNELS // 4), 16)}
    expected = {"m_axis_0": ([0] * (PIXELS * CHANNELS // 4), 8)}
    bench = stream_bench(point.module, tmp_path, inputs=stimulus, outputs=expected)
    assert (tmp_path / "s_axilite.writes.mem").read_text().split() == [
        "00000000d",
        "400000000",
        "800000002",
    ]
    assert "wire configured = s_axilite_written == 3;" in bench.text
    other = RegisterMap(((0, 1),))
    stream_bench(
        point.module,
        tmp_path / "other",
        inputs=stimulus,
        outputs=expected,
        registers={"s_axilite": other},
    )
    assert (tmp_path / "other/s_axilite.writes.mem").read_text().split() == ["000000001"]
    with pytest.raises(
        ValueError, match="no control bus s_axilite_1; its control buses: s_axilite"
    ):
        stream_bench(
            point.module,
            tmp_path / "unknown",
            inputs=stimulus,
            outputs=expected,
            registers={"s_axilite_1": other},
        )


def _toolchain(tmp_path: Path, xsim_prints: str) -> Toolchain:
    """A toolchain whose simulator tools are scripts in a command directory: each
    records its arguments, and xsim prints ``xsim_prints``."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in SIMULATOR_TOOLS:
        printed = xsim_prints if tool == "xsim" else ""
        script = tools / tool
        script.write_text(f'#!/bin/sh\necho "$0 $*" >> "{tmp_path}/calls"\necho "{printed}"\n')
        script.chmod(0o755)
    vivado = tmp_path / "vivado"
    environment = {"PATH": os.defpath, "XILINX_VIVADO": str(vivado)}
    return Toolchain(Selection(command_dir=str(tools)), environment)


def test_simulate_runs_the_tools_of_its_toolchain(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path, "PASS")
    (tmp_path / "sim").mkdir()
    output = simulate(["a.sv"], "module check; endmodule\n", tmp_path / "sim", toolchain=toolchain)
    assert "PASS" in output
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [Path(call.split()[0]).name for call in calls] == list(SIMULATOR_TOOLS)
    assert f"{tmp_path}/vivado/data/verilog/src/glbl.v" in calls[0]  # the toolchain's Vivado
    assert vivado_simulator(toolchain)


def test_simulate_without_pass_fails_as_an_assertion(tmp_path: Path) -> None:
    (tmp_path / "sim").mkdir()
    with pytest.raises(SimulationFailed, match="watchdog") as failed:
        simulate([], "", tmp_path / "sim", toolchain=_toolchain(tmp_path, "watchdog"))
    assert isinstance(failed.value, AssertionError)  # conformance collects these


def test_a_toolchain_naming_no_vivado_has_no_simulator(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path, "PASS")
    bare = Toolchain(toolchain.selection, {"PATH": os.defpath})
    assert not vivado_simulator(bare)
    with pytest.raises(LookupError, match="XILINX_VIVADO"):
        simulate([], "", tmp_path, toolchain=bare)


# -- in XSim ---------------------------------------------------------------------------------


@requires_xsim
def test_a_stopped_stream_fires_the_watchdog_naming_it_and_its_beat(tmp_path: Path) -> None:
    """The Chain given half a row (one of its two input beats a row) produces nothing: the
    watchdog names the output, stopped before its first beat, and not the input, whose
    one word was taken."""
    with pytest.raises(SimulationFailed) as failed:
        stream_through(
            chain().module,
            tmp_path,
            inputs={"s_axis_0": ([1], 6)},
            outputs=CHAIN_OUT,
            pacing=FREE,
        )
    message = str(failed.value)
    assert "watchdog: m_axis_0 stopped at beat 0 of 2, its last handshake at cycle -1" in message
    assert "s_axis_0 stopped" not in message
    assert "watchdog: the run did not complete within" in message


@requires_xsim
def test_words_that_differ_are_raised_as_they_arrived(tmp_path: Path) -> None:
    """Thresholds on one shared row, one expected word altered: the run fails naming that
    word, and every word the hardware presented is returned, the altered one as computed."""
    case = Case("shared", ((-3, 0, 2),))
    point = placed(case)
    form = vector_major((PIXELS, CHANNELS), case.pe)
    bits = point.activate.result_dtype.bitwidth()
    levels = (VALUES[..., None] >= np.array(case.initial[0])).sum(axis=-1)
    words = pack_beats(form, levels.ravel().tolist(), bits)
    wrong = [*words[:1], words[1] ^ 1, *words[2:]]
    with pytest.raises(WordsDiffer) as differ:
        stream_through(
            point.module,
            tmp_path,
            inputs={
                "s_axis_0": (
                    list(pack_beats(form, VALUES.ravel().tolist(), ELEMENT.bitwidth())),
                    ELEMENT.bitwidth() * case.pe,
                )
            },
            outputs={"m_axis_0": (wrong, bits * case.pe)},
            pacing=FREE,
        )
    assert differ.value.received == {"m_axis_0": tuple(words)}
    assert "m_axis_0 word 1: " in str(differ.value)


@requires_xsim
@pytest.mark.parametrize("pacing", [FREE, STALLED], ids=["free", "stalled"])
def test_writes_its_kernel_does_not_declare_reach_the_memories(
    tmp_path: Path, pacing: Pacing
) -> None:
    """Thresholds built with one shared row, written another through AXI-Lite (the writes a
    kernel holding the other row declares): every level is the written row's."""
    initial, written = ((-3, 0, 2),), ((-6, -1, 5),)
    point = _written(initial, written)
    registers = declared_registers(_written(written, written).module)
    pe = point.activate.pe
    form = vector_major((PIXELS, CHANNELS), pe)
    levels = (VALUES[..., None] >= np.array(written[0])).sum(axis=-1)
    assert not np.array_equal(levels, (VALUES[..., None] >= np.array(initial[0])).sum(axis=-1))
    result_bits = point.activate.result_dtype.bitwidth()
    stream_through(
        point.module,
        tmp_path,
        inputs={"s_axis_0": (list(pack_beats(form, VALUES.ravel().tolist(), 4)), 4 * pe)},
        outputs={
            "m_axis_0": (
                list(pack_beats(form, levels.ravel().tolist(), result_bits)),
                result_bits * pe,
            )
        },
        pacing=pacing,
        registers=registers,
        cycles=point.cycles,
    )
    assert ELEMENT.bitwidth() == 4
