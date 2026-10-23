# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim testbench (``finn.core.rtlsim_bench``): the stream testbench drives
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
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from support.chain import STEM, chain
from support.conformance import Sample, design_module, place
from support.helpers import (
    FULL_DSP48E2,
    built,
    cycles_of,
    dotp_extents,
    full_platform,
    point_for,
)
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import (
    LATENCY_ALLOWANCE,
    RESET_CYCLES,
    WATCHDOG_MARGIN,
    WORDS_DIFFER,
    SimulationFailed,
    Tap,
    Undriven,
    Words,
    WordsDiffer,
    simulate,
    stream_bench,
    stream_through,
)
from finn.core.rtlsim_pacing import FREE, STALLED, Pacing
from finn.core.space import Available
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import pack as pack_beats
from finn.dataflow.traversal import vector_major
from finn.kernels.artifacts.abi import ClockAlignment
from finn.kernels.artifacts.build import instance_net
from finn.kernels.artifacts.module import Held, RegisterMap, declared_registers
from finn.kernels.artifacts.sources import include_directories
from finn.kernels.dotp import Int8Dsp58DotpKernel
from finn.kernels.library import library_module
from finn.kernels.matmul import datatype_range
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import DspBlock
from finn.toolchain import xsim
from finn.toolchain.launcher import (
    Selection,
    Toolchain,
    ToolFailed,
    ToolUnavailable,
    host_toolchain,
)
from finn.toolchain.xsim import TOOLS, available
from kernels.sweeps.threshold_numeric import (
    CHANNELS,
    ELEMENT,
    PIXELS,
    VALUES,
    Case,
    placed,
    stem,
)

CHAIN_IN: dict[str, Words] = {"s_axis_0": ([1, 2, 3], 6)}


def chain_module() -> Any:
    """The Chain's module (``support.chain``)."""
    return built(chain(), STEM)


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
        stream_through(
            chain_module(), tmp_path, inputs=inputs, outputs=outputs, toolchain=host_toolchain()
        )
    assert not any(tmp_path.iterdir())  # refused before anything was built


def test_a_stream_given_no_words_is_an_undriven_input(tmp_path: Path) -> None:
    with pytest.raises(Undriven, match=r"m_axis_0_tready \(stream m_axis_0, given no words\)"):
        stream_bench(
            chain_module(), tmp_path, inputs=CHAIN_IN, outputs={}, toolchain=host_toolchain()
        )
    assert not any(tmp_path.iterdir())


def _thresholds_alone() -> Any:
    """thresholding_axi alone, a leaf module on its input: it holds its set stream (one
    set) and its AXI-Lite bus idle."""
    facts = dict(
        input_tensor=Tensor((1, 1), ScalarEncoding(DataType["INT4"])),
        threshold_dtype=DataType["INT4"],
        thresholds=IntegerTensorValue.of((((-2, 0, 3),),)),
        bias=0,
        platform=FULL_DSP48E2,
    )
    choices = dict(use_axilite=False, deep_pipeline=False, pe=1, ram_style="auto")
    return point_for(ThresholdingAxiKernel, facts, ultra_stages=0, **choices)


def test_a_leaf_s_held_inputs_are_driven_to_their_declared_values(tmp_path: Path) -> None:
    leaf = library_module(_thresholds_alone())
    held = dict(leaf.held.inputs)
    assert held["s_axilite_AWVALID"] == 0 and "s_axis_set_tvalid" in held
    bench = stream_bench(
        leaf,
        tmp_path,
        inputs={"s_axis": ([1], 8)},
        outputs={"m_axis": ([0], 8)},
        toolchain=host_toolchain(),
    )
    for pin, value in held.items():
        assert re.search(rf"logic (\[\d+:0\] )?{pin} = \d+'h{value:x};", bench.text), pin
    with pytest.raises(ValueError, match="the module holds the streams s_axis_set"):
        stream_bench(
            leaf,
            tmp_path / "held",
            inputs={"s_axis_set": ([1], 8)},
            outputs={},
            toolchain=host_toolchain(),
        )


def test_an_input_the_module_declares_no_value_for_is_refused_naming_it(tmp_path: Path) -> None:
    """Held nothing, the leaf's set stream, given no words, has no value: its AXI-Lite bus is
    the testbench's to write (no write declared) and its other streams are given words."""
    leaf = library_module(_thresholds_alone())
    unheld = replace(leaf, held=Held())
    with pytest.raises(Undriven) as refused:
        stream_bench(
            unheld,
            tmp_path,
            inputs={"s_axis": ([1], 8)},
            outputs={"m_axis": ([0], 8)},
            toolchain=host_toolchain(),
        )
    assert str(refused.value).startswith(
        "the testbench has no value for the inputs s_axis_set_tdata (stream s_axis_set, given "
        "no words), s_axis_set_tvalid (stream s_axis_set, given no words): "
    )
    assert not any(tmp_path.iterdir())


def test_stimulus_and_expectations_are_files_beside_the_stream_bench(tmp_path: Path) -> None:
    words = list(range(1000))
    bench = stream_bench(
        chain_module(),
        tmp_path,
        inputs={"s_axis_0": (words, 6)},
        outputs=CHAIN_OUT,
        toolchain=host_toolchain(),
    )
    assert (tmp_path / "s_axis_0.input.mem").read_text().split() == [f"{w:02x}" for w in words]
    assert (tmp_path / "m_axis_0.output.mem").read_text().split() == ["0004", "0005"]
    assert '$readmemh("s_axis_0.input.mem", s_axis_0_words)' in bench.text
    assert "3e7" not in bench.text  # no word inlined: the text does not grow with them


def test_the_stream_bench_keeps_each_outputs_words_for_a_decoding(tmp_path: Path) -> None:
    """A word that differs is displayed and counted, not fatal at once: the run fails when
    every output has presented its words, which it writes out first."""
    bench = stream_bench(
        chain_module(), tmp_path, inputs=CHAIN_IN, outputs=CHAIN_OUT, toolchain=host_toolchain()
    )
    assert "m_axis_0_got[m_axis_0_beats] <= m_axis_0_tdata[15:0];" in bench.text
    assert '$writememh("m_axis_0.received.mem", m_axis_0_got);' in bench.text
    assert f'$fatal(1, "{WORDS_DIFFER}: %0d output words", differing);' in bench.text
    assert 'if (!differing) $display("m_axis_0 word %0d: %h != %h"' in bench.text


def test_a_tap_reads_a_link_at_its_producers_nets_and_only_when_asked(tmp_path: Path) -> None:
    """A tap reads the netlist's nets from the testbench: the module's sources are the same,
    and a testbench without taps has none of a tap's text."""
    module = chain_module()
    link = next(
        link for link in module.fragment.links if link.source.instance and link.sink.instance
    )
    end = link.source
    assert end.instance is not None
    tap = Tap(end, 3, link.lane_bits)
    tapped = stream_bench(
        module,
        tmp_path / "tapped",
        inputs=CHAIN_IN,
        outputs=CHAIN_OUT,
        taps={"link_0": tap},
        toolchain=host_toolchain(),
    )
    plain = stream_bench(
        module, tmp_path / "plain", inputs=CHAIN_IN, outputs=CHAIN_OUT, toolchain=host_toolchain()
    )
    data = f"dut.{instance_net(end.instance, end.data)}"
    assert f"link_0_tapped[link_0_taps] <= {data}[{link.lane_bits - 1}:0];" in tapped.text
    assert f"dut.{instance_net(end.instance, end.valid)} && " in tapped.text
    assert '$writememh("link_0.tapped.mem", link_0_tapped);' in tapped.text
    assert "tap" not in plain.text and "dut.n__" not in plain.text
    assert [Path(s).name for s in tapped.sources] == [Path(s).name for s in plain.sources]
    for source in plain.sources:
        relative = Path(source).relative_to(tmp_path / "plain")
        assert (tmp_path / "tapped" / relative).read_bytes() == Path(source).read_bytes()


def test_a_tap_the_module_cannot_present_is_refused(tmp_path: Path) -> None:
    module = chain_module()
    link = next(link for link in module.fragment.links if link.source.instance)
    with pytest.raises(ValueError, match="places no instance nowhere"):
        stream_bench(
            module,
            tmp_path,
            inputs=CHAIN_IN,
            outputs=CHAIN_OUT,
            taps={"link_0": Tap(replace(link.source, instance="nowhere"), 3, 1)},
            toolchain=host_toolchain(),
        )
    with pytest.raises(ValueError, match="identifier no stream has"):
        stream_bench(
            module,
            tmp_path,
            inputs=CHAIN_IN,
            outputs=CHAIN_OUT,
            taps={"m_axis_0": Tap(link.source, 3, 1)},
            toolchain=host_toolchain(),
        )
    assert not any(tmp_path.iterdir())
    root = next(link for link in module.fragment.links if link.source.instance is None)
    with pytest.raises(ValueError, match="a root's port is streamed, not tapped"):
        Tap(root.source, 3, 1)
    with pytest.raises(ValueError, match="bits of a"):
        Tap(link.source, 3, link.source.data_bits + 1)


def test_a_taps_words_end_at_the_first_it_did_not_take(tmp_path: Path) -> None:
    from finn.core.rtlsim_bench import _read_taken  # noqa: PLC0415

    written = tmp_path / "link_0.tapped.mem"
    written.write_text("// memory\n@0\n0a\n1f\nxx\nxx\n")
    assert _read_taken(written) == [0x0A, 0x1F]


@pytest.mark.parametrize("pacing", [FREE, STALLED])
def test_the_watchdog_is_derived_from_the_paced_beats_and_names_each_stream(
    tmp_path: Path, pacing: Pacing
) -> None:
    bench = stream_bench(
        chain_module(),
        tmp_path,
        inputs=CHAIN_IN,
        outputs=CHAIN_OUT,
        pacing=pacing,
        cycles=12,
        toolchain=host_toolchain(),
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
            "result_range": datatype_range(
                6, ScalarEncoding(DataType["INT8"]), ScalarEncoding(DataType["INT8"])
            ),
            **dotp_extents(inputs["w_channel"]),
        },
    )
    found = design_module(point, Int8Dsp58DotpKernel)
    assert isinstance(found, Available)
    module = found.value
    assert module.abi.clock_alignments == (ClockAlignment("ap_clk", "ap_clk2x"),)
    words: dict[str, Words] = {name: ([0], 8) for name in inputs}
    bench = stream_bench(
        module, tmp_path, inputs=words, outputs={"y_channel": ([0], 8)}, toolchain=host_toolchain()
    )
    # Half the clock's period; each rising edge of ap_clk is one of ap_clk2x.
    assert (
        "always #2.5 begin\n        ap_clk2x = !ap_clk2x;\n        if (ap_clk2x) ap_clk = !ap_clk;"
        in bench.text
    )
    assert "always @(posedge ap_clk) #1 begin" in bench.text  # never on an ap_clk2x edge


def _written(initial: tuple[tuple[int, ...], ...], written: tuple[tuple[int, ...], ...]) -> Any:
    return placed(Case("written", initial, written))


def _module(point: Any) -> Any:
    """The written thresholds' module, ``finn_thresholds_written``."""
    return built(point, "thresholds_written")


def _cycles(point: Any) -> int:
    found = cycles_of(point)
    assert isinstance(found, Available), found
    return found.value


def test_a_control_bus_is_written_what_its_kernel_declares(tmp_path: Path) -> None:
    point = _written(((-3, 0, 2),), ((-6, -1, 5),))
    declared = declared_registers(_module(point))
    assert declared == {"s_axilite": point.member("activate").register_map}
    # One 32-bit word a threshold of the shared row (C = 1, N = 3): the word select alone.
    assert declared["s_axilite"] == RegisterMap(((0, 0xD), (4, 0x0), (8, 0x2)))
    stimulus = {"s_axis_0": ([0] * (PIXELS * CHANNELS // 4), 16)}
    expected = {"m_axis_0": ([0] * (PIXELS * CHANNELS // 4), 8)}
    bench = stream_bench(
        _module(point), tmp_path, inputs=stimulus, outputs=expected, toolchain=host_toolchain()
    )
    assert (tmp_path / "s_axilite.writes.mem").read_text().split() == [
        "00000000d",
        "400000000",
        "800000002",
    ]
    assert "wire configured = s_axilite_written == 3;" in bench.text
    other = RegisterMap(((0, 1),))
    stream_bench(
        _module(point),
        tmp_path / "other",
        inputs=stimulus,
        outputs=expected,
        registers={"s_axilite": other},
        toolchain=host_toolchain(),
    )
    assert (tmp_path / "other/s_axilite.writes.mem").read_text().split() == ["000000001"]
    with pytest.raises(
        ValueError, match="no control bus s_axilite_1; its control buses: s_axilite"
    ):
        stream_bench(
            _module(point),
            tmp_path / "unknown",
            inputs=stimulus,
            outputs=expected,
            registers={"s_axilite_1": other},
            toolchain=host_toolchain(),
        )


def _toolchain(tmp_path: Path, xsim_prints: str, fails: str | None = None) -> Toolchain:
    """A toolchain whose simulator tools are scripts in a command directory: each
    records its arguments, xsim prints ``xsim_prints``, and the tool ``fails`` names
    prints an error, writes its log and exits 1. Its Vivado holds ``glbl.v``."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in TOOLS:
        printed = xsim_prints if tool == "xsim" else ""
        failing = (
            f'echo "ERROR: [{tool}] it failed" | tee {tool}.log\nexit 1\n' if tool == fails else ""
        )
        script = tools / tool
        script.write_text(
            f'#!/bin/sh\necho "$0 $*" >> "{tmp_path}/calls"\necho "{printed}"\n{failing}'
        )
        script.chmod(0o755)
    vivado = tmp_path / "vivado"
    (vivado / "data/verilog/src").mkdir(parents=True)
    (vivado / "data/verilog/src/glbl.v").write_text("module glbl(); endmodule\n")
    environment = {"PATH": os.defpath, "XILINX_VIVADO": str(vivado)}
    return Toolchain(Selection(command_dir=str(tools)), environment)


def test_simulate_runs_the_tools_of_its_toolchain(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path, "PASS")
    (tmp_path / "sim").mkdir()
    output = simulate(["a.sv"], "module check; endmodule\n", tmp_path / "sim", toolchain=toolchain)
    assert "PASS" in output
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [Path(call.split()[0]).name for call in calls] == list(TOOLS)
    assert f"{tmp_path}/vivado/data/verilog/src/glbl.v" in calls[0]  # the toolchain's Vivado
    assert available(toolchain)


def test_a_script_replays_the_commands_simulate_runs(tmp_path: Path) -> None:
    """``finn.toolchain.xsim.script`` (``run.sh``'s commands) writes the words the driver
    runs for ``simulate``: ``glbl.v`` from ``$XILINX_VIVADO``, each of xvlog and xelab
    exiting with its own status when it fails, after a guard that exits with its own
    when ``$XILINX_VIVADO`` or a command is missing."""
    toolchain = _toolchain(tmp_path, "PASS")
    directory = tmp_path / "sim"
    directory.mkdir()
    sources = ["a.sv", "inc/b.svh"]
    simulate(sources, "module check; endmodule\n", directory, toolchain=toolchain)
    ran = [call.split()[1:] for call in (tmp_path / "calls").read_text().splitlines()]
    text = xsim.script(
        ["a.sv"],
        "check",
        includes=include_directories(sources),
        testbench=str(directory / "check.sv"),
        snapshot="check",
        threads=xsim.xelab_threads(toolchain.environment),
    )
    lines = text.replace(" \\\n    ", " ").splitlines()
    assert lines[0].startswith('[ -n "${XILINX_VIVADO:-}" ] || ')
    assert lines[1].startswith(f"for tool in {' '.join(TOOLS)}; do command -v ")
    assert all(
        line.endswith(f"exit {xsim.UNAVAILABLE_STATUS}; }}{end}")
        for line, end in (
            (lines[0], ""),
            (lines[1], "; done"),
        )
    )
    glbl = str(tmp_path / "vivado/data/verilog/src/glbl.v")
    written = []
    for line, status in zip(lines[2:], ("10", "11", "1"), strict=True):
        words = shlex.split(line)
        assert words[-3:] == ["||", "exit", status]
        written.append(
            [glbl if word == "$XILINX_VIVADO/data/verilog/src/glbl.v" else word for word in words]
        )
    assert [words[1:-3] for words in written] == ran
    assert xsim.EXIT_STATUS == {"xvlog": 10, "xelab": 11}


def test_simulate_without_pass_fails_as_an_assertion(tmp_path: Path) -> None:
    (tmp_path / "sim").mkdir()
    with pytest.raises(SimulationFailed, match="watchdog") as failed:
        simulate([], "", tmp_path / "sim", toolchain=_toolchain(tmp_path, "watchdog"))
    assert isinstance(failed.value, AssertionError)  # conformance collects these


def test_a_toolchain_naming_no_vivado_has_no_simulator(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path, "PASS")
    bare = Toolchain(toolchain.selection, {"PATH": os.defpath})
    assert not available(bare)
    with pytest.raises(ToolUnavailable, match="XILINX_VIVADO") as missing:
        simulate([], "", tmp_path, toolchain=bare)
    assert missing.value.tool == "vivado"
    assert "host file" in str(missing.value)  # what selects it
    # An installation without its glbl.v is none either.
    (tmp_path / "vivado/data/verilog/src/glbl.v").unlink()
    assert not available(toolchain)
    with pytest.raises(ToolUnavailable, match="glbl.v"):
        simulate([], "", tmp_path, toolchain=toolchain)


@pytest.mark.parametrize("tool", ["xvlog", "xelab"])
def test_a_tool_that_fails_before_the_testbench_runs_is_the_tools_failure(
    tmp_path: Path, tool: str
) -> None:
    """Compilation or elaboration failing is ``ToolFailed`` naming the tool, its exit
    status, its log and its first error line: not a verdict on the hardware."""
    (tmp_path / "sim").mkdir()
    with pytest.raises(ToolFailed) as failed:
        simulate([], "", tmp_path / "sim", toolchain=_toolchain(tmp_path, "PASS", fails=tool))
    assert not isinstance(failed.value, AssertionError)
    assert (failed.value.tool, failed.value.status) == (tool, 1)
    assert failed.value.transcript == tmp_path / "sim" / f"{tool}.log"
    assert failed.value.errors == (f"ERROR: [{tool}] it failed",)


def test_a_testbench_that_ends_in_an_error_is_a_verdict(tmp_path: Path) -> None:
    """xsim exiting with an error is the testbench's ``$fatal``: ``SimulationFailed``."""
    (tmp_path / "sim").mkdir()
    with pytest.raises(SimulationFailed, match="it failed"):
        simulate([], "", tmp_path / "sim", toolchain=_toolchain(tmp_path, "PASS", fails="xsim"))


# -- in XSim ---------------------------------------------------------------------------------


@requires_xsim
def test_a_stopped_stream_fires_the_watchdog_naming_it_and_its_beat(tmp_path: Path) -> None:
    """The Chain given half a row (one of its two input beats a row) produces nothing: the
    watchdog names the output, stopped before its first beat, and not the input, whose
    one word was taken."""
    with pytest.raises(SimulationFailed) as failed:
        stream_through(
            chain_module(),
            tmp_path,
            inputs={"s_axis_0": ([1], 6)},
            outputs=CHAIN_OUT,
            pacing=FREE,
            toolchain=host_toolchain(),
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
    bits = point.member("activate").result_dtype.bitwidth()
    levels = (VALUES[..., None] >= np.array(case.initial[0])).sum(axis=-1)
    words = pack_beats(form, levels.ravel().tolist(), bits)
    wrong = [*words[:1], words[1] ^ 1, *words[2:]]
    with pytest.raises(WordsDiffer) as differ:
        stream_through(
            built(point, stem(case)),
            tmp_path,
            inputs={
                "s_axis_0": (
                    list(pack_beats(form, VALUES.ravel().tolist(), ELEMENT.bitwidth())),
                    ELEMENT.bitwidth() * case.pe,
                )
            },
            outputs={"m_axis_0": (wrong, bits * case.pe)},
            pacing=FREE,
            toolchain=host_toolchain(),
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
    registers = declared_registers(_module(_written(written, written)))
    pe = point.member("activate").pe
    form = vector_major((PIXELS, CHANNELS), pe)
    levels = (VALUES[..., None] >= np.array(written[0])).sum(axis=-1)
    assert not np.array_equal(levels, (VALUES[..., None] >= np.array(initial[0])).sum(axis=-1))
    result_bits = point.member("activate").result_dtype.bitwidth()
    stream_through(
        _module(point),
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
        cycles=_cycles(point),
        toolchain=host_toolchain(),
    )
