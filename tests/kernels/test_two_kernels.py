# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two dot-product layers joined by one stream that plans and adapts between them.

The first layer produces its results PE = 4 lanes a beat, each row once; the
second reads them as activations, SIMD = 2 lanes a beat, each row once per
output fold, framed by reduction. Neither kernel knows the other: each
presents its own traversal of the hidden tensor, derived from its own schedule
over its own folds.
The stream between them plans a width conversion, a replay and the frame, and
its adapter places a ``vpc`` and an ``input_gen``. The composed module computes
``(x @ W1) @ W2`` in XSim, weights stored ``(k, n)``; a stream that admits no
adapter refuses it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, derived, design_space
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import TRAVERSAL, Traversal, period
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.composite import Design
from finn.kernels.configure import commit
from finn.kernels.rom import RomKernel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import exact_result_dtype
from finn.kernels.resources import resource_root, template_root
from finn.kernels.streams import (
    Stream,
)
from finn.kernels.target import DspBlock
from kernels.helpers import settled

ROOT = Path(__file__).resolve().parents[2]
ROWS, INPUTS, HIDDEN, OUTPUTS = 3, 4, 4, 4
PE1, SIMD1, PE2, SIMD2 = 4, 2, 2, 2
A = W = DataType["INT3"]
H = exact_result_dtype(INPUTS, A, W)
Y = exact_result_dtype(HIDDEN, H, W)
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))


def layered(*, adaptable: bool = True):
    class Layered(Design):
        x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
        w1 = Stream(tensor=Tensor((INPUTS, HIDDEN), ScalarEncoding(W)))
        h = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), adaptable=adaptable)
        w2 = Stream(tensor=Tensor((HIDDEN, OUTPUTS), ScalarEncoding(W)))
        y = Stream(tensor=Tensor((ROWS, OUTPUTS), ScalarEncoding(Y)), port="out0_V")
        first = PackedDotpKernel(
            target_dsp=DspBlock.DSP48E2, target_period_ns=5.0, x_stream=x, w_stream=w1, y_stream=h
        )
        second = PackedDotpKernel(
            target_dsp=DspBlock.DSP48E2, target_period_ns=5.0, x_stream=h, w_stream=w2, y_stream=y
        )

        # One pass of each layer's weights, in the order that layer reads them.
        @derived(semantics=TRAVERSAL)
        def first_period(self) -> Traversal:
            return period(self.first.w.sequence.form)

        @derived(semantics=TRAVERSAL)
        def second_period(self) -> Traversal:
            return period(self.second.w.sequence.form)

        rom1 = RomKernel(dtype=W, form=first_period, contents=W1, output_stream=w1)
        rom2 = RomKernel(dtype=W, form=second_period, contents=W2, output_stream=w2)

    point = commit(
        design_space(Layered()),
        {
            "rom1.rom_style": "auto",
            "rom2.rom_style": "auto",
            "first.pe": PE1,
            "first.simd": SIMD1,
            "first.compute_pumping": False,
            "second.pe": PE2,
            "second.simd": SIMD2,
            "second.compute_pumping": False,
        },
    )
    # A stream admitting no adapter keeps its Decision closed; the others settle theirs.
    return settled(point)


def test_the_hidden_stream_plans_width_replay_and_frame_and_places_vpc_and_input_gen():
    point = layered()
    assert point.h.plan.steps == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    assert [stage.name for stage in point.h.connection.stages] == ["vpc", "input_gen"]
    # The first layer's activations need only their frame closed.
    assert point.x.plan.steps == (Step.MARKERS,)
    instances = [item.instance_id for item in point.structure.structure.instances]
    assert {"u_first", "u_second", "u_x_input_gen", "u_h_vpc", "u_h_input_gen"} <= set(instances)
    # The ends belong to the layers' ports; the instances are the layers'.
    connection = point.h.connection
    assert (connection.source_owner, connection.sink_owner) == ("first.y", "second.x")
    vpc = dict(point.h.connection.stages[0].requirements.parameters)
    assert (vpc["PI"], vpc["PO"]) == (PE1, SIMD2)
    generator = dict(point.h.connection.stages[1].requirements.parameters)
    # Per row (two beats of two hidden values), the row once per output fold.
    assert (generator["FM_SIZE"], generator["DIMS"], generator["COEFS"]) == (
        2,
        "'{2, 2}",
        "'{0, 1}",
    )


def test_a_hidden_stream_admitting_no_adapter_refuses_the_pair():
    point = layered(adaptable=False)
    refused = point.h.query(Stream.connection)
    assert isinstance(refused, Rejected)
    plan = [finding for finding in refused.findings if finding.code == "stream-plan"]
    assert plan and "width_conversion -> reorder -> markers" in plan[0].message
    assert isinstance(point.query(type(point).structure), Rejected)


def _pack(values, bits):
    mask = (1 << bits) - 1
    return sum((value & mask) << (index * bits) for index, value in enumerate(values))


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("xvlog", "xelab", "xsim")),
    reason="Vivado simulator tools are unavailable",
)
@pytest.mark.parametrize("stalled", (False, True))
def test_the_two_layers_compute_in_xsim(tmp_path, stalled):
    requirements = layered().structure.requirements
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    sources = [str(Path(materialized.directory) / path) for path in materialized.files]
    hidden = [
        [sum(X[r][k] * W1[k][n] for k in range(INPUTS)) for n in range(HIDDEN)] for r in range(ROWS)
    ]
    y = [
        [sum(hidden[r][k] * W2[k][n] for k in range(HIDDEN)) for n in range(OUTPUTS)]
        for r in range(ROWS)
    ]
    a_bits, y_bits = A.bitwidth(), Y.bitwidth()
    words_in = [
        _pack(X[r][f : f + SIMD1], a_bits) for r in range(ROWS) for f in range(0, INPUTS, SIMD1)
    ]
    words_out = [
        _pack(y[r][f : f + PE2], y_bits) for r in range(ROWS) for f in range(0, OUTPUTS, PE2)
    ]
    in_width, out_width = (SIMD1 * a_bits + 7) // 8 * 8, (PE2 * y_bits + 7) // 8 * 8
    table_in = ", ".join(f"{in_width}'h{word:x}" for word in words_in)
    table_out = ", ".join(f"{PE2 * y_bits}'h{word:x}" for word in words_out)
    valid = "cycle % 3 != 0" if stalled else "1"
    ready = "cycle % 4 != 1" if stalled else "1"
    testbench = tmp_path / "check.sv"
    testbench.write_text(f"""`timescale 1ns/1ps
module check;
    logic ap_clk = 0, ap_rst_n = 0;
    logic [{in_width - 1}:0] in0_V_tdata; logic in0_V_tvalid = 0; wire in0_V_tready;
    wire [{out_width - 1}:0] out0_V_tdata; wire out0_V_tvalid; logic out0_V_tready = 0;
    logic [{in_width - 1}:0] words_in [{len(words_in)}] = '{{{table_in}}};
    logic [{PE2 * y_bits - 1}:0] words_out [{len(words_out)}] = '{{{table_out}}};
    always #5 ap_clk = !ap_clk;
    {prepared.abi.entry_point} dut (.*);
    int sent = 0, received = 0, cycle = 0;
    always @(posedge ap_clk) begin
        cycle <= cycle + 1;
        if (ap_rst_n) begin
            if (in0_V_tvalid && in0_V_tready) sent <= sent + 1;
            if (out0_V_tvalid && out0_V_tready) begin
                if (out0_V_tdata[{PE2 * y_bits - 1}:0] !== words_out[received])
                    $fatal(1, "word %0d: %h != %h", received, out0_V_tdata, words_out[received]);
                received <= received + 1;
            end
        end
    end
    always @(negedge ap_clk) begin
        in0_V_tvalid = ap_rst_n && sent < {len(words_in)} && ({valid});
        in0_V_tdata = words_in[sent < {len(words_in)} ? sent : 0];
        out0_V_tready = {ready};
    end
    initial begin
        repeat (16) @(posedge ap_clk);  // past the DSP models' startup recovery (GSR)
        ap_rst_n = 1;
        wait (received == {len(words_out)});
        repeat (4) @(posedge ap_clk);
        $display("TWO_LAYERS_PASS");
        $finish;
    end
    initial begin #100000; $fatal(1, "watchdog"); end
endmodule
""")
    vivado = Path(os.environ.get("XILINX_VIVADO", str(Path(str(shutil.which("xelab"))).parents[1])))
    commands = (
        ["xvlog", "--sv", *sources, str(vivado / "data/verilog/src/glbl.v"), str(testbench)],
        [
            "xelab",
            "work.check",
            "work.glbl",
            "--mt",
            "2",
            "-L",
            "unisims_ver",
            "--snapshot",
            "check",
            "--timescale",
            "1ns/1ps",
        ],
        ["xsim", "check", "--runall"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "TWO_LAYERS_PASS" in result.stdout, result.stdout + result.stderr
