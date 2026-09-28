# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Non-stream interfaces: exported control buses, tie-offs and child padding.

dotp feeds thresholding inside one composite; the activation stream's adapter
replays each row for dotp. dotp's padded AXIS result feeds
a child: the padding bits stay unconnected and the consumer's padding is
zero. Thresholding's AXI-Lite bus is exported through a ``ControlBus`` when
its thresholds are runtime-writable and otherwise held idle by its tie-offs,
as is the set selector of a single threshold set.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Members, Rejected, Space, design_space, view
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, StandardProtocol
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.control import EXPORTED, ControlBus
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import matmul_schedule
from finn.kernels.physical.structure import ConstantBits, PinSlice
from finn.kernels.physical.validation import abi_pins
from finn.kernels.resources import resource_root, template_root
from finn.kernels.streams import (
    CONNECTION,
    COMPOSED,
    MODULE,
    TIEOFFS,
    Composed,
    Stream,
    commit_adapters,
    netlist,
)
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel

ROOT = Path(__file__).resolve().parents[2]
REPETITIONS, WIDTH, HEIGHT, SIMD = 2, 4, 2, 2
FOLDS = WIDTH // SIMD
A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT9"]
THRESHOLDS = (((-5, 0, 7), (-2, 3, 10)),)
SCHEDULE = matmul_schedule(rows=REPETITIONS, reduction=WIDTH, outputs=HEIGHT, pe=1, simd=SIMD)
X = Tensor((REPETITIONS, WIDTH), ScalarEncoding(A))
WEIGHT_TENSOR = Tensor((WIDTH, HEIGHT), ScalarEncoding(W))
RESULT_TENSOR = Tensor((REPETITIONS, HEIGHT), ScalarEncoding(R))
LEVEL_TENSOR = Tensor((REPETITIONS, HEIGHT), ScalarEncoding(DataType["UINT2"]))


class Activated(Space):
    """dotp, then thresholding: a padded child result feeding a child."""

    activations = Stream(tensor=X, port="in0_V")
    weights = Stream(tensor=WEIGHT_TENSOR, port="in1_V")
    results = Stream(tensor=RESULT_TENSOR)
    levels = Stream(tensor=LEVEL_TENSOR, port="out0_V")
    config = ControlBus(port="s_axilite")
    compute = PackedDotpKernel(
        activation_dtype=A,
        weights_dtype=W,
        result_dtype=R,
        pe=1,
        simd=SIMD,
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
        activation_stream=activations,
        weights_stream=weights,
        result_stream=results,
        schedule=SCHEDULE,
    )
    activate = ThresholdingAxiKernel(
        input_dtype=R,
        threshold_dtype=R,
        thresholds=THRESHOLDS,
        bias=0,
        pe=1,
        depth_trigger_bram=0,
        depth_trigger_uram=0,
        input_stream=results,
        output_stream=levels,
        control=config,
    )
    modules = Members(MODULE)
    streams = Members(CONNECTION)
    tieoffs = Members(TIEOFFS)
    controls = Members(EXPORTED)

    @view(semantics=COMPOSED, requires=(modules, streams, tieoffs, controls))
    def structure(self) -> Composed | Rejected:
        return netlist(
            self.modules,
            self.streams,
            self.tieoffs,
            self.controls,
            module="activated",
            producer=ProducerIdentity("test.activated", "1"),
        )


def activated(*, writable: bool):
    return commit_adapters(
        design_space(Activated()).with_choices(
            {
                Activated.compute.compute_pumping: False,
                Activated.activate.use_axilite: writable,
                Activated.activate.deep_pipeline: False,
            }
        )
    )


def test_a_padded_child_result_feeds_a_child_and_its_padding_stays_unconnected():
    # Probe P3: one INT9 lane rides a 16-bit AXIS word into thresholding.
    structure = activated(writable=False).structure.structure
    (padding,) = [item for item in structure.unused_outputs if item.pin.instance_id == "u_compute"]
    assert (padding.pin.signal_id, padding.offset, padding.width) == ("m_axis_output_tdata", 9, 7)
    fed = {
        (wire.destination.bit_offset, wire.destination.bit_width): wire.source
        for wire in structure.wires
        if wire.destination.pin.instance_id == "u_activate"
        and wire.destination.pin.signal_id == "s_axis_tdata"
    }
    assert fed[(9, 7)] == ConstantBits(7, 0)
    assert isinstance(fed[(0, 9)], PinSlice)


def test_read_only_thresholds_tie_their_control_and_set_interfaces():
    # Probe P4: every thresholding input is driven, and nothing is exported.
    structure = activated(writable=False).structure.structure
    assert [port.name for port in structure.top_abi.ports] == [
        "ap_clk",
        "ap_rst_n",
        "in0_V",
        "in1_V",
        "out0_V",
    ]
    tied = {
        wire.destination.pin.signal_id: wire.source.value
        for wire in structure.wires
        if wire.destination.pin.instance_id == "u_activate"
        and isinstance(wire.source, ConstantBits)
    }
    assert tied["s_axilite_AWVALID"] == tied["s_axis_set_tvalid"] == 0
    unused = {
        item.pin.signal_id
        for item in structure.unused_outputs
        if item.pin.instance_id == "u_activate"
    }
    assert {"s_axilite_AWREADY", "s_axilite_RDATA", "s_axis_set_tready"} <= unused


def test_writable_thresholds_export_their_bus_through_the_control_node():
    structure = activated(writable=True).structure.structure
    (bus,) = [
        port
        for port in structure.top_abi.ports
        if isinstance(port, Bus) and port.name == "s_axilite"
    ]
    assert bus.protocol is StandardProtocol.AXILITE and bus.endpoint is Endpoint.TARGET
    assert (bus.associated_clock, bus.associated_reset) == ("ap_clk", "ap_rst_n")
    wired = {
        wire.destination.pin.signal_id: wire.source.pin.signal_id
        for wire in structure.wires
        if wire.destination.pin.instance_id == "u_activate"
        and isinstance(wire.source, PinSlice)
        and wire.source.pin.instance_id is None
    }
    assert wired["s_axilite_AWVALID"] == "s_axilite_AWVALID"


def test_writable_thresholds_without_a_control_bus_are_refused():
    class Unexported(Activated):
        activate = ThresholdingAxiKernel(
            input_dtype=R,
            threshold_dtype=R,
            thresholds=THRESHOLDS,
            bias=0,
            pe=1,
            depth_trigger_bram=0,
            depth_trigger_uram=0,
            input_stream=Activated.results,
            output_stream=Activated.levels,
        )

    point = design_space(Unexported()).with_choices(
        {
            Unexported.compute.compute_pumping: False,
            Unexported.activate.use_axilite: True,
            Unexported.activate.deep_pipeline: False,
        }
    )
    refused = point.activate.query(ThresholdingAxiKernel.tieoffs)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"threshold-control"}


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("xvlog", "xelab", "xsim")),
    reason="Vivado simulator tools are unavailable",
)
@pytest.mark.parametrize("writable", (False, True))
def test_the_composed_module_computes_thresholded_dot_products(tmp_path, writable):
    # Writable, the exported AXI-Lite bus is held idle by the testbench: the
    # thresholds are still the initial table.
    requirements = activated(writable=writable).structure.requirements
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    sources = [str(Path(materialized.directory) / path) for path in materialized.files]
    x = [[(3 * r + 5 * k) % 8 - 4 for k in range(WIDTH)] for r in range(REPETITIONS)]
    w = [[(7 * h + 3 * k) % 8 - 4 for k in range(WIDTH)] for h in range(HEIGHT)]
    levels = [
        sum(t <= sum(x[r][k] * w[h][k] for k in range(WIDTH)) for t in THRESHOLDS[0][h])
        for r in range(REPETITIONS)
        for h in range(HEIGHT)
    ]

    def word(values):
        return sum((v & 7) << (3 * i) for i, v in enumerate(values))

    # Each row once: the stream's input_gen presents it HEIGHT times, framed.
    activation_words = [
        word(x[r][f * SIMD : (f + 1) * SIMD]) for r in range(REPETITIONS) for f in range(FOLDS)
    ]
    weight_words = [
        word(w[h][f * SIMD : (f + 1) * SIMD])
        for _ in range(REPETITIONS)
        for h in range(HEIGHT)
        for f in range(FOLDS)
    ]
    count = len(weight_words)

    def table(name, bits, words):
        values = ", ".join(f"{bits}'h{value:x}" for value in words)
        return f"logic [{bits - 1}:0] {name} [{len(words)}] = '{{{values}}};"

    control = " ".join(
        f"logic [{info.width - 1}:0] {name} = 0;"
        if info.direction is Direction.IN
        else f"wire [{info.width - 1}:0] {name};"
        for name, info in abi_pins(requirements.abi).items()
        if name.startswith("s_axilite_")
    )
    testbench = tmp_path / "check.sv"
    testbench.write_text(f"""`timescale 1ns/1ps
module check;
    logic ap_clk = 0, ap_rst_n = 0;
    logic [7:0] in0_V_tdata = 0; logic in0_V_tvalid = 0; wire in0_V_tready;
    logic [7:0] in1_V_tdata = 0; logic in1_V_tvalid = 0; wire in1_V_tready;
    wire [7:0] out0_V_tdata; wire out0_V_tvalid; logic out0_V_tready = 1;
    {control}
    {table("activations", 8, activation_words)}
    {table("weights", 8, weight_words)}
    {table("expected", 2, levels)}
    always #5 ap_clk = !ap_clk;
    {prepared.abi.entry_point} dut (.*);
    int a = 0, b = 0, received = 0;
    always @(posedge ap_clk) if (ap_rst_n) begin
        if (in0_V_tvalid && in0_V_tready) a <= a + 1;
        if (in1_V_tvalid && in1_V_tready) b <= b + 1;
        if (out0_V_tvalid && out0_V_tready) begin
            if (out0_V_tdata[1:0] !== expected[received])
                $fatal(1, "level %0d: %0d != %0d", received, out0_V_tdata[1:0], expected[received]);
            received <= received + 1;
        end
    end
    always @* begin
        in0_V_tvalid = ap_rst_n && a < {len(activation_words)};
        in0_V_tdata = a < {len(activation_words)} ? activations[a] : 0;
        in1_V_tvalid = ap_rst_n && b < {count};
        in1_V_tdata = b < {count} ? weights[b] : 0;
    end
    initial begin
        repeat (16) @(posedge ap_clk);  // past the DSP models' startup recovery (GSR)
        ap_rst_n <= 1;
        wait (received == {len(levels)});
        repeat (4) @(posedge ap_clk);
        $display("ACTIVATED_PASS");
        $finish;
    end
    initial begin #20000; $fatal(1, "watchdog"); end
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
    assert "ACTIVATED_PASS" in result.stdout, result.stdout + result.stderr


def test_several_threshold_sets_take_a_set_selector_stream():
    # A sideband is an ordinary stream: one set index per input beat.
    selectors = Tensor((REPETITIONS * HEIGHT,), ScalarEncoding(DataType["UINT1"]))
    two_sets = (THRESHOLDS[0], ((-4, 1, 8), (-3, 2, 9)))

    class Selected(Space):
        values = Stream(tensor=RESULT_TENSOR, port="in0_V")
        sets = Stream(tensor=selectors, port="in1_V")
        levels = Stream(tensor=LEVEL_TENSOR, port="out0_V")
        activate = ThresholdingAxiKernel(
            input_dtype=R,
            threshold_dtype=R,
            thresholds=two_sets,
            bias=0,
            pe=1,
            depth_trigger_bram=0,
            depth_trigger_uram=0,
            input_stream=values,
            output_stream=levels,
            set_stream=sets,
        )

    point = design_space(Selected()).with_choices(
        {Selected.activate.use_axilite: False, Selected.activate.deep_pipeline: False}
    )
    connection = point.sets.query(Stream.connection)
    assert connection.value.sink.transport.name == "s_axis_set"  # type: ignore[union-attr]
    # One set index for every input beat: a shorter selector stream is refused.
    short = Tensor((HEIGHT,), ScalarEncoding(DataType["UINT1"]))

    class Short(Selected):
        sets = Stream(tensor=short, port="in1_V")

    refused = design_space(Short()).sets.query(Stream.connection)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"threshold-set-stream"}
