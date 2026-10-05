# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

from finn.core.space import Available, Rejected, design_space
from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.transport import MarkerKind, ReadyValidStream, StreamMarker
from kernels.test_migrated_rich import generator
from kernels.test_migrated_simple import eltwise
from kernels.helpers import finnlib_root, FULL_DSP48E2
from kernels.xsim import requires_xsim, simulate


def test_native_streams_are_inspectable_without_storage_choices():
    base = design_space(FifoKernel(word_bits=13, depth=8, platform=FULL_DSP48E2))
    source, sink = base.input.transport, base.output.transport
    assert source.data_width == sink.data_width == 13
    assert [pin.direction for pin in source.pins()] == [Direction.IN, Direction.IN, Direction.OUT]
    assert [pin.direction for pin in sink.pins()] == [Direction.OUT, Direction.OUT, Direction.IN]
    with pytest.raises(ValueError, match="byte aligned"):
        source.axis_bus()


def test_loop_markers_remain_native_even_when_one_bit():
    for dims, strides in (((6,), (1,)), ((3, 6), (0, 1))):
        output = generator(bits=16, dims=dims, strides=strides).output.transport
        assert output.markers == (StreamMarker("olst", MarkerKind.LOOP_END, len(dims)),)
        with pytest.raises(ValueError, match="single LAST"):
            output.axis_bus()


def test_axi_lowering_preserves_explicit_physical_mapping():
    stream = ReadyValidStream(
        "s_axis",
        16,
        Endpoint.TARGET,
        "word",
        "valid",
        "ready",
        "clk",
        "rst",
        (StreamMarker("end", MarkerKind.LAST),),
    )
    bus = stream.axis_bus()
    assert dict(bus.widths()) == {"word": 16, "valid": 1, "ready": 1, "end": 1}
    assert bus.associated_clock == "clk" and bus.associated_reset == "rst"


@requires_xsim
def test_fifo_capacity_and_effective_storage_agree_with_native_rtl(tmp_path):
    cases = (
        (2, "ultra"),
        (33, "block"),
        (64, "auto"),
        (65, "auto"),
        (257, "auto"),
        (40, "distributed"),
        (65, "ultra"),
        (650, "block"),
        (2029, "auto"),
        (4200, "ultra"),
    )
    instances = []
    for index, (depth, style) in enumerate(cases):
        storage = (
            design_space(FifoKernel(word_bits=9, depth=depth, platform=FULL_DSP48E2))
            .with_choices(ram_style=style)
            .storage
        )
        instances.append(
            f'fifo_capacity_case #(.DEPTH({depth}), .STYLE("{style}"), '
            f'.CAPACITY({storage.capacity}), .EFFECTIVE("{storage.effective_style}")) '
            f"c{index}(done[{index}]);"
        )
    simulate(
        [finnlib_root() / "rtl/infra/fifo.sv"],
        """
module fifo_capacity_case #(
    parameter int DEPTH=2, CAPACITY=5,
    parameter STYLE="auto", EFFECTIVE="shift"
)(output logic done=0);
    logic clk=0, rst=1, ivld=0;
    logic [8:0] idat=0;
    wire [8:0] odat;
    wire irdy, ovld;
    int accepted=0;
    always #5 clk=!clk;
    fifo #(.DEPTH(DEPTH), .DATA_WIDTH(9), .RAM_STYLE(STYLE)) dut(
        .clk,.rst,.idat,.ivld,.irdy,.odat,.ovld,.ordy(1'b0));
    always_ff @(posedge clk) if(!rst && ivld && irdy) accepted <= accepted+1;
    initial begin
        repeat(3) @(negedge clk);
        rst=0; ivld=1;
        repeat(CAPACITY+30) begin
            @(negedge clk);
            idat=accepted[8:0];
        end
        if(accepted != CAPACITY || irdy || dut.RAM_STYLE_EFF != EFFECTIVE)
            $fatal(1,"FIFO depth=%0d style=%s: observed capacity=%0d, expected=%0d",
                DEPTH,STYLE,accepted,CAPACITY);
        done=1;
    end
endmodule
module check;
"""
        + f"    wire [{len(cases) - 1}:0] done;\n"
        + "\n".join(instances)
        + """
    initial begin
        wait(&done);
        $display("FIFO_CAPACITY_PASS");
        $finish;
    end
    initial begin #100000; $fatal(1,"FIFO capacity watchdog"); end
endmodule
""",
        tmp_path,
    )


def test_eltwise_ports_carry_their_operands_unpadded_on_native_pins():
    mixed = eltwise(lhs="INT5", rhs="FLOAT32", pe=3)
    lhs, rhs, result = mixed.lhs.transport, mixed.rhs.transport, mixed.result.transport
    assert (lhs.data, lhs.valid, lhs.ready) == ("adat", "avld", "ardy")
    assert (lhs.data_width, rhs.data_width, result.data_width) == (15, 96, 96)
    # An operand the arithmetic does not take refuses the kernel, naming the operand.
    refused = eltwise(lhs="FLOAT16", rhs="FLOAT16")
    assert refused.lhs.transport.data_width == 32  # the pins do not wait for admission
    answer = refused.query(EltwiseKernel.module)
    assert isinstance(answer, Rejected)
    assert "operands_supported" in {finding.owner for finding in answer.findings}
    assert isinstance(eltwise().query(EltwiseKernel.module), Available)
