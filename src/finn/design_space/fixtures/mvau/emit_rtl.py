############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# The wrapper template is finn-rtllib/mvu/mvu_vvu_axi_wrapper.v verbatim; the
# bindings reproduce prepare_codegen_default (matrixvectoractivation_rtl.py:328)
# as a pure function of (point, context) with no self.onnx_node / model access.
############################################################################

"""Hermetic RTL codegen for the MVAU compute core (internal_embedded mode).

Both DSP bundles (softvec + packed) share THIS emit — they fill an IDENTICAL AXI-lite
wrapper (``mvu_vvu_axi_wrapper.v``) and differ ONLY in which compute core ``.sv`` is
in ``point.sources`` (the shared-wrapper 2c non-separation). The MVU core reads
weights over the AXI stream ``in1_V`` — there is no weight ROM — so embedded mode
needs NO weight data file: the artifacts are just the generated wrapper ``.v`` plus
the static ``.sv`` list read straight off ``point.sources``.

SCOPE — compute half only. This is the faithful *wrapper* artifact for the compute
core. Baseline FINN forces HLS for embedded RTL-MVAU (the RTL core has no embedded
weight ROM; RTL is exercised via the decoupled/memstream path). End-to-end rtlsim of
RTL-MVAU needs the decoupled weight-delivery path — the deferred composition arc — not
this embedded wrapper alone.
"""

from __future__ import annotations

from finn.design_space.space import (
    Artifacts,
    Direction,
    GeneratedFile,
    IPICommands,
    Kind,
    Port,
    Role,
    StaticFile,
    Template,
)

from .names import INPUT, WEIGHTS

# finn-rtllib/mvu/mvu_vvu_axi_wrapper.v — 14 $UPPERCASE$ slots (all compute-core
# config; none touch the weight-delivery cluster). Verilog expressions like
# (PE*SIMD*WEIGHT_WIDTH+7)/8 are not $NAME$ tokens and are left intact.
_V_WRAPPER = Template(
    """\
module $MODULE_NAME_AXI_WRAPPER$ #(
	parameter	IS_MVU = $IS_MVU$,
	parameter	VERSION = $VERSION$,	// Allowed versions - 1: DSP48E1, 2: DSP48E2, 3: DSP58
	parameter	PUMPED_COMPUTE = $PUMPED_COMPUTE$,
	parameter	MW = $MW$,
	parameter	MH = $MH$,
	parameter	PE = $PE$,
	parameter	SIMD = $SIMD$,
	parameter	ACTIVATION_WIDTH = $ACTIVATION_WIDTH$,
	parameter	WEIGHT_WIDTH = $WEIGHT_WIDTH$,
	parameter	ACCU_WIDTH = $ACCU_WIDTH$,
	parameter	NARROW_WEIGHTS = $NARROW_WEIGHTS$,
	parameter	SIGNED_ACTIVATIONS = $SIGNED_ACTIVATIONS$,
	parameter	SEGMENTLEN = $SEGMENTLEN$,

	// Safely deducible parameters
	parameter	WEIGHT_STREAM_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8 * 8,
	parameter 	INPUT_STREAM_WIDTH_BA = ((IS_MVU == 1 ? 1 : PE) * SIMD * ACTIVATION_WIDTH + 7) / 8 * 8,
	parameter 	OUTPUT_STREAM_WIDTH_BA = (PE*ACCU_WIDTH + 7)/8 * 8
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in1_V:in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk2x CLK" *)
	input   ap_clk2x,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// Weight Stream
	input	[WEIGHT_STREAM_WIDTH_BA-1:0]  in1_V_TDATA,
	input   in1_V_TVALID,
	output  in1_V_TREADY,
	// Input Stream
	input	[INPUT_STREAM_WIDTH_BA-1:0]  in0_V_TDATA,
	input	in0_V_TVALID,
	output	in0_V_TREADY,
	// Output Stream
	output	[OUTPUT_STREAM_WIDTH_BA-1:0]  out0_V_TDATA,
	output	out0_V_TVALID,
	input	out0_V_TREADY
);

mvu_vvu_axi #(
`ifdef FINN_SIMULATION
	.FORCE_BEHAVIORAL(1),
`endif
	.IS_MVU(IS_MVU), .VERSION(VERSION), .PUMPED_COMPUTE(PUMPED_COMPUTE), .MW(MW), .MH(MH), .PE(PE), .SIMD(SIMD),
	.ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACCU_WIDTH(ACCU_WIDTH), .NARROW_WEIGHTS(NARROW_WEIGHTS),
	.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN)
	) inst (
	.ap_clk(ap_clk),
	.ap_clk2x(ap_clk2x),
	.ap_rst_n(ap_rst_n),
	.s_axis_weights_tdata(in1_V_TDATA),
	.s_axis_weights_tvalid(in1_V_TVALID),
	.s_axis_weights_tready(in1_V_TREADY),
	.s_axis_input_tdata(in0_V_TDATA),
	.s_axis_input_tvalid(in0_V_TVALID),
	.s_axis_input_tready(in0_V_TREADY),
	.m_axis_output_tdata(out0_V_TDATA),
	.m_axis_output_tvalid(out0_V_TVALID),
	.m_axis_output_tready(out0_V_TREADY)
);

endmodule // $MODULE_NAME_AXI_WRAPPER$
"""
)


def emit_mvau_rtl(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Produce embedded-mode RTL MVAU compute-core artifacts from a resolved point."""
    idt = context.tensor_datatype(INPUT)
    wdt = context.tensor_datatype(WEIGHTS)

    # Padded weight-stream width: (PE*SIMD*WEIGHT_WIDTH + 7)//8 * 8 — matches the
    # wrapper's WEIGHT_STREAM_WIDTH_BA and the memstream m_axis_0 width the stitch
    # binds against.
    weight_width = (point.PE * point.SIMD * wdt.bitwidth() + 7) // 8 * 8

    bindings = {
        "MODULE_NAME_AXI_WRAPPER": module_name,
        "IS_MVU": 1,
        "VERSION": point.dsp_version,  # forced from device (DSP48E1/E2/DSP58 -> 1/2/3)
        "PUMPED_COMPUTE": int(point.get("pumpedCompute", 0)),
        "MW": point.MW,
        "MH": point.MH,
        "PE": point.PE,
        "SIMD": point.SIMD,
        "ACTIVATION_WIDTH": idt.bitwidth(),
        "WEIGHT_WIDTH": wdt.bitwidth(),
        "ACCU_WIDTH": point.accDataType.bitwidth(),
        "NARROW_WEIGHTS": int(point.narrow_weights),
        "SIGNED_ACTIVATIONS": 1 if idt.min() < 0 else 0,
        "SEGMENTLEN": point.SEGMENTLEN,
    }
    top = GeneratedFile(f"{module_name}.v", _V_WRAPPER, bindings)

    # Static .sv are read straight off point.sources — softvec ships mvu.sv, packed
    # ships mvu_vvu_8sx9_dsp58.sv, both share the wrapper/plumbing files. This is the
    # documented 2c overlap made explicit at the artifact boundary.
    static = tuple(
        StaticFile("finn.data", f"finn-rtllib/mvu/{s}") for s in point.sources
    )

    # The compute core exposes: activation in (in0_V), activation out (out0_V), a
    # weight-stream input (in1_V), clock, reset. in0_V/out0_V are dataflow-graph edges
    # (boundary — they export to the enclosing region). in1_V is a WEIGHT_SINK: the
    # stitch binds it to a delivery cell's WEIGHT_SOURCE. When there is no delivery
    # cell (embedded compilation), the stitch finds no complementary source and in1_V
    # exports up as a boundary weight port — the topology-driven role behaviour, decided
    # by the resolver at compose time, not hardcoded here.
    ports = (
        Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V", index=0,
             width=point.instream_width, boundary=True),
        Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0,
             width=point.outstream_width, boundary=True),
        Port(Direction.IN, Kind.AXIS, Role.WEIGHT_SINK, "in1_V", index=0,
             width=weight_width),
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk"),
        # ap_clk2x is a real port on the wrapper. For a non-pumped compute it must be
        # driven by the same clock as ap_clk (FINN ties them, emit_rtl's own IPI did the
        # self-tie); as a CLOCK-role port it fans out from the region ap_clk port in the
        # stitch broadcast — the non-pumped tie, done structurally. (A pumped design
        # would bind it to a distinct region 2x-clock port — a documented extension.)
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk2x"),
        Port(Direction.IN, Kind.RESET, Role.RESET, "ap_rst_n"),
    )

    return Artifacts(
        generated=(top,),
        data_files=(),  # embedded needs no weight file; weights arrive via in1_V stream
        static_files=static,
        ports=ports,
        ipi=IPICommands(
            (
                f"create_bd_cell -type hier -reference {module_name} {module_name}",
                # non-pumped: tie the core's ap_clk2x to ap_clk.
                f"connect_bd_net [get_bd_pins {module_name}/ap_clk] "
                f"[get_bd_pins {module_name}/ap_clk2x]",
            )
        ),
    )
