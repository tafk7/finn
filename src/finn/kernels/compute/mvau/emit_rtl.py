############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The wrapper template is finn-rtllib/mvu/mvu_vvu_axi_wrapper.v verbatim; the
# bindings reproduce prepare_codegen_default (matrixvectoractivation_rtl.py:328)
# as a pure function of (point, context) with no self.onnx_node / model access.
############################################################################

"""Hermetic RTL codegen for the MVAU compute core (internal_embedded mode).

Both DSP bundles (softvec + packed) share THIS emit — they fill the SAME AXI-lite
wrapper template (``mvu_vvu_axi_wrapper.v``) and differ in (a) which per-core wrapper
the top instantiates (the selected backend's ``rtl_core_module`` field ->
``$MODULE_NAME_COMPUTE_CORE$``) and (b) which compute core ``.sv`` + per-core wrapper is
in ``point.sources``. Post-2c-split
the two source sets are genuinely disjoint on the core/wrapper (they share only the
base ``.svh`` plumbing) — the fused ``mvu_vvu_axi.sv`` with its internal
genINT8/genSoftVec ``generate`` fork is retired from our emit path. The MVU core reads
weights over the AXI stream ``in1_V`` — there is no weight ROM — so embedded mode
needs NO weight data file: the artifacts are just the generated wrapper ``.v`` plus
the static ``.sv``/``.svh`` list read straight off ``point.sources``.

SCOPE — compute half only. This is the faithful *wrapper* artifact for the compute
core. Baseline FINN forces HLS for embedded RTL-MVAU (the RTL core has no embedded
weight ROM; RTL is exercised via the decoupled/memstream path). End-to-end rtlsim of
RTL-MVAU needs the decoupled weight-delivery path — the deferred composition arc — not
this embedded wrapper alone.
"""

from __future__ import annotations

from finn.kernels.model.artifacts import Artifacts, BitWidth, Bool, Dim, GeneratedFile, Raw, RtlModule, StaticFile, Template, bind
from finn.kernels.model.ports import Direction, Kind, Port, Role

from .geometry import mvau_geometry
from .op import INPUT, WEIGHTS

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

$MODULE_NAME_COMPUTE_CORE$ #(
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

# The TYPE signature of _V_WRAPPER (F4) — one type per $SLOT$. Owned by the artifact
# (this file), not the Backend: softvec + packed share this one wrapper, so one schema
# (design pitch §5). bind() type-checks values against it before render, catching the
# `[31:0]`-into-a-width-slot silent-wrongness class. Verilog-expression parameters
# (WEIGHT_STREAM_WIDTH_BA etc.) are deduced in-template and are NOT slots.
_V_WRAPPER_SCHEMA = RtlModule(
    "mvu_vvu_axi_wrapper",
    {
        "MODULE_NAME_AXI_WRAPPER": Raw,
        # the per-core wrapper the top instantiates — data, not silicon (2c split):
        # mvu_vvu_axi_softvec / mvu_vvu_axi_packed, from the selected backend's
        # rtl_core_module field.
        "MODULE_NAME_COMPUTE_CORE": Raw,
        "IS_MVU": Dim,
        "VERSION": Dim,
        "PUMPED_COMPUTE": Bool,
        "MW": Dim,
        "MH": Dim,
        "PE": Dim,
        "SIMD": Dim,
        "ACTIVATION_WIDTH": BitWidth,
        "WEIGHT_WIDTH": BitWidth,
        "ACCU_WIDTH": BitWidth,
        "NARROW_WEIGHTS": Bool,
        "SIGNED_ACTIVATIONS": Bool,
        "SEGMENTLEN": Dim,
    },
)


def emit_mvau_rtl(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Produce embedded-mode RTL MVAU compute-core artifacts from a resolved point."""
    from .op import mvau_kernel

    idt = context.tensor_datatype(INPUT)
    wdt = context.tensor_datatype(WEIGHTS)
    geo = mvau_geometry(point, context)
    # rtl_core_module is a STATIC Backend identity field (F5) — read off the selected
    # bundle, not re-projected onto the point.
    rtl_core_module = mvau_kernel().selected_backend(point).rtl_core_module

    # Padded weight-stream width: (PE*SIMD*WEIGHT_WIDTH + 7)//8 * 8 — matches the
    # wrapper's WEIGHT_STREAM_WIDTH_BA and the memstream m_axis_0 width the stitch
    # binds against.
    weight_width = (point.PE * point.SIMD * wdt.bitwidth() + 7) // 8 * 8

    # Typed bind against the wrapper's RtlModule schema (F4). ACCU_WIDTH from the
    # resolved accDataType (kernel-model.md §A1). bind() checks types + slot coverage
    # and returns the {slot: str} dict Template.render consumes — bytes unchanged.
    bindings = bind(
        _V_WRAPPER_SCHEMA,
        {
            "MODULE_NAME_AXI_WRAPPER": Raw(module_name),
            "MODULE_NAME_COMPUTE_CORE": Raw(rtl_core_module),
            "IS_MVU": Dim(1),
            "VERSION": Dim(point.dsp_version),  # forced from device (E1/E2/DSP58 -> 1/2/3)
            "PUMPED_COMPUTE": Bool(point.get("pumpedCompute", 0)),
            "MW": Dim(geo.MW),
            "MH": Dim(geo.MH),
            "PE": Dim(point.PE),
            "SIMD": Dim(point.SIMD),
            "ACTIVATION_WIDTH": BitWidth(idt.bitwidth()),
            "WEIGHT_WIDTH": BitWidth(wdt.bitwidth()),
            "ACCU_WIDTH": BitWidth(point.accDataType.bitwidth()),
            "NARROW_WEIGHTS": Bool(point.narrow_weights),
            "SIGNED_ACTIVATIONS": Bool(idt.min() < 0),
            "SEGMENTLEN": Dim(point.SEGMENTLEN),
        },
    )
    top = GeneratedFile(f"{module_name}.v", _V_WRAPPER, bindings)

    # Static .sv/.svh read straight off point.sources — softvec ships
    # mvu_vvu_axi_softvec.sv + mvu.sv, packed ships mvu_vvu_axi_packed.sv +
    # mvu_vvu_8sx9_dsp58.sv; both share only the base .svh plumbing. Post-2c-split the
    # per-core source sets are disjoint on the core/wrapper.
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
             width=point["stream_width.inp"], boundary=True),
        Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0,
             width=point["stream_width.out"], boundary=True),
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

    # No per-emit IPICommands: emit_composed builds the authoritative IPI from stitch
    # (compose_emit.py) — the per-emit instantiation line + the ap_clk2x self-tie were
    # dead output. The structural CLOCK-role ap_clk2x port + stitch's clock broadcast
    # already carry the non-pumped tie, so stitch is the sole IPI source (F5/F6).
    return Artifacts(
        generated=(top,),
        data_files=(),  # embedded needs no weight file; weights arrive via in1_V stream
        static_files=static,
        ports=ports,
    )
