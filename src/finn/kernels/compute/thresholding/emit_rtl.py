############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The .dat binary-search packing and the narrow-quant bias/threshold adjustment
# are faithful in BEHAVIOUR to FINN's thresholding_rtl.py
# (prepare_codegen_rtl_values / make_weight_file), rewritten as pure functions of
# (point, context) with no self.onnx_node / model access.
############################################################################

"""Hermetic RTL codegen for Thresholding (embedded ``.dat`` memory-init mode).

FINN's RTL thresholding is inherently *embedded*: thresholds are baked into
per-(PE, stage) ``.dat`` files read by ``$readmemh`` in the core. This emit reads the
resolved ``point`` (folding + knobs) and the frozen ``context`` (dtypes + threshold
VALUES via ``initializer``) — never the graph — and returns typed :class:`Artifacts`.
The generated top is the real finn-rtllib ``thresholding_axi`` wrapper, filled through
the typed :class:`Template`: ``$WI$``/``$WT$``/``$C$``… slots are validated while
Verilog ``$clog2(...)`` expressions are left untouched (they are not ``$UPPERCASE$``).

Decoupled/AXI-lite runtime-writable streaming is out of scope (embedded first cut).
"""

from __future__ import annotations

import math

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.kernels.emit.artifacts import Artifacts, DataFile, GeneratedFile, IPICommands, Template
from finn.kernels.emit.manifest import ArtifactManifest, SourceFile
from finn.util.data_packing import pack_innermost_dim_as_hex_string

from .names import INPUT, OUTPUT, THRESHOLDS

# The single source-of-truth for this backend's build sources (F9). Each file carries its
# own in-tree resolved path — the two finn-rtllib subdirs (thresholding/hdl, axi/hdl) that
# were previously scattered across inline StaticFile f-strings. Backend.sources reads
# ``.filenames``; the emit reads ``.static_files()`` — one list, two readers.
RTL_MANIFEST = ArtifactManifest(
    sources=(
        SourceFile("thresholding.sv", root="finn-rtllib/thresholding/hdl"),
        SourceFile("thresholding_axi.sv", root="finn-rtllib/thresholding/hdl"),
        SourceFile("axilite.sv", root="finn-rtllib/axi/hdl"),
    ),
)

# The real finn-rtllib IP-packaging wrapper, as a typed template. Slots are
# $UPPERCASE$; Verilog $clog2(...) is deliberately not a slot.
_V_WRAPPER = Template(
    """\
module $MODULE$ #(
	parameter  WI = $WI$,
	parameter  WT = $WT$,
	parameter  N = $N$,
	parameter  C = $C$,
	parameter  PE = $PE$,
	parameter  SIGNED = $SIGNED$,
	parameter  FPARG  = $FPARG$,
	parameter  BIAS   = $BIAS$,
	parameter  SETS = 1,
	parameter  THRESHOLDS_PATH = $THRESHOLDS_PATH$,
	parameter  USE_AXILITE = $USE_AXILITE$,
	parameter  DEPTH_TRIGGER_URAM = $DEPTH_TRIGGER_URAM$,
	parameter  DEPTH_TRIGGER_BRAM = $DEPTH_TRIGGER_BRAM$,
	parameter  DEEP_PIPELINE = $DEEP_PIPELINE$,
	parameter  O_BITS = $O_BITS$
)(
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	input   s_axilite_AWVALID,
	output  s_axilite_AWREADY,
	input [$clog2(SETS) + $clog2(C/PE) + $clog2(PE) + $clog2(N) + 1:0]  s_axilite_AWADDR,
	input         s_axilite_WVALID,
	output        s_axilite_WREADY,
	input [31:0]  s_axilite_WDATA,
	input [ 3:0]  s_axilite_WSTRB,
	output        s_axilite_BVALID,
	input         s_axilite_BREADY,
	output [1:0]  s_axilite_BRESP,
	input   s_axilite_ARVALID,
	output  s_axilite_ARREADY,
	input [$clog2(C/PE) + $clog2(PE) + $clog2(N) + 1:0]  s_axilite_ARADDR,
	output         s_axilite_RVALID,
	input          s_axilite_RREADY,
	output [31:0]  s_axilite_RDATA,
	output [ 1:0]  s_axilite_RRESP,

	output  in1_V_TREADY,
	input   in1_V_TVALID,
	input [(((SETS > 2? $clog2(SETS) : 1)+7)/8)*8-1:0]  in1_V_TDATA,

	output  in0_V_TREADY,
	input   in0_V_TVALID,
	input [((PE*WI+7)/8)*8-1:0]  in0_V_TDATA,

	input   out0_V_TREADY,
	output  out0_V_TVALID,
	output [((PE*O_BITS+7)/8)*8-1:0]  out0_V_TDATA
);

	thresholding_axi #(
		.N(N), .WI(WI), .WT(WT), .C(C), .PE(PE),
		.SIGNED(SIGNED), .FPARG(FPARG), .BIAS(BIAS), .SETS(SETS),
		.THRESHOLDS_PATH(THRESHOLDS_PATH), .USE_AXILITE(USE_AXILITE),
		.DEPTH_TRIGGER_URAM(DEPTH_TRIGGER_URAM), .DEPTH_TRIGGER_BRAM(DEPTH_TRIGGER_BRAM),
		.DEEP_PIPELINE(DEEP_PIPELINE)
	) core (
		.ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
		.s_axilite_AWVALID(s_axilite_AWVALID), .s_axilite_AWREADY(s_axilite_AWREADY), .s_axilite_AWADDR(s_axilite_AWADDR),
		.s_axilite_WVALID(s_axilite_WVALID), .s_axilite_WREADY(s_axilite_WREADY), .s_axilite_WDATA(s_axilite_WDATA), .s_axilite_WSTRB(s_axilite_WSTRB),
		.s_axilite_BVALID(s_axilite_BVALID), .s_axilite_BREADY(s_axilite_BREADY), .s_axilite_BRESP(s_axilite_BRESP),
		.s_axilite_ARVALID(s_axilite_ARVALID), .s_axilite_ARREADY(s_axilite_ARREADY), .s_axilite_ARADDR(s_axilite_ARADDR),
		.s_axilite_RVALID(s_axilite_RVALID), .s_axilite_RREADY(s_axilite_RREADY), .s_axilite_RDATA(s_axilite_RDATA), .s_axilite_RRESP(s_axilite_RRESP),
		.s_axis_set_tready(in1_V_TREADY), .s_axis_set_tvalid(in1_V_TVALID), .s_axis_set_tdata(in1_V_TDATA),
		.s_axis_tready(in0_V_TREADY), .s_axis_tvalid(in0_V_TVALID), .s_axis_tdata(in0_V_TDATA),
		.m_axis_tready(out0_V_TREADY), .m_axis_tvalid(out0_V_TVALID), .m_axis_tdata(out0_V_TDATA)
	);

endmodule
"""
)


def emit_thresholding_rtl(point, context, module_name: str = "thresholding_top") -> Artifacts:
    """Produce embedded-RTL Thresholding artifacts from a resolved point + context."""
    pe = point.PE
    num_channels = point.NumChannels

    idt = context.tensor_datatype(INPUT)
    wdt = context.tensor_datatype(THRESHOLDS)
    odt = point.outputDataType if "outputDataType" in point else context.tensor_datatype(OUTPUT)
    o_bits = odt.bitwidth()
    i_bits = idt.bitwidth()

    thresholds = np.asarray(context.initializer(THRESHOLDS))
    n_steps = thresholds.shape[1]
    act_val = int(point.ActVal)

    # Narrow-range quantization: the core expects 2^N-1 thresholds. If the op has one
    # fewer, adjust bias (signed) or widen wdt (unsigned).
    bias, wdt, thresholds, n_steps = _narrow_quant_adjust(
        thresholds, n_steps, o_bits, act_val, idt, odt, wdt
    )

    # If a single shared threshold row, the core treats C as PE.
    c_param = pe if thresholds.shape[0] == 1 else num_channels

    # Output container width (accounts for bias-shifted range).
    if bias >= 0:
        out_bits = math.ceil(math.log2(2 ** o_bits + bias))
    else:
        neg = -bias if -bias >= 2 ** (o_bits - 1) else 2 ** o_bits + bias
        out_bits = 1 + math.ceil(math.log2(neg))

    bindings = {
        "MODULE": module_name,
        "WI": i_bits,
        "WT": wdt.bitwidth(),
        "N": 2 ** o_bits - 1,
        "C": c_param,
        "PE": pe,
        "SIGNED": 1 if idt.signed() else 0,
        "FPARG": 0 if idt.is_integer() else 1,
        "BIAS": bias,
        "THRESHOLDS_PATH": '"./"',
        "USE_AXILITE": 0,  # runtime-writable weights out of scope
        "DEPTH_TRIGGER_URAM": int(point.get("depth_trigger_uram", 0)),
        "DEPTH_TRIGGER_BRAM": int(point.get("depth_trigger_bram", 0)),
        "DEEP_PIPELINE": int(point.get("deep_pipeline", 1)),
        "O_BITS": out_bits,
    }
    top = GeneratedFile(f"{module_name}.v", _V_WRAPPER, bindings)

    dat_files = _threshold_dat_files(thresholds, pe, c_param, o_bits, n_steps, wdt)

    return Artifacts(
        generated=(top,),
        data_files=tuple(dat_files),
        static_files=RTL_MANIFEST.static_files(),
        ipi=IPICommands(
            (f"create_bd_cell -type module -reference {module_name} {module_name}",)
        ),
    )


# ------------------------------------------------------------- pure helpers
# Faithful to FINN's make_weight_file / prepare_codegen_rtl_values; already self-free.


def _narrow_quant_adjust(thresholds, n_steps, o_bits, act_val, idt, odt, wdt):
    """Handle narrow-range quant: prepend/append a dummy threshold and shift bias/wdt
    so the core always sees 2^N-1 steps."""
    expected = 2 ** o_bits - 1
    bias = act_val
    if expected != n_steps:
        if odt.signed():
            bias = bias - 1
            thresholds = np.insert(thresholds, 0, wdt.min(), axis=1)
        else:
            max_val = wdt.max()
            if max_val > idt.max():
                thresholds = np.insert(thresholds, thresholds.shape[1], max_val, axis=1)
            else:
                max_val = max_val + 1
                wdt = (
                    DataType.get_smallest_possible(max_val)
                    if not wdt.signed()
                    else DataType.get_smallest_possible(-max_val - 1)
                )
                thresholds = np.insert(thresholds, thresholds.shape[1], max_val, axis=1)
        n_steps += 1
    return bias, wdt, thresholds, n_steps


def _threshold_dat_files(thresholds, pe, num_channels, o_bits, n_steps, wdt):
    """Produce threshs_{pe}_{stage}.dat memory-init files in the binary-search layout
    the core's $readmemh expects.

    DELIBERATELY NOT routed through the shared ``layout`` serializer: this is the FAR SIDE
    of the static-vs-data-dependent boundary predicate. The ROM address here is the runtime
    comparison outcome (which node the descending binary search visits at each stage —
    ``(i << (o_bits-stage)) + 2**(o_bits-stage-1) - 1``), NOT a static loop counter. The
    scatter across ``o_bits × PE`` files IS the binary-search decision tree; flattening it to
    a single ``layout`` blob would delete the resource win that is this RTL backend's reason
    to exist. So this threshold storage is FUSED to the compute core and stays here, by
    principle — unlike the HLS threshold ROM (separable, static-schedule), which DOES route
    through ``layout`` (``emit_hls.py``)."""
    t_expand = np.expand_dims(thresholds, axis=-1)
    bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
    t_packed = pack_innermost_dim_as_hex_string(t_expand, wdt, bw_hexdigit, prefix="")

    if t_packed.shape[0] == 1:
        t_packed = np.broadcast_to(t_packed, (pe, 2 ** o_bits - 1))
        num_channels = pe

    channel_fold = num_channels // pe
    files = []
    for stage in range(o_bits):
        for pe_value in range(pe):
            threshs = np.zeros([channel_fold * (2 ** stage)], dtype="object")
            for ch in range(channel_fold):
                for i in range(2 ** stage):
                    threshs[(ch << stage) + i] = t_packed[ch * pe + pe_value][
                        (i << (o_bits - stage)) + 2 ** (o_bits - stage - 1) - 1
                    ]
            content = "\n".join(str(v) for v in threshs) + "\n"
            files.append(DataFile(f"threshs_{pe_value}_{stage}.dat", content))
    return files
