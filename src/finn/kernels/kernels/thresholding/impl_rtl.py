############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# The .dat binary-search packing and the narrow-quant bias/threshold
# adjustment are lifted in behaviour from FINN's thresholding_rtl.py
# (prepare_codegen_rtl_values / make_weight_file), refactored to run as pure
# functions of (design_point, params, config) with no self.onnx_node access.
############################################################################
"""RTL backend for Thresholding.

FINN's RTL thresholding is inherently *embedded*: thresholds are baked into
per-(PE, stage) ``.dat`` memory-init files read by ``$readmemh`` in the core.
So this backend is a faithful second implementation for the embedded first cut,
and the decoupled/stitch machinery stays dormant (it only matters for the HLS
streaming mode, which is out of scope).

The generated top is the real ``finn-rtllib`` wrapper, filled through the typed
:class:`Template` — the ``$WI$``/``$WT$``/``$C$``… slots are validated, while
Verilog ``$clog2(...)`` expressions are left untouched (they are not
``$UPPERCASE$`` tokens).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.util.data_packing import pack_innermost_dim_as_hex_string

from ...derivation import ParameterSpec
from ...implementation import (
    Artifacts,
    DataFile,
    GeneratedFile,
    Implementation,
    IPICommands,
    ParamBundle,
    SelectionContext,
    StaticFile,
    Template,
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


def _has_uram(fpgapart: str) -> bool:
    """UltraRAM exists only on UltraScale+ and Versal. 7-series (``xc7*``),
    Zynq-7000, and Spartan have none. This is a coarse but real device rule;
    inside full FINN, prefer ``finn.util.basic`` part-family helpers."""
    p = fpgapart.lower()
    if p.startswith(("xcv", "xqv", "xcvm", "xcvc", "xcvp", "xcve")):  # Versal
        return True
    if p.startswith("xc7") or p.startswith("xa7") or p.startswith("xq7"):  # 7-series
        return False
    # UltraScale+ parts carry a trailing speed/family marker; the reliable
    # signal is the 'u' family tag (e.g. xcku, xczu, xcau). Plain UltraScale
    # (xcvu without '+') also lacks URAM, but xcu*+ / xczu* do.
    return "u+" in p or p.startswith(("xczu", "xcku", "xcau", "xcvu"))


@dataclass(frozen=True)
class _UramRequiresUltraScale:
    """Realization constraint: a URAM threshold-memory request is only buildable
    on parts that have UltraRAM. A ``realization``-phase constraint — evaluated
    at selection time against a device-aware context, not by the op-level builder.
    """

    @property
    def evaluation_phase(self) -> str:
        return "realization"

    def describe(self) -> str:
        return "URAM threshold memory requires an UltraScale+/Versal part"

    def check(self, ctx) -> str | None:
        try:
            wants_uram = int(ctx.get_param("depth_trigger_uram")) > 0
        except KeyError:
            wants_uram = False
        if wants_uram and not _has_uram(ctx.fpgapart):
            return (
                f"depth_trigger_uram set but {ctx.fpgapart} has no UltraRAM "
                f"(UltraScale+/Versal only)"
            )
        return None


class ThresholdingRTL(Implementation):
    """Embedded RTL implementation of Thresholding (finn-rtllib)."""

    name = "Thresholding_rtl"
    op_kind = "Thresholding"
    language = "rtl"
    priority = 0  # preferred over HLS when feasible (lower = better)

    def dse_parameters(self) -> Mapping[str, ParameterSpec]:
        # RTL-only design-space parameters. Contributed to the composed space
        # when RTL is selected; auto-registered as nodeattrs, round-trip, sweepable.
        # depth_trigger_{uram,bram}: force local mems of >= this depth into
        # URAM/BRAM (0 = off). deep_pipeline: extra timing-closure stages.
        return {
            "depth_trigger_uram": ParameterSpec(
                "depth_trigger_uram", [0, 256, 512, 1024, 2048], default=0
            ),
            "depth_trigger_bram": ParameterSpec(
                "depth_trigger_bram", [0, 256, 512, 1024, 2048], default=0
            ),
            "deep_pipeline": ParameterSpec("deep_pipeline", {0, 1}, default=1),
        }

    # ------------------------------------------------------------- feasibility
    def realization_constraints(self):
        """Device-aware feasibility as ``realization``-phase constraint data — the
        seam neither prior system had on the backend itself. The finn-rtllib core
        packs threshold memory into URAM when ``depth_trigger_uram`` is requested,
        and UltraRAM exists only on UltraScale+/Versal; the HLS path (LUTRAM/BRAM)
        is unaffected. Exactly the ``fpgapart``-dependent guard the baseline
        centralized in ``_mvu_rtl_possible`` and the prototype's context-free
        ``Callable[[Kernel], bool]`` could not express."""
        return [_UramRequiresUltraScale()]

    # -------------------------------------------------------------------- emit
    def emit(
        self,
        design_point: Any,
        params: ParamBundle,
        config: Mapping[str, Any],
    ) -> Artifacts:
        dp = design_point
        module = config["module_name"]
        pe = dp.config["PE"]
        num_channels = dp.inputs["input"].tensor_shape[-1]

        idt = dp.inputs["input"].datatype
        wdt = dp.inputs["thresholds"].datatype
        odt = dp.outputs["output"].datatype
        o_bits = odt.bitwidth()
        i_bits = idt.bitwidth()

        thresholds = np.asarray(params["thresholds"])
        n_steps = thresholds.shape[1]
        act_val = int(config["act_val"])

        # Narrow-range quantization: the core expects 2^N-1 thresholds. If the op
        # has one fewer, adjust bias (signed) or widen wdt (unsigned).
        bias, wdt, thresholds, n_steps = self._narrow_quant_adjust(
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
            "MODULE": module,
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
            "DEPTH_TRIGGER_URAM": int(config.get("depth_trigger_uram", 0)),
            "DEPTH_TRIGGER_BRAM": int(config.get("depth_trigger_bram", 0)),
            "DEEP_PIPELINE": int(config.get("deep_pipeline", 1)),
            "O_BITS": out_bits,
        }
        top = GeneratedFile(f"{module}.v", _V_WRAPPER, bindings)

        dat_files = self._threshold_dat_files(
            thresholds, pe, c_param, o_bits, n_steps, wdt
        )

        return Artifacts(
            generated=(top,),
            data_files=tuple(dat_files),
            static_files=(
                StaticFile("finn.data", "finn-rtllib/thresholding/hdl/thresholding.sv"),
                StaticFile("finn.data", "finn-rtllib/thresholding/hdl/thresholding_axi.sv"),
                StaticFile("finn.data", "finn-rtllib/axi/hdl/axilite.sv"),
            ),
            ipi=IPICommands((f"create_bd_cell -type module -reference {module} {module}",)),
        )

    # ------------------------------------------------------------- pure helpers
    @staticmethod
    def _narrow_quant_adjust(thresholds, n_steps, o_bits, act_val, idt, odt, wdt):
        """Handle narrow-range quant: prepend/append a dummy threshold and shift
        bias/wdt so the core always sees 2^N-1 steps. Pure; lifted from
        make_weight_file + prepare_codegen_rtl_values."""
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

    @staticmethod
    def _threshold_dat_files(thresholds, pe, num_channels, o_bits, n_steps, wdt):
        """Produce threshs_{pe}_{stage}.dat memory-init files in the binary-search
        layout the core's $readmemh expects. Pure; lifted from make_weight_file
        (internal_embedded mode)."""
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
