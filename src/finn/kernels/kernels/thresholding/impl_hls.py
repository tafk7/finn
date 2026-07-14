############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Threshold packing math (get_hw_compatible_threshold_tensor, the thresh.h
# ThresholdsActivation writer) is lifted verbatim in behaviour from FINN's
# thresholding_hls.py — only its *home* changes: it now runs as a pure function
# of (design_point, params, config), never off self.onnx_node.
############################################################################
"""HLS backend for Thresholding — embedded mode.

Thresholds are baked into a generated ``thresh.h`` (``ThresholdsActivation<>``);
the compute is a single ``Thresholding_Batch<>`` call in the generated ``.cpp``.
No ``mem_mode`` branching: this backend *is* the embedded strategy, so the
~15-method cross-cutting conditional of the mixin designs simply does not exist
here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

from finn.util.data_packing import numpy_to_hls_code

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

# The generated top .cpp. Typed slots (validated by Template), no $KEY$ str.replace.
_CPP_TEMPLATE = Template(
    """\
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "ap_int.h"
#include "hls_stream.h"
#include "bnn-library.h"
#include "activations.hpp"
#include "thresh.h"

$DEFINES$

void $MODULE$(
    hls::stream<ap_uint<$IN_WIDTH$>> &in0_V,
    hls::stream<ap_uint<$OUT_WIDTH$>> &out0_V
) {
$PRAGMAS$
$DOCOMPUTE$
}
"""
)


class ThresholdingHLS(Implementation):
    """Embedded-mode HLS implementation of Thresholding."""

    name = "Thresholding_hls"
    op_kind = "Thresholding"
    language = "hls"
    priority = 10  # HLS is the fallback; RTL (lower number) preferred when feasible
    knob_specs: Mapping[str, tuple] = {
        # threshold-memory resource type; nodeattr so it round-trips + is DSE-visible
        "ram_style": ("s", False, "distributed", {"distributed", "block"}),
    }

    # ------------------------------------------------------------- feasibility
    def precondition(self, ctx: SelectionContext) -> bool:
        """HLS-embedded builds integer thresholding on any part. Thresholding's
        schema already constrains input/output to integer; nothing device- or
        dtype-specific blocks HLS, so it is always feasible (the always-viable
        fallback)."""
        return True

    # -------------------------------------------------------------------- emit
    def emit(
        self,
        design_point: Any,
        params: ParamBundle,
        config: Mapping[str, Any],
    ) -> Artifacts:
        dp = design_point
        inp = dp.inputs["input"]
        idt = inp.datatype
        tdt = dp.inputs["thresholds"].datatype
        odt = dp.outputs["output"].datatype

        num_channels = inp.tensor_shape[-1]
        pe = dp.config["PE"]
        # Folding is DERIVED, not hand-computed: block_folding_factor is the
        # channel fold (== NumChannels/PE == TMEM); tensor_blocks_shape counts
        # the spatial vectors (N*H*W). Read them from the design point rather
        # than open-coding // pe and np.prod(tensor_shape[:-1]).
        tmem = inp.block_folding_factor
        img_dim = int(np.prod(inp.tensor_blocks_shape))
        act_val = int(config["act_val"])

        in_width = dp.inputs["input"].stream_width_bits
        out_width = dp.outputs["output"].stream_width_bits
        module = config["module_name"]

        thresholds = self._hw_threshold_tensor(params["thresholds"], num_channels, pe, tmem)
        n_steps = thresholds.shape[-1]

        # --- param file: thresh.h with the ThresholdsActivation instance ---
        thresh_h = self._thresh_header(thresholds, tmem, pe, n_steps, tdt, odt, act_val)

        # --- generated top .cpp via typed template ---
        tsrc = f"Slice<{idt.get_hls_datatype_str()}>"
        tdst = f"Slice<{odt.get_hls_datatype_str()}>"
        bindings = {
            "AP_INT_MAX_W": max(in_width, out_width),
            "MODULE": module,
            "IN_WIDTH": in_width,
            "OUT_WIDTH": out_width,
            "DEFINES": [
                f"#define NumChannels1 {num_channels}",
                f"#define PE1 {pe}",
                f"#define numReps 1",
                f"#define ImgDim1 {img_dim}",
            ],
            "PRAGMAS": self._pragmas(pe, num_channels, config["ram_style"]),
            "DOCOMPUTE": (
                f"    Thresholding_Batch<ImgDim1, NumChannels1, PE1, {tsrc}, {tdst}>"
                f"(in0_V, out0_V, threshs, numReps);"
            ),
        }
        cpp = GeneratedFile(f"{module}.cpp", _CPP_TEMPLATE, bindings)

        return Artifacts(
            generated=(cpp,),
            data_files=(DataFile("thresh.h", thresh_h),),
            static_files=(StaticFile(package="finn.data", resource="finn-hlslib"),),
            ipi=IPICommands((f"create_bd_cell -type ip -vlnv $VLNV$ {module}",)),
        )

    # ------------------------------------------------------------- pure helpers
    @staticmethod
    def _hw_threshold_tensor(
        orig: np.ndarray, num_channels: int, pe: int, tmem: int
    ) -> np.ndarray:
        """Reshape thresholds to (PE, TMEM, n_steps) with PE-interleaving.
        Lifted from get_hw_compatible_threshold_tensor; pure over its inputs."""
        assert num_channels % pe == 0, f"NumChannels={num_channels} not divisible by PE={pe}"
        assert orig.ndim == 2, "threshold matrix must be 2-D"
        n_steps = orig.shape[1]
        ret = orig
        if ret.shape[0] == 1:
            ret = np.tile(ret, (num_channels, 1))
        assert ret.shape[0] == num_channels, "threshold channel count mismatch"
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        return ret.reshape(pe, tmem, n_steps)

    @staticmethod
    def _thresh_header(
        thresholds: np.ndarray,
        tmem: int,
        pe: int,
        n_steps: int,
        tdt: DataType,
        odt: DataType,
        act_val: int,
    ) -> str:
        """Render thresh.h. numpy_to_hls_code needs (1, PE, TMEM, n_steps)."""
        arr = thresholds.reshape(1, pe, tmem, n_steps)
        code = numpy_to_hls_code(arr, tdt, "thresholds", False, True)
        tdt_hls = tdt.get_hls_datatype_str()
        export_odt = DataType["BINARY"] if odt == DataType["BIPOLAR"] else odt
        odt_hls = export_odt.get_hls_datatype_str()
        decl = (
            f"static ThresholdsActivation<{tmem},{pe},{n_steps},{tdt_hls},{odt_hls},"
            f"{act_val},comp::less_equal<{tdt_hls}, {tdt_hls}>> threshs = "
        )
        return decl + code

    @staticmethod
    def _pragmas(pe: int, num_channels: int, ram_style: str) -> list[str]:
        out = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1",
            "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3",
        ]
        if pe < num_channels:
            core = "ROM_2P_LUTRAM" if ram_style == "distributed" else "ROM_2P_BRAM"
            out.append(f"#pragma HLS RESOURCE variable=threshs.m_thresholds core={core}")
        return ["    " + p for p in out]
