############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# The .cpp fragments reproduce thresholding_hls.py (global_includes/defines/docompute/
# blackboxfunction/pragmas, internal_embedded branch) and the thresh.h packing reproduces
# get_hw_compatible_threshold_tensor + make_weight_file("hls_header") via the shared
# parameter serializer (layout, THRESHOLD traversal), as pure functions of (point, context)
# with no node/model access.
############################################################################

"""Hermetic HLS codegen for the Thresholding compute core (internal_embedded mode).

Assembles the Vitis-HLS ``top_*.cpp`` from the resolved flat ``Point`` and bakes the
threshold initializer into ``thresh.h`` (the ``ThresholdsActivation`` C++ object the
finn-hlslib ``Thresholding_Batch`` template expects). The threshold ROM is a SEPARABLE,
static-schedule memory — read in full each output beat, indexed by the output-channel
LOOP — so its serialization routes through the shared ``layout`` (THRESHOLD traversal),
the same serializer MVAU's fused-threshold ROM uses. (The RTL binary-search backend's
threshold storage is data-dependent-addressed and stays FUSED to that core; see
``emit_rtl.py``.)

SCOPE — embedded threshold delivery (baked ROM, no stream port). The decoupled
threshold-STREAM path (``Thresholding_Stream_Batch`` + in1_V) is deferred, exactly as the
MVAU HLS compute half deferred its weight-stream path.
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Artifacts,
    ArtifactManifest,
    DataFile,
    Direction,
    GeneratedFile,
    Kind,
    Port,
    Role,
    SourceFile,
    Template,
)
from finn.kernels.ops.parameters.serialize import layout, threshold_constraint

from .names import INPUT, OUTPUT, THRESHOLDS

# The single source-of-truth for the HLS backend's static finn-hlslib headers (F9). The HLS
# bundle's `sources` is its Python codegen module (thresholding_hls.py — a bookkeeping
# marker), so the manifest here covers the emitted static build sources the top .cpp needs.
HLS_STATIC_MANIFEST = ArtifactManifest(
    sources=(
        SourceFile("activations.hpp", root="deps/finn-hlslib"),
        SourceFile("bnn-library.h", root="deps/finn-hlslib"),
    ),
)

# The ipgen .cpp template (mirrors the MVAU one). $DEFINES$/$DOCOMPUTE$/$PRAGMAS$ etc. are
# filled from the Point; $AP_INT_MAX_W$ is a fixed HLS knob.
_CPP = Template(
    """\
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "bnn-library.h"

$GLOBALS$

$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$

$DOCOMPUTE$
}
"""
)


def emit_thresholding_hls(point, context, module_name: str = "thresholding_top") -> Artifacts:
    """Produce embedded-mode HLS Thresholding compute-core artifacts from a resolved point."""
    idt = context.tensor_datatype(INPUT)
    odt = point.outputDataType if "outputDataType" in point else context.tensor_datatype(OUTPUT)
    export_odt = DataType["BINARY"] if odt == DataType["BIPOLAR"] else odt

    idt_hls = idt.get_hls_datatype_str()
    odt_hls = export_odt.get_hls_datatype_str()

    num_channels = point.NumChannels
    pe = point.PE
    total_spatial = int(np.prod(context.tensor_shape(INPUT)[:-1]))

    globals_ = ['#include "activations.hpp"', '#include "thresh.h"']

    defines = [
        f"#define NumChannels1 {num_channels}",
        f"#define PE1 {pe}",
        "#define numReps 1",
        f"#define ImgDim1 {total_spatial}",
    ]

    # TSrcI=Slice<idt>, TDstI=Slice<odt>; the baked `threshs` ThresholdsActivation carries
    # the comparison + bias (the constant-mode embedded threshold delivery).
    docompute = [
        f"Thresholding_Batch<ImgDim1, NumChannels1, PE1, Slice<{idt_hls}>, Slice<{odt_hls}>>\n"
        f"                (in0_V, out0_V, threshs, numReps);"
    ]

    in_width = point.instream_width
    out_width = point.outstream_width
    blackbox = (
        f"void {module_name}(hls::stream<ap_uint<{in_width}>> &in0_V,\n"
        f"                    hls::stream<ap_uint<{out_width}>> &out0_V\n"
        f"                    )"
    )

    pragmas = [
        "#pragma HLS INTERFACE axis port=in0_V",
        "#pragma HLS INTERFACE axis port=out0_V",
        "#pragma HLS INTERFACE ap_ctrl_none port=return",
        "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1",
        "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3",
    ]

    bindings = {
        "AP_INT_MAX_W": max(in_width, out_width, 32),
        "GLOBALS": globals_,
        "DEFINES": defines,
        "BLACKBOXFUNCTION": blackbox,
        "PRAGMAS": pragmas,
        "DOCOMPUTE": docompute,
    }
    top = GeneratedFile(f"top_{module_name}.cpp", _CPP, bindings)

    data_files = (DataFile("thresh.h", _thresh_h(point, context)),)

    # HLS embedded: thresholds are compiled into thresh.h (a baked ThresholdsActivation), so
    # there is NO threshold-stream port — the blackbox exposes only in0_V/out0_V (dataflow
    # boundary edges) + clk/rst. The absence of a WEIGHT_SINK IS the constant (embedded)
    # topology: the stitch finds nothing to bind, which is correct.
    ports = (
        Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V", index=0,
             width=in_width, boundary=True),
        Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0,
             width=out_width, boundary=True),
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk"),
        Port(Direction.IN, Kind.RESET, Role.RESET, "ap_rst_n"),
    )

    return Artifacts(
        generated=(top,),
        data_files=data_files,
        ports=ports,
        static_files=HLS_STATIC_MANIFEST.static_files(),
    )


# ------------------------------------------------------------- pure helpers


def _thresh_h(point, context) -> str:
    """Bake thresholds into a ThresholdsActivation C++ initializer (thresholding_hls.py:227),
    the constant-mode embedded threshold delivery. The serialized body comes from the shared
    ``layout`` (THRESHOLD traversal, CPP_HEADER form) — the SEPARABLE static-schedule ROM."""
    thresholds = np.asarray(context.initializer(THRESHOLDS))
    tdt = point.thresholdDataType
    odt = point.outputDataType if "outputDataType" in point else context.tensor_datatype(OUTPUT)
    export_odt = DataType["BINARY"] if odt == DataType["BIPOLAR"] else odt

    n_steps = thresholds.shape[-1]
    tmem = point.TMEM
    mh = point.NumChannels
    hls_code = layout(
        thresholds, threshold_constraint(point.PE, tmem, n_steps, mh, tdt)
    ).text

    tdt_hls = tdt.get_hls_datatype_str()
    odt_hls = export_odt.get_hls_datatype_str()
    head = "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs = ".format(
        tmem, point.PE, n_steps, tdt_hls, odt_hls, point.ActVal,
        f"comp::less_equal<{tdt_hls}, {tdt_hls}>",
    )
    return head + hls_code
