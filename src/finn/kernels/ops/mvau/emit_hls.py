############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The .cpp fragments reproduce matrixvectoractivation_hls.py (global_includes/
# defines/docompute/blackboxfunction/pragmas, internal_embedded branch) and the
# params.h packing reproduces get_hw_compatible_weight_tensor + make_weight_file
# ("hls_header"), as pure functions of (point, context) with no node/model access.
############################################################################

"""Hermetic HLS codegen for the MVAU compute core (internal_embedded mode).

Assembles the Vitis-HLS ``top_*.cpp`` from the resolved flat ``Point`` and packs the
weight initializer into ``params.h`` (the ``FixedPointWeights``/``BinaryWeights`` C++
header the finn-hlslib ``Matrix_Vector_Activate_Batch`` template expects). Static
finn-hlslib headers are referenced by path as ``StaticFile`` (the adapter resolves
them under full FINN; emit never touches ``FINN_ROOT``).

SCOPE — embedded, noActivation (PassThroughActivation) compute half only. The
decoupled weight-STREAM path (``Matrix_Vector_Activate_Stream_Batch`` + in1_V) is the
deferred composition arc.
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

from finn.kernels.space import (
    Artifacts,
    DataFile,
    Direction,
    GeneratedFile,
    Kind,
    Port,
    Role,
    StaticFile,
    Template,
)
from finn.util.data_packing import numpy_to_hls_code

from .geometry import mvau_geometry
from .op import INPUT, OUTPUT, THRESHOLDS, WEIGHTS

_MULT_STYLE = {"auto": "ap_resource_dflt()", "lut": "ap_resource_lut()", "dsp": "ap_resource_dsp()"}

# The ipgen .cpp template (templates.py:ipgen_template), typed. $DEFINES$/$DOCOMPUTE$/
# $PRAGMAS$ etc. are filled from the Point; $AP_INT_MAX_W$ is a fixed HLS knob.
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

# NB (F4 HLS asymmetry): unlike the RTL wrapper's discrete #() params, the HLS
# geometry lands INSIDE free-form `#define` strings ($DEFINES$ is a list of lines),
# not discrete typed slots. So HLS keeps Raw slots and the typed-bind win is RTL-side;
# the geometry here is still sourced from the ONE shared accessor (mvau_geometry), so
# HLS and RTL cannot diverge (F3). Full HLS typing = standardize the HLS surface = Arc 2.


def emit_mvau_hls(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Produce embedded-mode HLS MVAU compute-core artifacts from a resolved point."""
    idt = context.tensor_datatype(INPUT)
    odt = point.outputDataType if "outputDataType" in point else context.tensor_datatype(OUTPUT)
    geo = mvau_geometry(point, context)

    idt_hls = idt.get_hls_datatype_str()
    odt_hls = odt.get_hls_datatype_str()

    # THRESHOLDS are OPTIONAL (a 3-input fused MVU) and ALWAYS constant here — baked into
    # thresh.h as a ThresholdsActivation, exactly like weights bake into params.h. Existence
    # is emergent: a threshold initializer is attached. When absent, the activation is a
    # PassThrough (output IS the accumulator, base:517).
    has_thresh = context.initializer(THRESHOLDS) is not None

    globals_ = ['#include "weights.hpp"', '#include "activations.hpp"', '#include "mvau.hpp"']
    if has_thresh:
        globals_.append('#include "thresh.h"')

    defines = [
        f"#define MW1 {geo.MW}",
        f"#define MH1 {geo.MH}",
        f"#define SIMD1 {point.SIMD}",
        f"#define PE1 {point.PE}",
        f"#define WMEM1 {geo.depth}",
        f"#define TMEM1 {geo.tdepth}",
        f"#define numReps {geo.num_reps}",
    ]

    # Non-bipolar integer case: TSrcI=Slice<idt>, TWeightI=Identity, TDstI=Slice<odt>. The
    # activation is the baked `threshs` ThresholdsActivation when present, else PassThrough.
    activation = "threshs" if has_thresh else f"PassThroughActivation<{odt_hls}>()"
    docompute = [
        f"Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, "
        f"Slice<{idt_hls}>, Slice<{odt_hls}>, Identity>\n"
        f"                (in0_V, out0_V, weights, {activation}, "
        f"numReps, {_MULT_STYLE[point.resType]});"
    ]

    blackbox = (
        f"void {module_name}(hls::stream<ap_uint<{point['stream_width.inp']}>> &in0_V,\n"
        f"                    hls::stream<ap_uint<{point['stream_width.out']}>> &out0_V\n"
        f"                    )"
    )

    pragmas = [
        "#pragma HLS INTERFACE axis port=in0_V",
        "#pragma HLS INTERFACE axis port=out0_V",
        "#pragma HLS INTERFACE ap_ctrl_none port=return",
        '#include "params.h"',
        "#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1",
    ]
    if has_thresh:
        pragmas.append("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1")
        pragmas.append("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3")

    bindings = {
        "AP_INT_MAX_W": _ap_int_max_w(point, context),
        "GLOBALS": globals_,
        "DEFINES": defines,
        "BLACKBOXFUNCTION": blackbox,
        "PRAGMAS": pragmas,
        "DOCOMPUTE": docompute,
    }
    top = GeneratedFile(f"top_{module_name}.cpp", _CPP, bindings)

    params_h = DataFile("params.h", _params_h(point, context, geo))
    # thresh.h — the baked ThresholdsActivation, only when the node has thresholds (constant
    # mode). Like params.h, no port: it is compiled into the core.
    data_files = (params_h,)
    if has_thresh:
        data_files += (DataFile("thresh.h", _thresh_h(point, context, geo)),)

    # HLS embedded: weights are compiled into params.h (FixedPointWeights), so there
    # is NO weight-stream port — the blackbox exposes only in0_V/out0_V (both dataflow
    # boundary edges) + clk/rst. The absence of a WEIGHT_SINK IS the embedded topology:
    # the stitch finds nothing to bind, which is correct.
    ports = (
        Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V", index=0,
             width=point["stream_width.inp"], boundary=True),
        Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0,
             width=point["stream_width.out"], boundary=True),
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk"),
        Port(Direction.IN, Kind.RESET, Role.RESET, "ap_rst_n"),
    )

    return Artifacts(
        generated=(top,),
        data_files=data_files,
        ports=ports,
        static_files=(
            StaticFile("finn.data", "deps/finn-hlslib/weights.hpp"),
            StaticFile("finn.data", "deps/finn-hlslib/activations.hpp"),
            StaticFile("finn.data", "deps/finn-hlslib/mvau.hpp"),
            StaticFile("finn.data", "deps/finn-hlslib/bnn-library.h"),
        ),
    )


def _ap_int_max_w(point, context) -> int:
    # A safe upper bound on the widest ap_uint the design streams (baseline computes
    # this from stream widths; the packed weight header dominates here).
    wdt = context.tensor_datatype(WEIGHTS)
    weight_stream = point.PE * point.SIMD * wdt.bitwidth()
    return max(point["stream_width.inp"], point["stream_width.out"], weight_stream, 32)


# ------------------------------------------------------------- pure helpers


def _hw_weight_tensor(weights, mw, mh, pe, simd, wmem, wdt):
    """get_hw_compatible_weight_tensor (matrixvectoractivation.py:602): transpose to
    hlslib layout, bipolar->binary, interleave rows across PEs, reshape (1,PE,WMEM,
    SIMD), reverse SIMD. Pure."""
    ret = weights.T
    if wdt == DataType["BIPOLAR"]:
        ret = (ret + 1) / 2
    ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
    ret = ret.reshape(1, pe, wmem, simd)
    ret = np.flip(ret, axis=-1)
    return ret


def _params_h(point, context, geo) -> str:
    """make_weight_file "hls_header" (matrixvectoractivation.py:659): pack the weight
    tensor into a FixedPointWeights/BinaryWeights C++ initializer."""
    weights = np.asarray(context.initializer(WEIGHTS))
    wdt = context.tensor_datatype(WEIGHTS)
    export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt

    tensor = _hw_weight_tensor(weights, geo.MW, geo.MH, point.PE, point.SIMD, geo.depth, wdt)
    hls_code = numpy_to_hls_code(tensor, export_wdt, "weights", True, True)

    if export_wdt.bitwidth() != 1:
        head = "const FixedPointWeights<{},{},{},{}> weights = ".format(
            point.SIMD, export_wdt.get_hls_datatype_str(), point.PE, geo.depth
        )
    else:
        head = "const BinaryWeights<{},{},{}> weights = ".format(
            point.SIMD, point.PE, geo.depth
        )
    return head + hls_code


def _hw_threshold_tensor(thresholds, mh, pe, tmem, n_steps):
    """get_hw_compatible_threshold_tensor (matrixvectoractivation.py:584): tile a
    per-tensor (1, n_steps) threshold matrix up to MH channels, interleave rows across
    PEs, reshape (1, PE, TMEM, n_steps). Pure."""
    ret = thresholds
    if ret.shape[0] == 1:
        ret = np.tile(ret, (mh, 1))
    ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
    return ret.reshape(1, pe, tmem, n_steps)


def _thresh_h(point, context, geo) -> str:
    """Bake thresholds into a ThresholdsActivation C++ initializer (matrixvectoractivation.py:
    868-914), the constant-mode HLS threshold delivery. Pure over (point, context): reads the
    threshold VALUES + the resolved threshold dtype (``thresholdDataType``), never the graph."""
    thresholds = np.asarray(context.initializer(THRESHOLDS))
    tdt = point.thresholdDataType
    odt = point.outputDataType if "outputDataType" in point else context.tensor_datatype(OUTPUT)
    export_odt = DataType["BINARY"] if odt == DataType["BIPOLAR"] else odt

    n_steps = thresholds.shape[-1]
    tensor = _hw_threshold_tensor(thresholds, geo.MH, point.PE, geo.tdepth, n_steps)
    hls_code = numpy_to_hls_code(tensor, tdt, "thresholds", False, True)

    tdt_hls = tdt.get_hls_datatype_str()
    odt_hls = export_odt.get_hls_datatype_str()
    head = "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs = ".format(
        geo.tdepth, point.PE, n_steps, tdt_hls, odt_hls, point.ActVal,
        f"comp::less_equal<{tdt_hls}, {tdt_hls}>",
    )
    return head + hls_code
