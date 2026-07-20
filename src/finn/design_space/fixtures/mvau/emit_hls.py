############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
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

from finn.design_space.space import (
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

from .names import INPUT, OUTPUT, WEIGHTS

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


def emit_mvau_hls(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Produce embedded-mode HLS MVAU compute-core artifacts from a resolved point."""
    idt = context.tensor_datatype(INPUT)
    odt = point.outputDataType if "outputDataType" in point else context.tensor_datatype(OUTPUT)

    idt_hls = idt.get_hls_datatype_str()
    odt_hls = odt.get_hls_datatype_str()

    globals_ = ['#include "weights.hpp"', '#include "activations.hpp"', '#include "mvau.hpp"']

    defines = [
        f"#define MW1 {point.MW}",
        f"#define MH1 {point.MH}",
        f"#define SIMD1 {point.SIMD}",
        f"#define PE1 {point.PE}",
        f"#define WMEM1 {point.WMEM}",
        f"#define TMEM1 {point.TMEM}",
        f"#define numReps {int(np.prod(point.numInputVectors))}",
    ]

    # Non-bipolar integer case: TSrcI=Slice<idt>, TWeightI=Identity, TDstI=Slice<odt>.
    docompute = [
        f"Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, "
        f"Slice<{idt_hls}>, Slice<{odt_hls}>, Identity>\n"
        f"                (in0_V, out0_V, weights, PassThroughActivation<{odt_hls}>(), "
        f"numReps, {_MULT_STYLE[point.resType]});"
    ]

    blackbox = (
        f"void {module_name}(hls::stream<ap_uint<{point.instream_width}>> &in0_V,\n"
        f"                    hls::stream<ap_uint<{point.outstream_width}>> &out0_V\n"
        f"                    )"
    )

    pragmas = [
        "#pragma HLS INTERFACE axis port=in0_V",
        "#pragma HLS INTERFACE axis port=out0_V",
        "#pragma HLS INTERFACE ap_ctrl_none port=return",
        '#include "params.h"',
        "#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1",
    ]

    bindings = {
        "AP_INT_MAX_W": _ap_int_max_w(point, context),
        "GLOBALS": globals_,
        "DEFINES": defines,
        "BLACKBOXFUNCTION": blackbox,
        "PRAGMAS": pragmas,
        "DOCOMPUTE": docompute,
    }
    top = GeneratedFile(f"top_{module_name}.cpp", _CPP, bindings)

    params_h = DataFile("params.h", _params_h(point, context))

    # HLS embedded: weights are compiled into params.h (FixedPointWeights), so there
    # is NO weight-stream port — the blackbox exposes only in0_V/out0_V (both dataflow
    # boundary edges) + clk/rst. The absence of a WEIGHT_SINK IS the embedded topology:
    # the stitch finds nothing to bind, which is correct.
    ports = (
        Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V", index=0,
             width=point.instream_width, boundary=True),
        Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0,
             width=point.outstream_width, boundary=True),
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk"),
        Port(Direction.IN, Kind.RESET, Role.RESET, "ap_rst_n"),
    )

    return Artifacts(
        generated=(top,),
        data_files=(params_h,),
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
    return max(point.instream_width, point.outstream_width, weight_stream, 32)


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


def _params_h(point, context) -> str:
    """make_weight_file "hls_header" (matrixvectoractivation.py:659): pack the weight
    tensor into a FixedPointWeights/BinaryWeights C++ initializer."""
    weights = np.asarray(context.initializer(WEIGHTS))
    wdt = context.tensor_datatype(WEIGHTS)
    export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt

    tensor = _hw_weight_tensor(weights, point.MW, point.MH, point.PE, point.SIMD, point.WMEM, wdt)
    hls_code = numpy_to_hls_code(tensor, export_wdt, "weights", True, True)

    if export_wdt.bitwidth() != 1:
        head = "const FixedPointWeights<{},{},{},{}> weights = ".format(
            point.SIMD, export_wdt.get_hls_datatype_str(), point.PE, point.WMEM
        )
    else:
        head = "const BinaryWeights<{},{},{}> weights = ".format(
            point.SIMD, point.PE, point.WMEM
        )
    return head + hls_code
