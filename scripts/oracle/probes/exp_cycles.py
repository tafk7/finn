# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The HWCustomOp flow's cycle estimate (``get_exp_cycles``) of ``MVAU`` and
``Thresholding`` at each folding ``tests/kernel_ops/measure_cycles.py`` measures: the
Chain's layers and TFC_W2A2's at 16 lanes. The nodes are made as ``measure_cycles``
makes them: the attributes below, INT8 values, INT32 accumulators."""

from _probe import arguments, write
from onnx import helper
from qonnx.custom_op.registry import getCustomOp

MVAU = [
    # MW, MH, SIMD, PE, numInputVectors
    (4, 4, 2, 2, [3]),  # the Chain's first and second
    (784, 64, 16, 16, [1]),  # TFC's MatMul_0
    (64, 64, 16, 16, [1]),  # TFC's MatMul_1 and MatMul_2
    (64, 10, 16, 10, [1]),  # TFC's MatMul_3
]
THRESHOLDING = [
    # NumChannels, PE, numInputVectors
    (4, 2, [3]),  # the Chain's activate
    (784, 16, [1]),  # TFC's MultiThreshold_0
    (64, 16, [1]),  # TFC's MultiThreshold_1 to _3
]
DOMAIN = "finn.custom_op.fpgadataflow"

raw, _ = arguments()
values = []
for mw, mh, simd, pe, vectors in MVAU:
    attributes = dict(MW=mw, MH=mh, SIMD=simd, PE=pe, numInputVectors=vectors)
    node = helper.make_node(
        "MVAU",
        ["x", "w"],
        ["y"],
        domain=DOMAIN,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        **attributes,
    )
    values.append(
        {"op": "MVAU", "attributes": attributes, "cycles": getCustomOp(node).get_exp_cycles()}
    )
for channels, pe, vectors in THRESHOLDING:
    attributes = dict(NumChannels=channels, PE=pe, numInputVectors=vectors)
    node = helper.make_node(
        "Thresholding",
        ["x", "t"],
        ["y"],
        domain=DOMAIN,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT8",
        **attributes,
    )
    values.append(
        {
            "op": "Thresholding",
            "attributes": attributes,
            "cycles": getCustomOp(node).get_exp_cycles(),
        }
    )
write(raw, values)
