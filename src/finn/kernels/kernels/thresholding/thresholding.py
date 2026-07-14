############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Thresholding op identity.

Applies a multi-threshold activation: each input value is compared against a
per-channel set of thresholds to produce a quantized output. The schema mirrors
the real Brainsmith ``THRESHOLDING_SCHEMA``; the golden model delegates to
QONNX ``multithreshold``.

Scope: **embedded-only**. The thresholds are baked into the generated hardware
(HLS header / RTL ``.dat``); the orthogonal ``mem_mode`` / decoupled-streaming
axis is deliberately out of the first cut.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold

from ...derivation import (
    FULL_DIM,
    VALUE_OPTIMIZED,
    DatatypeInteger,
    DimensionDivisible,
    InputSchema,
    IsDynamic,
    IsStatic,
    KernelSchema,
    OutputSchema,
    ShapeHierarchy,
    derive_dim,
)
from ...implementation import ParamBundle
from ...kernel_op import KernelOp

# =============================================================================
# Schema (mirrors real Brainsmith THRESHOLDING_SCHEMA)
# =============================================================================

THRESHOLDING_SCHEMA = KernelSchema(
    name="Thresholding",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=[FULL_DIM],  # process full spatial dims
            stream_tiling=["PE"],  # parallel channels via PE
            required_layout="NHWC",
        ),
        InputSchema(
            name="thresholds",
            block_tiling=[],  # static data: not tiled
            stream_tiling=[],  # static data: not streamed
            datatype=VALUE_OPTIMIZED,  # narrow to actual values
        ),
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],
            stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],  # match input PE
            datatype=None,  # from graph nodeattr
            required_layout="NHWC",
        )
    ],
    kernel_params={
        "num_steps": ("i", True, 1),  # threshold steps
        "act_val": ("i", False, 0),  # activation bias (ActVal)
    },
    constraints=[
        IsDynamic(("input",)),
        IsStatic(("thresholds",)),
        DimensionDivisible("input", -1, "PE", hierarchy=ShapeHierarchy.STREAM),
        DatatypeInteger(("input", "output")),
    ],
)


class ThresholdingOp(KernelOp):
    """Identity for multi-threshold activation."""

    op_kind = "Thresholding"

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> KernelSchema:
        return THRESHOLDING_SCHEMA

    # --------------------------------------------------------------- reference
    def reference(
        self, inputs: Mapping[str, Any], attrs: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Golden model. Input is NHWC (or NC); ``multithreshold`` expects
        channels-first, so transpose 4-D inputs around the call."""
        inp = np.asarray(inputs["input"])
        thr = np.asarray(inputs["thresholds"])
        out_bias = attrs.get("act_val", 0)

        is_4d = inp.ndim == 4
        if is_4d:
            inp = np.transpose(inp, (0, 3, 1, 2))

        y = multithreshold(inp, thr, out_bias=out_bias)

        if is_4d:
            y = y.transpose(0, 2, 3, 1)

        out_dt = attrs.get("output0Datatype")
        if out_dt is not None and DataType[out_dt] == DataType["BIPOLAR"]:
            y = 2 * y - 1

        return {"output": y.astype(np.float32)}

    # --------------------------------------------------------------- inference
    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        if node.op_type != "MultiThreshold":
            return False
        from qonnx.custom_op.registry import getCustomOp

        mt = getCustomOp(node)
        return mt.get_nodeattr("out_scale") == 1.0 and int(
            mt.get_nodeattr("out_bias")
        ) == mt.get_nodeattr("out_bias")

    # ----------------------------------------------------------- param extract
    def extract_params(self, node: NodeProto, model: ModelWrapper) -> ParamBundle:
        thresholds = model.get_initializer(node.input[1])
        if thresholds is None:
            raise ValueError(f"{node.name}: thresholds input has no initializer")
        return ParamBundle({"thresholds": np.asarray(thresholds)})
