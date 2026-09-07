# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN pass owners for QONNX inference over DataflowOps.

Import these transformations for graphs containing DataflowOps. QONNX's
ModelWrapper.transform copies/preprocesses the model before calling apply, so
the source-analysis context belongs inside apply, on its actual argument.
Each fixed-point iteration is a separate pass with a fresh summary result.
The pinned QONNX transformations and ModelWrapper contract are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from qonnx.transformation.infer_datatypes import (  # type: ignore[import-not-found]
    InferDataTypes as QonnxInferDataTypes,
)
from qonnx.transformation.infer_shapes import (  # type: ignore[import-not-found]
    InferShapes as QonnxInferShapes,
)

from finn.dataflow.ops.base import DATAFLOW_DOMAIN
from finn.dataflow.ops.reconstruction import source_analysis


@contextmanager
def _inference_source_analysis(model: Any) -> Iterator[None]:
    if any(node.domain == DATAFLOW_DOMAIN for node in model.graph.node):
        with source_analysis(model, fresh=True):
            yield
    else:
        # Ordinary QONNX/legacy FINN graphs need no DataflowOp source analysis.
        yield


class InferDataTypes(QonnxInferDataTypes):  # type: ignore[misc]
    """QONNX datatype inference with one initializer analysis per apply pass."""

    def apply(self, model: Any) -> tuple[Any, bool]:
        with _inference_source_analysis(model):
            return cast("tuple[Any, bool]", super().apply(model))


class InferShapes(QonnxInferShapes):  # type: ignore[misc]
    """QONNX shape inference with one initializer analysis per apply pass."""

    def apply(self, model: Any) -> tuple[Any, bool]:
        with _inference_source_analysis(model):
            return cast("tuple[Any, bool]", super().apply(model))


__all__ = ["InferDataTypes", "InferShapes"]
