# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graphs a KernelOp's spec states (``finn.harness.reference.OpSpec``): ONNX source
graphs, before any conversion, their inputs shaped and annotated."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.core.containers import numpy_type

GENERAL = "qonnx.custom_op.general"

Stated = tuple[list[int] | None, str | None]
"""A graph input's shape (None: unstated) and annotation (None: unannotated)."""


def source(
    nodes: list[NodeProto],
    inputs: Mapping[str, Stated],
    stored: Mapping[str, tuple[Any, str | None]],
    container: int = TensorProto.FLOAT,
) -> ModelWrapper:
    """A source graph of ``nodes``: its ``inputs`` (shape, datatype) and ``stored``
    initializers (values, datatype); the last node's output the graph's. Every tensor
    is held in ``container`` (``finn.core.containers``), the export's float32 unless
    stated, as graph preparation's P6 leaves a widened region."""
    infos = [
        helper.make_tensor_value_info(name, container, dims) for name, (dims, _) in inputs.items()
    ]
    y = helper.make_tensor_value_info(nodes[-1].output[0], container, None)
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "spec", infos, [y]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(GENERAL, 1)],
        )
    )
    for name, (values, _) in stored.items():
        model.set_initializer(name, np.asarray(values, dtype=numpy_type(container)))
    for name, (_, dtype) in {**inputs, **stored}.items():
        if dtype is not None:
            model.set_tensor_datatype(name, DataType[dtype])
    return model


__all__ = ["GENERAL", "Stated", "source"]
