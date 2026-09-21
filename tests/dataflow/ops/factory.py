# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test construction through QONNX's actual model-aware CustomOp factory."""

from typing import Any
from unittest.mock import patch
from qonnx.custom_op import registry

from finn.custom_op.dataflow import custom_op
from finn.dataflow.ops.base import DataflowOp
from finn.dataflow.ops.space import DataflowSpace


def make_op(
    model: Any,
    node: Any = None,
    *,
    space_type: type[DataflowSpace] | None = None,
    build: Any = None,
    graph_context: Any = None,
) -> DataflowOp:
    node = model.graph.node[0] if node is None else node
    if space_type is None:
        operation = model.get_customop_wrapper(node)
    else:
        definition = space_type

        class FixtureAdapter(DataflowOp):
            space_type = definition

        domain_registry = {
            **registry._OP_REGISTRY.get(node.domain, {}),
            node.op_type: {1: FixtureAdapter},
        }
        with (
            patch.dict(custom_op, {node.op_type: FixtureAdapter}),
            patch.dict(registry._OP_REGISTRY, {node.domain: domain_registry}),
        ):
            operation = model.get_customop_wrapper(node)
    assert isinstance(operation, DataflowOp)
    if build is not None or graph_context is not None:
        operation.set_context(build=build, graph_context=graph_context)
    return operation


def make_space(model: Any, node: Any = None, **kwargs: Any) -> DataflowSpace:
    return make_op(model, node, **kwargs).space
