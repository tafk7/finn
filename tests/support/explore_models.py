# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel-path model open to exploration, and the choices it states."""

from __future__ import annotations

import copy

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.graph.annotations.channel import CHANNEL, channel_choices
from finn.core.graph.annotations.platform import write_target
from finn.custom_op.kernels.base import kernel_op
from finn.kernels.explore import KernelExplorer
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors, ToKernelOps, explore_kernel_choices
from support.models import TARGET, chain_source


def kernel_model(target: Target = TARGET) -> ModelWrapper:
    model = chain_source().transform(InferShapes()).transform(ToKernelOps(TARGET))
    write_target(model, target)
    return model.transform(InferKernelTensors())


def choices(model: ModelWrapper) -> dict[str, dict[str, object]]:
    """Every choice the model states, by graph name: each KernelOp node's (empty when it
    holds none), then each tensor's channel choices."""
    found = {
        node.name: kernel_op(model, node).choices()
        for node in model.graph.node
        if node.domain == "finn.custom_op.kernels"
    }
    for tensor in model.tensors_stating(CHANNEL):
        found[tensor] = channel_choices(model, tensor)
    return found


def explored(model: ModelWrapper, *strategies: KernelExplorer) -> ModelWrapper:
    """A copy of ``model`` explored by ``strategies``, in order, on one seam
    (``explore_kernel_choices``): the model itself is not changed."""
    found = ModelWrapper(copy.deepcopy(model.model))
    explore_kernel_choices(found, strategies)
    return found
