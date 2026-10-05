# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A model's KernelOp choices as a configuration ``ApplyConfig`` applies.

FINN's ``extract_model_config_to_json`` reads every attribute through
``get_nodeattr``, which answers an absent choice with its default, so an open
choice would come back as a value. This export is sparse: each KernelOp node's
choices as its node holds them, absent ones absent.
"""

from __future__ import annotations

from typing import Any

from finn.custom_op.kernels.base import KernelOp
from finn.transformation.fpgadataflow.kernel_partitions import KERNEL_OPS_DOMAIN


def kernel_choices_config(model: Any) -> dict[str, dict[str, object]]:
    """Each KernelOp node's choices by node name, for ``ApplyConfig``; a node with none
    has no entry."""
    config: dict[str, dict[str, object]] = {}
    for node in model.graph.node:
        if node.domain != KERNEL_OPS_DOMAIN:
            continue
        op = model.get_customop_wrapper(node)
        if isinstance(op, KernelOp) and (choices := op.choices()):
            config[node.name] = choices
    return config


__all__ = ["kernel_choices_config"]
