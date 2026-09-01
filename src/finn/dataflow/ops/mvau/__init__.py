# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public MVAU dataflow-operation façade."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.dataflow.ops.mvau.op import MVAUDataflowBuildContext, MvauDataflowOp

_LAZY_EXPORTS = {
    name: ("finn.dataflow.ops.mvau.op", name)
    for name in ("MVAUDataflowBuildContext", "MvauDataflowOp")
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = ["MVAUDataflowBuildContext", "MvauDataflowOp"]
