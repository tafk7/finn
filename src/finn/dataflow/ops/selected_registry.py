# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The production selected-construction catalogue."""

from typing import cast

from finn.dataflow.ops.mvau.kernels.dot_product import DotProductKernel
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.native import operation_choice_schema
from finn.dataflow.ops.replay.kernel import ActivationReplayKernel
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.selected import ConstructionRegistry, SelectedConstruction

assert ActivationReplayKernel.selected_construction is not None
assert DotProductKernel.selected_construction is not None
_REPLAY = cast(SelectedConstruction[object, object], ActivationReplayKernel.selected_construction)
_MVAU = cast(SelectedConstruction[object, object], DotProductKernel.selected_construction)

DEFAULT_SELECTED_CONSTRUCTIONS = ConstructionRegistry(
    {
        (_REPLAY.family, _REPLAY.version): _REPLAY,
        (_MVAU.family, _MVAU.version): _MVAU,
    },
    {
        (_REPLAY.family, _REPLAY.version): operation_choice_schema(ActivationReplayOp),
        (_MVAU.family, _MVAU.version): operation_choice_schema(MvauDataflowOp),
    },
)

__all__ = ["DEFAULT_SELECTED_CONSTRUCTIONS"]
