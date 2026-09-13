# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The production selected-construction catalogue."""

from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.native import operation_choice_schema
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.selected import ConstructionRegistry

assert ActivationReplayDesign.selected_graph is not None
assert DotProductDesign.selected_graph is not None
_REPLAY = ActivationReplayDesign.selected_graph.construction
_MVAU = DotProductDesign.selected_graph.construction

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
