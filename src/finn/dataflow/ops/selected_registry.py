# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The production selected-construction catalogue."""

from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.mvau.selected import MVAU_SELECTED_CONSTRUCTION
from finn.dataflow.ops.native import operation_choice_schema
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.replay.selected import REPLAY_SELECTED_CONSTRUCTION
from finn.dataflow.ops.selected import ConstructionRegistry

_REPLAY = REPLAY_SELECTED_CONSTRUCTION
_MVAU = MVAU_SELECTED_CONSTRUCTION

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
