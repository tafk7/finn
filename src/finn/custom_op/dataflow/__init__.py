# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The logical FINN dataflow custom-operation domain.

QONNX resolves a node's ``domain`` to this module and its ``op_type`` to a name
in ``custom_op``.  Registration is therefore the one place the two naming
authorities meet, and it is deliberately a flat table rather than a scan: an
operation appears in a graph because someone put it here.
"""

from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.replay.op import ActivationReplayOp

custom_op = {
    "MvauDataflowOp": MvauDataflowOp,
    "ActivationReplayOp": ActivationReplayOp,
}

__all__ = ["ActivationReplayOp", "MvauDataflowOp", "custom_op"]
