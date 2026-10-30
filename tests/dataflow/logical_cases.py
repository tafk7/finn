# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tiny logical graph for immutable query and descriptor case checks."""

from __future__ import annotations

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.logical import Edge, LogicalPlan, Node, Port, Source, Transfer
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    BeatSequence,
    LevelEnd,
    Loop,
    Traversal,
    vector_major,
)

INT8 = ScalarEncoding(resolve_qonnx_datatype_name("INT8"))


def tiny() -> LogicalPlan:
    tensor = Tensor((2, 6), INT8)
    row = BeatSequence(vector_major(tensor.shape, 2), markers=(LevelEnd(3),))
    # Two-lane column words visit source word indices [0,3,1,4,2,5].
    column = BeatSequence(
        Traversal(tensor.shape, (Loop(3, 2), Loop(2, 6)), (Loop(2, 1),)), markers=(LevelEnd(2),)
    )
    ports = (
        Port("out", "producer", "write", "data", tensor, row),
        Port("in", "consumer", "read", "data", tensor, column),
        Port("branch", "other", "read", "data", tensor, row),
    )
    return LogicalPlan(
        "plan",
        (Node("producer"), Node("consumer"), Node("other")),
        ports,
        (Edge("edge", "out", "in"), Edge("fork", "out", "branch")),
        (Source("data", tensor),),
        (Transfer("publish", "producer", (), ("out",)),),
    )
