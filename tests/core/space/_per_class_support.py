# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-class record (``_running.per_class.PerClass``) against today's evaluator, on one
compiled model: shared by the engine's test of the record and the chain's
(``test_per_class``, ``tests/kernel_ops/test_per_class_chain.py``)."""

from __future__ import annotations

from typing import cast

from finn.core.space import Space
from finn.core.space._structure import Structure
from finn.core.space.compiler import Model


def same_as_todays(model: Model[Space]) -> list[str]:
    """Where ``model``'s per-class record answers otherwise than its evaluator: the inferred
    reads of every node, the flow order, every choice's and fact's presence readers."""
    record, today = model.per_class, model.evaluator
    linked = model.linked
    differences: list[str] = []
    indices = tuple(cast(Structure, linked).all_nodes())
    for index in indices:
        key = linked.header(index).key
        for part in ("methods", "unresolved", "existence", "cases"):
            mine = getattr(record.reads, part).get(index, "none")
            todays = getattr(today.reads, part).get(index, "none")
            if mine != todays:
                differences.append(f"{key}: {part} {mine} against {todays}")
    if record.flow() != today.flow():
        differences.append(f"flow {record.flow()} against {today.flow()}")
    for source in (*linked.parameters, *linked.decisions):
        if record.presence_readers(source) != today.presence_readers(source):
            differences.append(f"{linked.header(source).key}: presence readers")
    return differences


__all__ = ["same_as_todays"]
