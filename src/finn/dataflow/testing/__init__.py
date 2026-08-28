# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable conformance checks for contributor-authored dataflow operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found]

from finn.dataflow.op import (
    AssignmentMapping,
    DataflowBuildConfigView,
    DataflowOp,
    DataflowOpError,
)
from finn.dataflow.resolution import NetworkRef, RegionRef, ResolvedDataflowOp


@dataclass(frozen=True)
class DataflowOpConformanceCase:
    """Inputs for the shared node lifecycle conformance check."""

    model: ModelWrapper
    node_name: str
    operation_type: type[DataflowOp]
    config: DataflowBuildConfigView
    complete_assignments: AssignmentMapping
    rejected_assignments: AssignmentMapping
    reload_path: Path
    stale_config: DataflowBuildConfigView | None = None
    mutate_graph_problem: Callable[[ModelWrapper], None] | None = None


@dataclass(frozen=True)
class DataflowOpConformanceResult:
    """Resolved values observed before and after persistence."""

    original: ResolvedDataflowOp
    restored: ResolvedDataflowOp


def _expects_dataflow_error(action: Callable[[], object], code: str | None = None) -> None:
    try:
        action()
    except DataflowOpError as exc:
        if code is not None:
            assert code in {finding.code for finding in exc.findings}
    else:
        raise AssertionError("operation unexpectedly succeeded")


def assert_dataflow_op_conforms(
    case: DataflowOpConformanceCase,
) -> DataflowOpConformanceResult:
    """Exercise the common model-aware projection and persistence lifecycle."""

    nodes = tuple(node for node in case.model.graph.node if node.name == case.node_name)
    assert len(nodes) == 1
    node = nodes[0]
    operation = case.model.get_customop_wrapper(node)
    assert isinstance(operation, case.operation_type)
    assert isinstance(operation, DataflowOp)
    assert operation._attached_model() is case.model

    bare = getCustomOp(node)
    assert isinstance(bare, DataflowOp)
    _expects_dataflow_error(
        lambda: bare.problem_instance(case.config),
        "dataflow-model-required",
    )

    model_before = case.model.model.SerializeToString(deterministic=True)
    first_problem = operation.problem_instance(case.config)
    second_problem = operation.problem_instance(case.config)
    assert first_problem == second_problem
    assert case.model.model.SerializeToString(deterministic=True) == model_before
    assert operation.read_assignments() == {}

    node_before = node.SerializeToString(deterministic=True)
    _expects_dataflow_error(
        lambda: operation.commit_dataflow_assignments(case.config, case.rejected_assignments)
    )
    assert node.SerializeToString(deterministic=True) == node_before

    assignment_items = tuple(case.complete_assignments.items())
    split = max(1, len(assignment_items) // 2) if assignment_items else 0
    if split:
        operation.commit_dataflow_assignments(case.config, dict(assignment_items[:split]))
    partial = operation.hydrate_dataflow_point(case.config)
    assert len(partial.assignments) == split
    case.model.save(case.reload_path)
    reloaded_model = ModelWrapper(str(case.reload_path))
    reloaded_node = next(node for node in reloaded_model.graph.node if node.name == case.node_name)
    reloaded = reloaded_model.get_customop_wrapper(reloaded_node)
    assert isinstance(reloaded, case.operation_type)
    assert reloaded.hydrate_dataflow_point(case.config).assignments == partial.assignments

    if split < len(assignment_items):
        reloaded.commit_dataflow_assignments(case.config, dict(assignment_items[split:]))
    original = reloaded.resolve_dataflow(case.config)
    assert isinstance(original.result, (RegionRef, NetworkRef))
    reloaded_model.save(case.reload_path)
    final_model = ModelWrapper(str(case.reload_path))
    final_node = next(node for node in final_model.graph.node if node.name == case.node_name)
    final_operation = final_model.get_customop_wrapper(final_node)
    assert isinstance(final_operation, case.operation_type)
    restored = final_operation.resolve_dataflow(case.config)
    assert restored.point.assignments == original.point.assignments
    assert restored.result == original.result
    assert restored.source_association == original.source_association
    assert restored.source_scope_id == original.source_scope_id

    if case.stale_config is not None:
        stale_config = case.stale_config
        _expects_dataflow_error(
            lambda: final_operation.hydrate_dataflow_point(stale_config),
            "dataflow-selection-problem-mismatch",
        )
    if case.mutate_graph_problem is not None:
        stale_model = ModelWrapper(final_model.model, make_deepcopy=True)
        case.mutate_graph_problem(stale_model)
        stale_node = next(node for node in stale_model.graph.node if node.name == case.node_name)
        stale_operation = stale_model.get_customop_wrapper(stale_node)
        assert isinstance(stale_operation, case.operation_type)
        _expects_dataflow_error(
            lambda: stale_operation.hydrate_dataflow_point(case.config),
            "dataflow-selection-problem-mismatch",
        )

    return DataflowOpConformanceResult(original, restored)


__all__ = [
    "DataflowOpConformanceCase",
    "DataflowOpConformanceResult",
    "assert_dataflow_op_conforms",
]
