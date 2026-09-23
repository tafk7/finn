# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Family-independent checks for model-aware adapters and immutable source Spaces.

The model factory returns an initialized adapter with a separate Space. Exploration
and effect planning operate on that frozen Space; saving and QONNX callbacks belong
to the model-facing adapter. The checks distinguish current-target choice reuse
from applying precomputed effects whose recorded inputs have changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from finn.kernels._engine import Decided, RequestError
from finn.kernels.space.declarations import Space
from finn.kernels.space.occurrence import ProjectionAssessment
from finn.dataflow.ops.base import DataflowOp
from finn.dataflow.ops.space import DataflowSpace, DataflowOpError
from finn.dataflow.ops.native import (
    proposed_choice_values,
    read_attributes,
    SCHEMA_VERSION_ATTRIBUTE,
)


@dataclass(frozen=True)
class DataflowOpConformanceCase:
    """One registered adapter and operation-specific immutable choice exploration."""

    model: Any
    node_name: str
    operation_type: type[DataflowOp]
    build: Any
    configure: Callable[[DataflowSpace], DataflowSpace]
    reload_path: Path
    mutate_problem: Callable[[Any], None] | None = None
    other_build: Any = None
    execution_context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DataflowOpConformanceResult:
    committed: DataflowSpace
    restored: DataflowSpace


class ConformanceFailure(AssertionError):
    """One named promise of the operation layer was not kept."""


def _require(condition: object, promise: str) -> None:
    if not condition:
        raise ConformanceFailure(promise)


def _node(model: Any, name: str) -> Any:
    nodes = [item for item in model.graph.node if item.name == name]
    _require(len(nodes) == 1, f"the graph holds exactly one node named {name!r}")
    return nodes[0]


def _adapter(model: Any, name: str) -> DataflowOp:
    before = model.model.SerializeToString(deterministic=True)
    operation = model.get_customop_wrapper(_node(model, name))
    _require(isinstance(operation, DataflowOp), "the model factory returns a DataflowOp")
    _require(not isinstance(operation, Space), "the graph adapter is distinct from Space")
    _require(
        isinstance(operation.space, DataflowSpace) and operation.space is not operation,
        "the model factory initializes a separate source Space",
    )
    _require(
        type(operation.space) is operation.space_type, "the adapter owns its declared Space type"
    )
    _require(
        not hasattr(operation.space, "_model") and not hasattr(operation.space, "onnx_node"),
        "the immutable Space has no live model pointer or CustomOp node identity",
    )
    _require(
        model.model.SerializeToString(deterministic=True) == before,
        "model-aware creation does not mutate the graph",
    )
    return cast(DataflowOp, operation)


def assert_dataflow_op_conforms(case: DataflowOpConformanceCase) -> DataflowOpConformanceResult:
    """Exercise factory construction, exploration, checked save and QONNX callbacks."""
    model = case.model
    adapter = _adapter(model, case.node_name)
    _require(
        isinstance(adapter, case.operation_type),
        f"the model factory returns a {case.operation_type.__name__}",
    )
    _verification_is_reachable_without_build_context(case)
    before = model.model.SerializeToString(deterministic=True)
    bound = adapter.set_context(build=case.build).space
    _require(
        isinstance(bound.dataflow, ProjectionAssessment),
        "the source Space exposes an assessment, not only a reduced answer",
    )
    frozen_source = bound.source
    frozen_node = bound.node_snapshot().SerializeToString(deterministic=True)
    configured = case.configure(bound)
    _require(type(configured) is type(bound), "exploration returns the same Space class")
    _require(
        configured.problem_snapshot is bound.problem_snapshot,
        "exploration shares immutable facts instead of rereading the graph",
    )
    _require(adapter.space is bound, "exploration leaves the adapter Space pointer unchanged")
    _require(
        bound.source == frozen_source
        and bound.node_snapshot().SerializeToString(deterministic=True) == frozen_node,
        "the old Space remains a description of its frozen source",
    )
    _require(
        model.model.SerializeToString(deterministic=True) == before,
        "context setup and immutable exploration leave graph bytes unchanged",
    )
    _require(isinstance(configured.network, Decided), "the configured Space resolves its Network")

    _effects_write_nothing(model, configured)
    _a_current_build_is_revalidated(configured, case)
    _a_new_factory_space_observes_current_facts(case, bound)

    committed = adapter.save_space(configured)
    _require(isinstance(committed, DataflowSpace), "save_space returns a frozen source Space")
    _require(adapter.space is committed, "successful save updates the adapter Space pointer")
    _require(
        bound.source == frozen_source
        and bound.node_snapshot().SerializeToString(deterministic=True) == frozen_node,
        "saving leaves the old Space and its node snapshot unchanged",
    )
    _require(
        SCHEMA_VERSION_ATTRIBUTE in read_attributes(_node(model, case.node_name)),
        "saving writes the native interpretation schema",
    )
    _the_saved_space_describes_current_node(case, committed)
    restored = _survives_a_save_and_reload(case, committed)
    _stale_effects_change_nothing(case)
    _executes_with_an_attached_model(case)
    return DataflowOpConformanceResult(committed, restored)


def _a_new_factory_space_observes_current_facts(
    case: DataflowOpConformanceCase, bound: DataflowSpace
) -> None:
    if case.mutate_problem is None:
        return
    from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found] # noqa: PLC0415

    edited = ModelWrapper(case.model.model, make_deepcopy=True)
    previous = _adapter(edited, case.node_name).set_context(build=case.build).space
    frozen = previous.source
    case.mutate_problem(edited)
    fresh = _adapter(edited, case.node_name).set_context(build=case.build).space
    _require(
        previous.source == frozen, "an old Space still holds its frozen source after a graph edit"
    )
    _require(
        fresh.source != bound.source
        or fresh.node_snapshot().SerializeToString(deterministic=True)
        != bound.node_snapshot().SerializeToString(deterministic=True),
        "new model-aware construction observes current facts",
    )


def _the_saved_space_describes_current_node(
    case: DataflowOpConformanceCase, committed: DataflowSpace
) -> None:
    node = _node(case.model, case.node_name)
    _require(
        committed.node_snapshot().SerializeToString(deterministic=True)
        == node.SerializeToString(deterministic=True),
        "the returned Space snapshots the node produced by saving",
    )
    current = _adapter(case.model, case.node_name).set_context(build=case.build).space
    _require(
        current.source == committed.source,
        "saved and newly created Spaces read the same current source",
    )
    _require(
        dict(current.recorded()) == dict(committed.recorded()),
        "saved and newly created Spaces decode the same choices",
    )


def _verification_is_reachable_without_build_context(case: DataflowOpConformanceCase) -> None:
    from finn.analysis.verify_custom_nodes import verify_nodes  # noqa: PLC0415

    before = case.model.model.SerializeToString(deterministic=True)
    report: Mapping[str, Any] = verify_nodes(case.model)
    for messages in report.values():
        for message in list(messages):
            _require(
                "ERROR" not in str(message).upper(),
                f"verify_nodes reports a clean source, got {message!r}",
            )
    operation = _adapter(case.model, case.node_name)
    _require(
        isinstance(operation.verify_node(), list), "verify_node returns QONNX's list of messages"
    )
    _require(
        case.model.model.SerializeToString(deterministic=True) == before,
        "verification is observational and needs no build context",
    )


def _effects_write_nothing(model: Any, configured: DataflowSpace) -> None:
    before = model.model.SerializeToString(deterministic=True)
    effects = configured.graph_effects()
    _require(effects is not None, "graph_effects produces a detached plan")
    _require(
        model.model.SerializeToString(deterministic=True) == before,
        "planning effects does not mutate the graph",
    )


def _a_current_build_is_revalidated(
    configured: DataflowSpace, case: DataflowOpConformanceCase
) -> None:
    if case.other_build is None:
        return
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    candidate = ModelWrapper(case.model.model, make_deepcopy=True)
    before = candidate.model.SerializeToString(deterministic=True)
    operation = _adapter(candidate, case.node_name)
    original = operation.space
    try:
        operation.set_context(build=case.other_build)
    except (DataflowOpError, RequestError):
        _require(
            candidate.model.SerializeToString(deterministic=True) == before,
            "a refused context update leaves graph bytes unchanged",
        )
        _require(operation.space is original, "a failed context update preserves the Space pointer")
        return
    original = operation.space
    expected_build = dict(original._frozen_build_values())
    expected_valid = True
    try:
        expected = original.commit_choices(proposed_choice_values(original, configured))
        expected.graph_effects()
    except (DataflowOpError, RequestError):
        expected_valid = False
    try:
        saved = operation.save_space(configured)
    except (DataflowOpError, RequestError):
        _require(not expected_valid, "choices valid at the current target must remain saveable")
        _require(
            candidate.model.SerializeToString(deterministic=True) == before,
            "a refused current-target save leaves graph bytes unchanged",
        )
        _require(
            operation.space is original, "a failed context/save leaves the Space pointer unchanged"
        )
        return
    _require(expected_valid, "choices rejected by current-target obligations must not be saved")
    _require(
        dict(saved._frozen_build_values()) == expected_build,
        "saving uses current target build facts, never proposal build facts",
    )


def _survives_a_save_and_reload(
    case: DataflowOpConformanceCase, committed: DataflowSpace
) -> DataflowSpace:
    case.model.save(str(case.reload_path))
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    reloaded_model = ModelWrapper(str(case.reload_path))
    restored = _adapter(reloaded_model, case.node_name).set_context(build=case.build).space
    recorded, restored_recorded = dict(committed.recorded()), dict(restored.recorded())
    _require(restored_recorded == recorded, "every selected value survives native save/reload")
    for name, value in recorded.items():
        _require(
            type(restored_recorded[name]) is type(value),
            f"{name} returns as {type(value).__name__}, not merely its encoded spelling",
        )
    before, after = committed.network, restored.network
    if not isinstance(before, Decided) or not isinstance(after, Decided):
        raise ConformanceFailure("both Spaces resolve Networks")
    _require(after.value == before.value, "the reloaded Space resolves the same Network")
    return restored


def _stale_effects_change_nothing(case: DataflowOpConformanceCase) -> None:
    if case.mutate_problem is None:
        return
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from finn.dataflow.ops.model_effects import validate_model_read_set  # noqa: PLC0415
    from finn.dataflow.ops.persistence import apply_graph_effects  # noqa: PLC0415

    current = _adapter(case.model, case.node_name).set_context(build=case.build).space
    frozen = current.source
    effects = current.graph_effects()
    stale_model = ModelWrapper(case.model.model, make_deepcopy=True)
    case.mutate_problem(stale_model)
    try:
        validate_model_read_set(stale_model, effects.read_set)
    except DataflowOpError:
        pass
    else:
        raise ConformanceFailure("the deferred plan records the source facts that changed")
    before = stale_model.model.SerializeToString(deterministic=True)
    try:
        apply_graph_effects(stale_model, effects)
    except DataflowOpError:
        pass
    else:
        raise ConformanceFailure("applying stale precomputed effects must refuse")
    _require(
        stale_model.model.SerializeToString(deterministic=True) == before,
        "refusing stale effects leaves the graph byte-identical",
    )
    _require(current.source == frozen, "failed effect application does not change an old Space")
    report = _adapter(stale_model, case.node_name).verify_node()
    _require(
        not any("ERROR" in str(item).upper() for item in report),
        "source verification remains independent of a stale deferred plan",
    )


def _executes_with_an_attached_model(case: DataflowOpConformanceCase) -> None:
    if case.execution_context is None:
        return
    operation = _adapter(case.model, case.node_name)
    context = dict(case.execution_context)
    output = operation.onnx_node.output[0]
    context.pop(output, None)
    operation.execute_node(context, case.model.graph)
    _require(context.get(output) is not None, "QONNX execution writes the output tensor")


__all__ = [
    "ConformanceFailure",
    "DataflowOpConformanceCase",
    "DataflowOpConformanceResult",
    "assert_dataflow_op_conforms",
]
