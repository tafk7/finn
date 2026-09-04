# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One lifecycle check every DataflowOp must pass, whoever wrote it.

The layer's claim is that it is not MVAU-shaped: a contributor adds an
operation, declares its source schema and its Designs, and gets binding,
projection, persistence, staleness, verification and execution without writing
any of them.  A claim like that is worth exactly as much as the check that a
*third* operation would pass, so the check lives here rather than in the tests
of the two operations that exist.

Every step below was a defect at some point in this stack's history, and each
is named so that a regression reports which promise broke rather than which
assertion failed:

``bind``
    An unbound wrapper is what QONNX constructs; binding is what attaches it to
    a graph and freezes the source it read.

``project``
    A ``ProjectionAssessment``, not a bare answer -- readiness, constraints and
    output are separable, and a caller that got only the reduction could not
    tell "unresolved" from "refused".

``graph_effects``
    Produces a value and writes nothing.  The whole-model bytes are compared
    before and after.

``commit``
    Returns a bound occurrence over the *post-commit* graph, and refuses a build
    that differs from the frozen one before mutating anything.

``save`` / ``reload``
    Decoded values, not just canonical bytes: an Enum that came back as its
    string would compare equal in JSON and behave differently in Python.

``verify_node``
    Through FINN's real path -- an ordinary model-attached, design-space-unbound
    wrapper, with **no build context**.  A harness that verified a bound
    occurrence would exercise a path FINN never takes.

``recorded`` (unbound)
    Refuses, and names the function that reads a raw node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finn.dataflow._engine import Decided
from finn.dataflow.model.occurrence import ProjectionAssessment
from finn.dataflow.ops.base import DataflowOp, DataflowOpError
from finn.dataflow.ops.state import decode_dataflow_state


@dataclass(frozen=True)
class DataflowOpConformanceCase:
    """One operation, one graph, and the two callables only its author can write."""

    model: Any
    node_name: str
    operation_type: type[DataflowOp]
    build: Any
    #: A bound occurrence in, a *fully specialized* bound occurrence out.  The
    #: only operation-specific step: which Decisions exist is the operation's
    #: business, and enumerating them here would make this harness MVAU-shaped.
    configure: Callable[[Any], Any]
    reload_path: Path
    #: Change the graph so the recorded problem no longer describes it.  A
    #: staleness claim with no way to make a node stale is untested.
    mutate_problem: Callable[[Any], None] | None = None
    #: A build whose facts differ, for the commit-equivalence refusal.
    other_build: Any = None
    #: Inputs for ``execute_node``, keyed by tensor name.  The output tensor's
    #: entry may be omitted; execution is expected to write it.
    execution_context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DataflowOpConformanceResult:
    """What the lifecycle produced, for a caller that wants to assert more."""

    committed: Any
    restored: Any


class ConformanceFailure(AssertionError):
    """One named promise of the operation layer was not kept."""


def _require(condition: object, promise: str) -> None:
    if not condition:
        raise ConformanceFailure(promise)


def _node(model: Any, name: str) -> Any:
    nodes = [item for item in model.graph.node if item.name == name]
    _require(len(nodes) == 1, f"the graph holds exactly one node named {name!r}")
    return nodes[0]


def _unbound(model: Any, name: str) -> Any:
    operation = model.get_customop_wrapper(_node(model, name))
    _require(
        isinstance(operation, DataflowOp),
        "QONNX's registry returns a DataflowOp for this op_type",
    )
    return operation


def assert_dataflow_op_conforms(
    case: DataflowOpConformanceCase,
) -> DataflowOpConformanceResult:
    """Run one operation through the whole lifecycle and check each promise."""

    model = case.model
    unbound = _unbound(model, case.node_name)
    _require(
        isinstance(unbound, case.operation_type),
        f"the wrapper is a {case.operation_type.__name__}",
    )
    _require(not unbound.is_bound, "a QONNX-constructed wrapper is not bound")

    _verification_is_reachable_without_a_design_space(case)
    _recorded_refuses_on_an_unbound_wrapper(unbound)

    bound = unbound.bind(model, case.build)
    _require(type(bound) is type(unbound), "binding preserves the operation class")
    _require(bound.is_bound, "a bound occurrence says so")
    _require(bound.binding.node_bytes, "the occurrence froze the bytes it read")

    assessment = bound.dataflow
    _require(
        isinstance(assessment, ProjectionAssessment),
        "the operation projects an assessment, not a bare answer",
    )

    configured = case.configure(bound)
    _require(type(configured) is type(bound), "specializing produces the same class")
    _require(
        configured.binding is bound.binding,
        "specializing does not re-read the graph; the binding is the same frozen one",
    )
    _require(
        isinstance(configured.network, Decided),
        "a fully specialized occurrence resolves its Network",
    )

    _effects_write_nothing(model, configured)
    _a_mismatched_build_cannot_change_the_graph(model, configured, case)

    committed = configured.commit(model, case.build)
    _require(committed.is_bound, "commit returns a bound occurrence")
    _require(
        decode_dataflow_state(_node(model, case.node_name)) is not None,
        "the commit wrote a state document to the node",
    )

    restored = _survives_a_save_and_reload(case, committed)
    _staleness_is_reported_and_a_stale_plan_changes_nothing(case, model)
    _executes_with_an_attached_model(case)

    return DataflowOpConformanceResult(committed, restored)


def _recorded_refuses_on_an_unbound_wrapper(unbound: Any) -> None:
    try:
        unbound.recorded()
    except DataflowOpError as error:
        _require(
            "decode_dataflow_state" in str(error),
            "the refusal names the function that reads a raw node",
        )
        return
    raise ConformanceFailure("recorded() refuses on an unbound wrapper")


def _verification_is_reachable_without_a_design_space(
    case: DataflowOpConformanceCase,
) -> None:
    """The path ``verify_nodes(model)`` takes, and no other.

    Two separable claims, both of which were once false: verification runs on an
    ordinary wrapper with no build context, and it does *not* replay whatever
    choices happen to be recorded on the node.  The second is why this runs
    before the commit as well as being re-checked after it.
    """

    from finn.analysis.verify_custom_nodes import verify_nodes  # noqa: PLC0415

    report: Mapping[str, Any] = verify_nodes(case.model)
    for messages in report.values():
        for message in list(messages):
            _require(
                "ERROR" not in str(message).upper(),
                f"verify_nodes reports a clean node, got {message!r}",
            )

    operation = _unbound(case.model, case.node_name)
    _require(
        not operation.is_bound,
        "verification runs on a wrapper that is not bound to a design space",
    )
    _require(
        isinstance(operation.verify_node(), list),
        "verify_node returns QONNX's list of messages",
    )


def _effects_write_nothing(model: Any, configured: Any) -> None:
    before = model.model.SerializeToString(deterministic=True)
    effects = configured.graph_effects()
    _require(effects is not None, "graph_effects produces a value")
    _require(
        model.model.SerializeToString(deterministic=True) == before,
        "planning the effects wrote nothing to the graph",
    )


def _a_mismatched_build_cannot_change_the_graph(
    model: Any, configured: Any, case: DataflowOpConformanceCase
) -> None:
    """A build that differs from the frozen one is refused *before* any mutation."""

    if case.other_build is None:
        return
    before = model.model.SerializeToString(deterministic=True)
    try:
        configured.commit(model, case.other_build)
    except DataflowOpError:
        _require(
            model.model.SerializeToString(deterministic=True) == before,
            "a refused commit left the graph byte-identical",
        )
        return
    raise ConformanceFailure("committing under a different build is refused")


def _survives_a_save_and_reload(case: DataflowOpConformanceCase, committed: Any) -> Any:
    case.model.save(str(case.reload_path))
    from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found] # noqa: PLC0415

    reloaded_model = ModelWrapper(str(case.reload_path))
    restored = _unbound(reloaded_model, case.node_name).bind(reloaded_model, case.build)

    recorded = dict(committed.recorded())
    restored_recorded = dict(restored.recorded())
    _require(restored_recorded == recorded, "every recorded choice survived the round trip")
    for name, value in recorded.items():
        _require(
            type(restored_recorded[name]) is type(value),
            f"{name} came back as {type(value).__name__}, not as its canonical spelling",
        )
    before = committed.network
    after = restored.network
    _require(isinstance(before, Decided) and isinstance(after, Decided), "both resolve a Network")
    _require(after.value == before.value, "the reloaded occurrence resolves the same Network")
    return restored


def _staleness_is_reported_and_a_stale_plan_changes_nothing(
    case: DataflowOpConformanceCase, model: Any
) -> None:
    if case.mutate_problem is None:
        return
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    committed = _unbound(model, case.node_name).bind(model, case.build)
    stale_model = ModelWrapper(model.model, make_deepcopy=True)
    case.mutate_problem(stale_model)
    _require(
        committed.is_stale((stale_model, case.build)),
        "an occurrence whose graph changed under it reports itself stale",
    )

    # And a document written against the old problem is refused on the way in,
    # with the graph untouched -- a stale plan does not half-apply.
    before = stale_model.model.SerializeToString(deterministic=True)
    try:
        _unbound(stale_model, case.node_name).bind(stale_model, case.build)
    except DataflowOpError as error:
        _require(
            "different problem" in str(error),
            f"the refusal says the problem changed, got {error}",
        )
    else:
        raise ConformanceFailure("binding refuses a document made against another problem")
    _require(
        stale_model.model.SerializeToString(deterministic=True) == before,
        "the refused binding left the graph byte-identical",
    )

    # Verification still passes: whether stored choices still fit is not what
    # verification asks.
    report = _unbound(stale_model, case.node_name).verify_node()
    _require(
        not any("ERROR" in str(item).upper() for item in report),
        "verification does not replay persisted choices",
    )


def _executes_with_an_attached_model(case: DataflowOpConformanceCase) -> None:
    if case.execution_context is None:
        return
    operation = _unbound(case.model, case.node_name)
    context = dict(case.execution_context)
    operation.execute_node(context, case.model.graph)
    output = operation.onnx_node.output[0]
    _require(context.get(output) is not None, "execution wrote the output tensor")


__all__ = [
    "ConformanceFailure",
    "DataflowOpConformanceCase",
    "DataflowOpConformanceResult",
    "assert_dataflow_op_conforms",
]
