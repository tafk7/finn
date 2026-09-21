# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One lifecycle check every DataflowOp must pass, whoever wrote it.

The layer's claim is that it is not MVAU-shaped: a contributor adds an
operation, declares its source schema and its Kernels, and gets binding,
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
    Returns a bound occurrence over the *post-commit* graph after revalidating
    proposed choices against current target facts.

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
from finn.dataflow.space.occurrence import ProjectionAssessment
from finn.dataflow.ops.base import DataflowOp, DataflowOpError
from finn.dataflow.ops.native import read_attributes, SCHEMA_VERSION_ATTRIBUTE


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
    _require(
        bound.onnx_node.SerializeToString(deterministic=True),
        "the occurrence froze the bytes it read",
    )

    assessment = bound.dataflow
    _require(
        isinstance(assessment, ProjectionAssessment),
        "the operation projects an assessment, not a bare answer",
    )

    configured = case.configure(bound)
    _require(type(configured) is type(bound), "specializing produces the same class")
    _require(
        configured.problem_snapshot is bound.problem_snapshot,
        "specializing does not re-read the graph; the binding is the same frozen one",
    )
    _require(
        isinstance(configured.network, Decided),
        "a fully specialized occurrence resolves its Network",
    )

    _effects_write_nothing(model, configured)
    _a_current_build_is_revalidated(model, configured, case)
    _a_fresh_rebind_reads_the_graph_again(case, bound)

    committed = configured.commit(model, case.build)
    _require(committed.is_bound, "commit returns a bound occurrence")
    _require(
        SCHEMA_VERSION_ATTRIBUTE in read_attributes(_node(model, case.node_name)),
        "the commit wrote native attributes to the node",
    )
    _the_committed_occurrence_describes_the_post_commit_node(case, model, committed)

    restored = _survives_a_save_and_reload(case, committed)
    _staleness_is_reported_and_a_stale_plan_changes_nothing(case, model)
    _executes_with_an_attached_model(case)

    return DataflowOpConformanceResult(committed, restored)


def _a_fresh_rebind_reads_the_graph_again(case: DataflowOpConformanceCase, bound: Any) -> None:
    """A frozen source stays frozen; ``rebind`` is how a caller gets a new one.

    Both halves matter and only together. An occurrence that quietly re-read the
    graph would make every answer depend on when it was asked; one that had no
    way to re-read it would be stuck describing a node that no longer exists.
    So the bound occurrence keeps its snapshot across an edit, and the
    explicitly rebound successor observes the edit.

    Run *before* the commit, deliberately: after one, native choices written against
    the old problem is refused on the way in — which is a different promise, and
    is checked separately.
    """

    if case.mutate_problem is None:
        return
    from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found] # noqa: PLC0415

    edited = ModelWrapper(case.model.model, make_deepcopy=True)
    case.mutate_problem(edited)

    frozen = bound.source
    rebound = bound.rebind(edited, case.build)

    _require(
        bound.source == frozen,
        "the original occurrence still holds the source it froze",
    )
    _require(
        rebound.problem_fingerprint != bound.problem_fingerprint,
        "an explicit rebind over a changed graph observes a changed problem",
    )
    _require(
        rebound.onnx_node.SerializeToString(deterministic=True)
        != bound.onnx_node.SerializeToString(deterministic=True)
        or rebound.source != bound.source,
        "the rebound occurrence read the graph again rather than reusing the snapshot",
    )


def _the_committed_occurrence_describes_the_post_commit_node(
    case: DataflowOpConformanceCase, model: Any, committed: Any
) -> None:
    """``commit`` continues the lifecycle; it does not end it at the mutation.

    Returning the live node, or an occurrence still bound to the pre-commit
    graph, would leave every later question being answered from the graph as it
    was *before* the caller's own commit — and that failure looks like a
    successful commit, which is worse than an error.
    """

    node = _node(model, case.node_name)
    _require(
        committed.onnx_node.SerializeToString(deterministic=True)
        == node.SerializeToString(deterministic=True),
        "the returned occurrence froze the bytes of the node the commit produced",
    )
    _require(
        not committed.is_stale((model, case.build)),
        "the returned occurrence is not stale against the model it just committed to",
    )


def _recorded_refuses_on_an_unbound_wrapper(unbound: Any) -> None:
    try:
        unbound.recorded()
    except DataflowOpError as error:
        _require(
            "read_attributes" in str(error),
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


def _a_current_build_is_revalidated(
    model: Any, configured: Any, case: DataflowOpConformanceCase
) -> None:
    """A proposed choice point cannot overwrite current target build facts."""

    if case.other_build is None:
        return
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    candidate = ModelWrapper(model.model, make_deepcopy=True)
    before = candidate.model.SerializeToString(deterministic=True)
    try:
        current = _unbound(candidate, case.node_name).save_space(
            candidate, configured, case.other_build
        )
    except DataflowOpError:
        _require(
            candidate.model.SerializeToString(deterministic=True) == before,
            "a refused commit left the graph byte-identical",
        )
        return
    _require(
        dict(current._frozen_build_values()) == dict(configured._build_values(case.other_build)),
        "a fresh save uses current build facts rather than proposal facts",
    )


def _survives_a_save_and_reload(case: DataflowOpConformanceCase, committed: Any) -> Any:
    case.model.save(str(case.reload_path))
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

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
    effects = committed.graph_effects()
    stale_model = ModelWrapper(model.model, make_deepcopy=True)
    case.mutate_problem(stale_model)
    _require(
        committed.is_stale((stale_model, case.build)),
        "an occurrence whose graph changed under it reports itself stale",
    )

    # Deferred effects carry reads; unlike a fresh save, they cannot reuse stale
    # computed writes merely because the choice values might still be legal.
    from finn.dataflow.ops.persistence import apply_graph_effects  # noqa: PLC0415

    before = stale_model.model.SerializeToString(deterministic=True)
    try:
        apply_graph_effects(stale_model, effects)
    except DataflowOpError as error:
        _require(
            "different problem" in str(error),
            f"the refusal says the problem changed, got {error}",
        )
    else:
        raise ConformanceFailure("applying a stale precomputed effect plan is refused")
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
