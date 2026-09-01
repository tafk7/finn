# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 7 gate: a replaceable selection-policy seam over whole points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Decided, Engine, QualifiedPath
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY, FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.ops.mvau import (
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    NetworkRef,
)
from finn.dataflow.ops.mvau.op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.transformation.fpgadataflow.select_dataflow_design import (
    DataflowSelectionContext,
    DataflowSelectionPolicy,
    ExplicitAssignmentsPolicy,
    FirstFeasiblePolicy,
    SelectDataflowDesign,
)

PART = "xczu3eg-sbva484-1-e"
SCOPE = "selection_scope"
NODE = "logical_mvau"


@dataclass
class _BuildConfig:
    synth_clk_period_ns: float = 5.0
    fpga_part: str | None = PART

    def _resolve_fpga_part(self) -> str:
        if self.fpga_part is None:
            raise ValueError("no target part")
        return self.fpga_part


def _context() -> MVAUDataflowBuildContext:
    return MVAUDataflowBuildContext(_BuildConfig())


def _model(*, with_initializer: bool = True) -> ModelWrapper:
    # A 2x2 matrix keeps the PE/SIMD domains small enough that enumerating
    # every coherent point stays a unit-test-sized job.
    width = height = rows = 2
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weights"],
        ["output"],
        name=NODE,
        domain="finn.custom_op.dataflow",
        dataflow_scope_id=SCOPE,
        noActivation=1,
        binaryXnorMode=0,
        accDataType="INT16",
        ActVal=0,
    )
    graph = helper.make_graph(
        [node],
        "selection",
        [
            helper.make_tensor_value_info("activation", TensorProto.FLOAT, [rows, width]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [width, height]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, height])],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="selection-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT16"])
    if with_initializer:
        model.set_initializer("weights", np.ones((width, height), dtype=np.float32))
    return model


def _operation(model: ModelWrapper) -> MvauDataflowOp:
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    return operation


def _direct_assignments() -> dict[QualifiedPath, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    return {
        assembly.inventory.design_path: DotProductDesign.id,
        assembly.dot_product.pe.path: 2,
        assembly.dot_product.simd.path: 2,
        assembly.compute_pumping.path: False,
        assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
    }


def _cyclic_assignments() -> dict[QualifiedPath, object]:
    assembly = MVAU_DESIGN_INVENTORY
    return {
        **_direct_assignments(),
        assembly.input_supply.declaration.choice.path: FINN_RTL_MEMSTREAM_SUPPLY,
        assembly.input_supply.settings.ram_style.path: CyclicRamStyle.BRAM,
        assembly.input_supply.settings.pumped_memory.path: False,
    }


def _embedded_assignments() -> dict[QualifiedPath, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    return {
        assembly.inventory.design_path: BatchInterleavedDesign.id,
        assembly.batch_interleaved.pe.path: 2,
        assembly.batch_interleaved.simd.path: 2,
        assembly.batch_interleaved.interleave.path: 2,
        assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
    }


def _run(
    policy: DataflowSelectionPolicy, model: ModelWrapper | None = None
) -> tuple[ModelWrapper, SelectDataflowDesign]:
    transform = SelectDataflowDesign(
        policy,
        _context(),
    )
    target = model or _model()
    return target.transform(transform, cleanup=False), transform


# -- explicit policies -------------------------------------------------------


def test_an_explicit_policy_reproduces_the_direct_point() -> None:
    lowered, transform = _run(ExplicitAssignmentsPolicy({SCOPE: _direct_assignments()}))
    resolved = _operation(lowered).resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)
    assert resolved.result.source_association.parameter_topology is MVAUParameterTopology.DIRECT
    assert transform.report.scope(SCOPE).committed


def test_an_explicit_policy_reproduces_the_cyclic_point() -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _cyclic_assignments()}))
    resolved = _operation(lowered).resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)
    assert resolved.result.source_association.parameter_topology is MVAUParameterTopology.CYCLIC


def test_an_explicit_policy_can_select_the_semantic_only_design() -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _embedded_assignments()}))
    resolved = _operation(lowered).resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)
    assert resolved.result.source_association.design_id == BatchInterleavedDesign.id


def test_a_policy_is_keyed_by_stable_operation_scope() -> None:
    lowered, transform = _run(ExplicitAssignmentsPolicy({"another_scope": _direct_assignments()}))
    assert _operation(lowered).read_assignments() == {}
    assert {finding.code for finding in transform.report.scope(SCOPE).findings} == {
        "dataflow-selection-no-point"
    }


def test_invalid_policy_output_leaves_the_node_byte_identical() -> None:
    model = _model()
    before = model.graph.node[0].SerializeToString(deterministic=True)
    lowered, transform = _run(
        ExplicitAssignmentsPolicy({SCOPE: {MVAUDataflowOpPaths.DESIGN: "absent-design"}}),
        model,
    )
    assert lowered.graph.node[0].SerializeToString(deterministic=True) == before
    assert {finding.code for finding in transform.report.scope(SCOPE).findings} == {
        "dataflow-selection-commit-failed"
    }


# -- the reference policy ----------------------------------------------------


def test_the_reference_policy_commits_a_complete_coherent_point() -> None:
    lowered, transform = _run(FirstFeasiblePolicy())
    operation = _operation(lowered)
    resolved = operation.resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)
    report = transform.report.scope(SCOPE)
    assert report.structural_readiness is not None
    assert report.structural_readiness.ready is True
    for assessment in report.feasibility.values():
        assert assessment.verdict is not False


def test_readiness_and_feasibility_are_reported_separately() -> None:
    _, transform = _run(FirstFeasiblePolicy())
    report = transform.report.scope(SCOPE)
    assert report.structural_readiness is not None
    assert report.artifact_readiness is not None
    # One operation-level set covers the selected design and its placements.
    assert set(report.feasibility) == set(MvauDataflowOp.feasibility_constraint_sets())
    assert set(report.feasibility) == {"mvau_op_feasibility"}


def test_enumeration_returns_whole_points_not_one_decision_at_a_time() -> None:
    model = _model()
    operation = _operation(model)
    engine = Engine()
    context = DataflowSelectionContext(
        SCOPE,
        operation,
        engine,
        operation.hydrate_dataflow_point(_context()),
        tuple(sorted(MvauDataflowOp.decision_nodeattrs())),
        "mvau_op_feasibility",
    )
    points = context.feasible_points()
    assert points
    for point in points:
        assert MVAUDataflowOpPaths.DESIGN in point.assignments
        assert point.assignments[MVAUDataflowOpPaths.DESIGN] == DotProductDesign.id
        result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
        assert isinstance(result, Decided)


def test_initializer_free_runtime_writable_points_defer_memstream_supply() -> None:
    model = _model(with_initializer=False)
    operation = _operation(model)
    engine = Engine()
    context = DataflowSelectionContext(
        SCOPE,
        operation,
        engine,
        operation.hydrate_dataflow_point(
            MVAUDataflowBuildContext(_BuildConfig(), runtime_writable_weights=True)
        ),
        tuple(sorted(MvauDataflowOp.decision_nodeattrs())),
        "mvau_op_feasibility",
    )
    suppliers = {
        point.assignments.get(MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path)
        for point in context.feasible_points()
    }
    assert suppliers == {EXTERNAL_SUPPLY}


def test_committed_choices_survive_save_and_reload(tmp_path: Path) -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _cyclic_assignments()}))
    original = _operation(lowered).resolve_dataflow(_context())
    path = Path(str(tmp_path)) / "selected.onnx"
    lowered.save(path)
    restored = _operation(ModelWrapper(str(path))).resolve_dataflow(_context())
    assert restored.point.assignments == original.point.assignments
    assert restored.result == original.result
    assert restored.source_association.kernel_ids == (
        "dotp_axi",
        "replay_buffer",
        FINN_RTL_MEMSTREAM_SUPPLY,
    )


def test_the_transform_needs_no_operation_specific_configuration() -> None:
    """The operation names its own contract, so the caller supplies none."""

    assert MvauDataflowOp.selection_constraint_set() == "mvau_op_feasibility"
    assert MvauDataflowOp.structural_readiness_profile() == "mvau_op_structural"
    assert MvauDataflowOp.artifact_readiness_profile() == "artifact_inputs"
    assert not hasattr(MvauDataflowOp, "kernel_selections")


def test_a_policy_sees_every_scope_in_one_call() -> None:
    """Coordination across scopes is possible because the policy sees them all."""

    seen: list[tuple[str, ...]] = []

    class _Recording(DataflowSelectionPolicy):
        def select(
            self,
            model: ModelWrapper,
            contexts: "Sequence[DataflowSelectionContext]",
        ) -> dict[str, dict[QualifiedPath, object]]:
            assert model is not None
            seen.append(tuple(item.scope_id for item in contexts))
            return {}

    _run(_Recording())
    assert seen == [(SCOPE,)]


def test_selection_does_not_mutate_the_node_class_or_domain() -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _cyclic_assignments()}))
    node = lowered.graph.node[0]
    assert node.op_type == "MvauDataflowOp"
    assert node.domain == "finn.custom_op.dataflow"


def test_a_partial_policy_leaves_an_explorable_point() -> None:
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    partial = {
        MVAU_DESIGN_INVENTORY.inventory.design_path: DotProductDesign.id,
        MVAU_DESIGN_INVENTORY.dot_product.pe.path: 2,
    }
    lowered, transform = _run(ExplicitAssignmentsPolicy({SCOPE: partial}))
    operation = _operation(lowered)
    point = operation.hydrate_dataflow_point(_context())
    assert set(point.assignments) == set(partial)
    report = transform.report.scope(SCOPE)
    assert report.structural_readiness is not None
    assert report.structural_readiness.ready is None
