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
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import (
    LEGACY_HLS_PATHS,
    MVAU_COMPUTE_SELECTION,
    SOFT_VECTOR_PATHS,
    MVAUComputeKernelId,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.ops.mvau import (
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    NetworkRef,
    RegionRef,
)
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    MVAUWeightSupplyKernelId,
    WeightOrganization,
)
from finn.transformation.fpgadataflow.select_dataflow_design import (
    DataflowSelectionContext,
    DataflowSelectionPolicy,
    ExplicitAssignmentsPolicy,
    FirstFeasiblePolicy,
    SelectDataflowDesign,
    committed_kernel_ids,
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
    return {
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.SOFT_VECTOR.value,
        SOFT_VECTOR_PATHS.pe: 2,
        SOFT_VECTOR_PATHS.simd: 2,
        SOFT_VECTOR_PATHS.compute_pumping: False,
        MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
    }


def _cyclic_assignments() -> dict[QualifiedPath, object]:
    return {
        **_direct_assignments(),
        MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: (
            MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
        ),
        FINN_RTL_MEMSTREAM_PATHS.organization: WeightOrganization.AS_DEMANDED,
        FINN_RTL_MEMSTREAM_PATHS.ram_style: CyclicRamStyle.BRAM,
        FINN_RTL_MEMSTREAM_PATHS.pumped_memory: False,
        MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel: NO_KERNEL,
    }


def _embedded_assignments() -> dict[QualifiedPath, object]:
    return {
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.LEGACY_HLS.value,
        LEGACY_HLS_PATHS.pe: 2,
        LEGACY_HLS_PATHS.simd: 2,
        LEGACY_HLS_PATHS.resource: MVAUHlsResource.LUT,
        LEGACY_HLS_PATHS.weight_source: MVAUWeightSource.EMBEDDED,
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
    assert isinstance(resolved.result, RegionRef)
    assert resolved.result.source_association.parameter_topology is MVAUParameterTopology.DIRECT
    assert transform.report.scope(SCOPE).committed


def test_an_explicit_policy_reproduces_the_cyclic_point() -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _cyclic_assignments()}))
    resolved = _operation(lowered).resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)
    assert resolved.result.source_association.parameter_topology is MVAUParameterTopology.CYCLIC


def test_an_explicit_policy_reproduces_the_embedded_point() -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _embedded_assignments()}))
    resolved = _operation(lowered).resolve_dataflow(_context())
    assert isinstance(resolved.result, RegionRef)
    assert resolved.result.source_association.parameter_topology is MVAUParameterTopology.EMBEDDED


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
        ExplicitAssignmentsPolicy({SCOPE: {MVAU_COMPUTE_SELECTION.paths.kernel: "absent-kernel"}}),
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
    assert isinstance(resolved.result, (RegionRef, NetworkRef))
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
    # One feasibility set per declared pool, named by the operation itself.
    assert set(report.feasibility) == set(MvauDataflowOp.feasibility_constraint_sets())
    assert set(report.feasibility) == {
        MVAU_COMPUTE_SELECTION.feasibility_constraint_set,
        MVAU_WEIGHT_SUPPLY_SELECTION.feasibility_constraint_set,
        MVAU_WEIGHT_ADAPTER_SELECTION.feasibility_constraint_set,
    }


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
        assert MVAU_COMPUTE_SELECTION.paths.kernel in point.assignments
        result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
        assert isinstance(result, Decided)


def test_a_supplier_veto_removes_joint_points_but_leaves_others() -> None:
    """FinnLib cannot serve a runtime-written array; the RTL streamer can."""

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
        point.assignments.get(MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel)
        for point in context.feasible_points()
    }
    assert MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM.value not in suppliers


def test_committed_choices_survive_save_and_reload(tmp_path: Path) -> None:
    lowered, _ = _run(ExplicitAssignmentsPolicy({SCOPE: _cyclic_assignments()}))
    original = _operation(lowered).resolve_dataflow(_context())
    path = Path(str(tmp_path)) / "selected.onnx"
    lowered.save(path)
    restored = _operation(ModelWrapper(str(path))).resolve_dataflow(_context())
    assert restored.point.assignments == original.point.assignments
    assert restored.result == original.result
    identities = committed_kernel_ids(
        restored.engine,
        restored.point,
        (
            MVAU_COMPUTE_SELECTION.paths.selected_kernel,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.selected_kernel,
        ),
    )
    assert identities[MVAU_COMPUTE_SELECTION.paths.selected_kernel] == (
        MVAUComputeKernelId.SOFT_VECTOR.value
    )
    assert identities[MVAU_WEIGHT_SUPPLY_SELECTION.paths.selected_kernel] == (
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
    )


def test_the_transform_needs_no_operation_specific_configuration() -> None:
    """The operation names its own contract, so the caller supplies none."""

    assert MvauDataflowOp.selection_constraint_set() == "mvau_op_feasibility"
    assert MvauDataflowOp.structural_readiness_profile() == "mvau_op_structural"
    assert MvauDataflowOp.artifact_readiness_profile() == "artifact_inputs"
    assert MvauDataflowOp.kernel_selections() == (
        MVAU_COMPUTE_SELECTION,
        MVAU_WEIGHT_SUPPLY_SELECTION,
        MVAU_WEIGHT_ADAPTER_SELECTION,
    )


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
    partial = {
        MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.SOFT_VECTOR.value,
        SOFT_VECTOR_PATHS.pe: 2,
    }
    lowered, transform = _run(ExplicitAssignmentsPolicy({SCOPE: partial}))
    operation = _operation(lowered)
    point = operation.hydrate_dataflow_point(_context())
    assert set(point.assignments) == set(partial)
    report = transform.report.scope(SCOPE)
    assert report.structural_readiness is not None
    assert report.structural_readiness.ready is None
