# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Decided, Engine, QualifiedPath
from finn.dataflow.mvau.artifacts import (
    MVAUWeightPayloadKind,
    build_mvau_rtl_artifact_requirements,
)
from finn.dataflow.mvau.definition import MVAUComputeBinding, MVAUComputeKernelPaths
from finn.dataflow.mvau.elaboration import elaborate_mvau_rtl_softvec
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_batch_interleaved_mvau_weight_port,
)
from finn.dataflow.mvau.source import (
    MVAUProjectionContext,
    MVAUResolvedDesign,
    project_mvau_source,
    start_mvau_projection,
)
from finn.dataflow.op import DataflowOpError
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAUConnectionTopology,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    MVAUWeightDeliveryDeclaration,
    NetworkRef,
    RegionRef,
)
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterBinding,
    CyclicParameterKernelPaths,
    CyclicRamStyle,
)
from finn.dataflow.region import BeatSequence, NumericElementType
from finn.dataflow.testing import DataflowOpConformanceCase, assert_dataflow_op_conforms

NODE_ID = "logical_mvau0"
PART = "xczu3eg-sbva484-1-e"
VERSAL_PART = "xcvc1902-vsva2197-2MP-e-S"


@dataclass
class _BuildConfig:
    synth_clk_period_ns: float = 5.0
    fpga_part: str | None = PART

    def _resolve_fpga_part(self) -> str:
        if self.fpga_part is None:
            raise ValueError("no target part")
        return self.fpga_part


def _model(
    *,
    repetitions: int = 4,
    fused: bool = False,
    with_initializer: bool = True,
) -> ModelWrapper:
    matrix_width = 4
    matrix_height = 4
    inputs = ["activation", "weights"]
    graph_inputs = [
        helper.make_tensor_value_info("activation", TensorProto.FLOAT, [repetitions, matrix_width]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [matrix_width, matrix_height]),
    ]
    if fused:
        inputs.append("thresholds")
        graph_inputs.append(
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [matrix_height, 3])
        )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [repetitions, matrix_height]
    )
    node = helper.make_node(
        "MvauDataflowOp",
        inputs,
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.dataflow",
        noActivation=0 if fused else 1,
        binaryXnorMode=0,
        accDataType="INT16",
        ActVal=0,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], "logical-mvau", graph_inputs, [output]),
            producer_name="logical-mvau-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["UINT2"] if fused else DataType["INT16"])
    weights = np.asarray(
        [
            [-128, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=np.float32,
    )
    if with_initializer:
        model.set_initializer("weights", weights)
    if fused:
        model.set_tensor_datatype("thresholds", DataType["INT16"])
        model.set_initializer("thresholds", np.zeros((matrix_height, 3), dtype=np.float32))
    return model


def _wrapped(model: ModelWrapper) -> MvauDataflowOp:
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    return operation


def _change_mvau_shape(model: ModelWrapper) -> None:
    model.set_tensor_shape("activation", [2, 4])
    model.set_tensor_shape("output", [2, 4])


def _context(
    *,
    part: str = PART,
    clock: float = 5.0,
    runtime_writable: bool = False,
    external_weight_sequence: BeatSequence | None = None,
) -> MVAUDataflowBuildContext:
    return MVAUDataflowBuildContext(
        _BuildConfig(clock, part),
        runtime_writable_weights=runtime_writable,
        external_weight_sequence=external_weight_sequence,
    )


def _compute(
    declaration: MVAURegionDeclaration,
    topology: MVAUParameterTopology,
    *,
    binding: MVAUComputeBinding = MVAUComputeBinding.RTL_SOFTVEC,
) -> dict[QualifiedPath | str, object]:
    assignments: dict[QualifiedPath | str, object] = {
        MVAUComputeKernelPaths.PE: 2,
        MVAUComputeKernelPaths.SIMD: 2,
        MVAUComputeKernelPaths.REGION_DECLARATION: declaration,
        MVAUComputeKernelPaths.BINDING: binding,
        MVAUDataflowOpPaths.PARAMETER_TOPOLOGY: topology,
    }
    if declaration is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED:
        assignments[MVAUComputeKernelPaths.INTERLEAVE] = 2
    if binding in {MVAUComputeBinding.RTL_SOFTVEC, MVAUComputeBinding.RTL_PACKED}:
        assignments[MVAUComputeKernelPaths.COMPUTE_PUMPING] = False
    return assignments


def _cyclic(
    *,
    delivery: MVAUWeightDeliveryDeclaration = MVAUWeightDeliveryDeclaration.STANDARD_FULL_TILE,
    connection: MVAUConnectionTopology = MVAUConnectionTopology.DIRECT,
) -> dict[QualifiedPath | str, object]:
    assignments: dict[QualifiedPath | str, object] = {
        MVAUDataflowOpPaths.DELIVERY_PE: 2,
        MVAUDataflowOpPaths.DELIVERY_SIMD: 2,
        MVAUDataflowOpPaths.DELIVERY_DECLARATION: delivery,
        MVAUDataflowOpPaths.CONNECTION_TOPOLOGY: connection,
        CyclicParameterKernelPaths.BINDING: CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        CyclicParameterKernelPaths.RAM_STYLE: CyclicRamStyle.BRAM,
        CyclicParameterKernelPaths.PUMPED_MEMORY: False,
    }
    if delivery is MVAUWeightDeliveryDeclaration.BATCH_INTERLEAVED_CHUNKED:
        assignments[MVAUDataflowOpPaths.DELIVERY_INTERLEAVE] = 2
    return assignments


def _standalone(
    model: ModelWrapper,
    context: MVAUDataflowBuildContext,
    assignments: dict[QualifiedPath | str, object],
) -> MVAUResolvedDesign:
    build = cast(_BuildConfig, context.build_config)
    projection = project_mvau_source(
        model,
        NODE_ID,
        MVAUProjectionContext(
            context.accumulator_type_analysis_owner,
            fpga_part=build.fpga_part,
            clock_period_ns=build.synth_clk_period_ns,
            supports_initialized_uram=context.supports_initialized_uram,
            external_weight_sequence=context.external_weight_sequence,
            runtime_writable_weights=context.runtime_writable_weights,
        ),
    )
    return start_mvau_projection(projection, assignments)


def _assert_resolution_parity(
    current: MVAUResolvedDesign,
    expected: MVAUResolvedDesign,
) -> None:
    assert current.point.problem == expected.point.problem
    assert current.point.assignments == expected.point.assignments
    assert current.result == expected.result
    assert current.source_association == expected.source_association
    for set_name in ("mvau_op_structural", "binding_feasibility"):
        assert current.engine.evaluate_constraint_set(
            current.point, set_name
        ) == expected.engine.evaluate_constraint_set(expected.point, set_name)
    assert current.engine.check_readiness(
        current.point, "artifact_inputs"
    ) == expected.engine.check_readiness(expected.point, "artifact_inputs")


def test_logical_mvau_registration_static_spec_and_problem_projection_parity() -> None:
    model = _model()
    operation = _wrapped(model)
    context = _context()
    expected = project_mvau_source(
        model,
        NODE_ID,
        MVAUProjectionContext(
            "finn.MinimizeAccumulatorWidth",
            fpga_part=PART,
            clock_period_ns=5.0,
            runtime_writable_weights=False,
        ),
    )

    assert operation._attached_model() is model
    assert type(operation).build_design_space_spec() is MVAU_DATAFLOW_OP_SPEC
    assert operation.problem_instance(context) == expected.problem_data
    assert operation.read_assignments() == {}


def test_graph_and_build_projection_keep_fact_ownership_explicit() -> None:
    model = _model(with_initializer=False)
    operation = _wrapped(model)
    context = _context(part=VERSAL_PART, clock=3.0, runtime_writable=True)
    graph_problem = operation.project_graph_problem()
    build_problem = operation.project_build_problem(context)
    problem = operation.problem_instance(context)

    assert MVAUDataflowOpPaths.TARGET_FPGA_PART not in graph_problem
    assert MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS not in graph_problem
    assert CyclicParameterKernelPaths.RUNTIME_WRITABLE not in graph_problem
    assert build_problem[MVAUDataflowOpPaths.TARGET_FPGA_PART] == VERSAL_PART
    assert build_problem[MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS] == 3.0
    assert build_problem[CyclicParameterKernelPaths.RUNTIME_WRITABLE] is True
    assert problem[MVAUComputeKernelPaths.WEIGHTS_NARROW] is False
    attribute_names = {attribute.name for attribute in operation.onnx_node.attribute}
    assert not attribute_names & {
        "MW",
        "MH",
        "inputDataType",
        "weightDataType",
        "outputDataType",
        "weight_initializer",
        "runtime_writeable_weights",
    }


def test_real_dataflow_build_config_supplies_target_and_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_ROOT", str(Path.cwd()))
    module = importlib.import_module("finn.builder.build_dataflow_config")
    config = module.DataflowBuildConfig("build", 4.0, [], fpga_part=PART)
    problem = _wrapped(_model()).problem_instance(config)
    assert problem[MVAUDataflowOpPaths.TARGET_FPGA_PART] == PART
    assert problem[MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS] == 4.0


def test_every_mvau_decision_has_one_stable_node_attribute() -> None:
    operation = _wrapped(_model())
    codecs = type(operation).decision_nodeattrs()
    space = type(operation).validated_design_space()
    assert set(codecs) == set(space.decisions)
    assert len({codec.attribute_name for codec in codecs.values()}) == len(codecs)
    assert not set(codecs) & set(space.properties)
    assert not set(codecs) & set(space.constraints)
    assert dict(operation.get_nodeattr_types()) == dict(operation.get_nodeattr_types())


@pytest.mark.parametrize(
    "declaration,topology,expected_type",
    [
        (
            MVAURegionDeclaration.STANDARD_EMBEDDED,
            MVAUParameterTopology.EMBEDDED,
            RegionRef,
        ),
        (
            MVAURegionDeclaration.STANDARD_STREAMED,
            MVAUParameterTopology.DIRECT,
            RegionRef,
        ),
    ],
)
def test_logical_mvau_region_topologies_match_standalone_resolution(
    tmp_path: Path,
    declaration: MVAURegionDeclaration,
    topology: MVAUParameterTopology,
    expected_type: type[RegionRef],
) -> None:
    model = _model()
    operation = _wrapped(model)
    binding = (
        MVAUComputeBinding.LEGACY_HLS_LUT
        if declaration is MVAURegionDeclaration.STANDARD_EMBEDDED
        else MVAUComputeBinding.RTL_SOFTVEC
    )
    assignments = _compute(declaration, topology, binding=binding)
    committed = operation.commit_dataflow_assignments(_context(), assignments)
    resolved = operation.resolve_dataflow(_context())
    expected = project_mvau_source(
        model,
        NODE_ID,
        MVAUProjectionContext(
            "finn.MinimizeAccumulatorWidth",
            fpga_part=PART,
            clock_period_ns=5.0,
            runtime_writable_weights=False,
        ),
    )
    assert isinstance(resolved, MVAUResolvedDesign)
    assert isinstance(resolved.result, expected_type)
    assert resolved.point.problem == expected.problem_data
    assert resolved.point.assignments == committed.point.assignments
    assert resolved.result.source_association == resolved.source_association
    assert resolved.source_scope_id == operation.get_nodeattr(operation.SCOPE_ID_ATTR)
    _assert_resolution_parity(resolved, start_mvau_projection(expected, assignments))
    path = tmp_path / f"{topology.value}.onnx"
    model.save(path)
    restored = _wrapped(ModelWrapper(str(path))).resolve_dataflow(_context())
    _assert_resolution_parity(restored, resolved)


def test_batch_interleaved_external_contract_resolves_directly(tmp_path: Path) -> None:
    external = construct_batch_interleaved_mvau_weight_port(
        4, 4, 4, NumericElementType("int", 8), 2, 2, 2
    ).beat_sequence
    model = _model()
    operation = _wrapped(model)
    context = _context(part=VERSAL_PART, external_weight_sequence=external)
    assignments = _compute(
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
        MVAUParameterTopology.DIRECT,
        binding=MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58,
    )
    operation.commit_dataflow_assignments(context, assignments)
    resolved = operation.resolve_dataflow(context)
    assert isinstance(resolved.result, RegionRef)
    assessment = resolved.engine.evaluate_constraint_set(resolved.point, "mvau_op_structural")
    assert assessment.verdict is True
    _assert_resolution_parity(resolved, _standalone(model, context, assignments))
    path = tmp_path / "batch-interleaved-direct.onnx"
    model.save(path)
    restored = _wrapped(ModelWrapper(str(path))).resolve_dataflow(context)
    _assert_resolution_parity(restored, resolved)


@pytest.mark.parametrize("adapter", [False, True])
def test_cyclic_network_and_adapter_topologies_round_trip(tmp_path: Path, adapter: bool) -> None:
    model = _model()
    operation = _wrapped(model)
    compute_declaration = (
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
        if adapter
        else MVAURegionDeclaration.STANDARD_STREAMED
    )
    context = _context(part=VERSAL_PART if adapter else PART)
    assignments = {
        **_compute(
            compute_declaration,
            MVAUParameterTopology.CYCLIC,
            binding=(
                MVAUComputeBinding.RTL_BATCH_INTERLEAVED_DSP58
                if adapter
                else MVAUComputeBinding.RTL_SOFTVEC
            ),
        ),
        **_cyclic(
            delivery=MVAUWeightDeliveryDeclaration.STANDARD_FULL_TILE,
            connection=(
                MVAUConnectionTopology.ADAPTER if adapter else MVAUConnectionTopology.DIRECT
            ),
        ),
    }
    operation.commit_dataflow_assignments(context, assignments)
    original = operation.resolve_dataflow(context)
    assert isinstance(original.result, NetworkRef)
    expected_nodes = (
        ("compute", "delivery", "weight_adapter") if adapter else ("compute", "delivery")
    )
    assert tuple(node.id for node in original.result.network.nodes) == expected_nodes
    _assert_resolution_parity(original, _standalone(model, context, assignments))
    path = tmp_path / f"cyclic-{adapter}.onnx"
    model.save(path)
    restored = _wrapped(ModelWrapper(str(path))).resolve_dataflow(context)
    assert restored.point.assignments == original.point.assignments
    assert restored.result == original.result
    assert restored.source_scope_id == original.source_scope_id


def test_node_backed_result_is_accepted_by_existing_provider_path() -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(
        _context(),
        _compute(MVAURegionDeclaration.STANDARD_STREAMED, MVAUParameterTopology.DIRECT),
    )
    resolved = operation.resolve_dataflow(_context())
    elaboration = elaborate_mvau_rtl_softvec(resolved)
    requirements = build_mvau_rtl_artifact_requirements(resolved, elaboration, model, Path.cwd())
    assert elaboration.semantic_result == resolved.result
    assert requirements.elaboration.origin == elaboration.origin


def test_runtime_writable_policy_is_projected_from_build_context() -> None:
    model = _model(with_initializer=False)
    operation = _wrapped(model)
    context = _context(runtime_writable=True)
    assignments = {
        **_compute(MVAURegionDeclaration.STANDARD_STREAMED, MVAUParameterTopology.CYCLIC),
        **_cyclic(),
    }
    operation.commit_dataflow_assignments(context, assignments)
    resolved = operation.resolve_dataflow(context)
    elaboration = elaborate_mvau_rtl_softvec(resolved)
    requirements = build_mvau_rtl_artifact_requirements(resolved, elaboration, model, Path.cwd())
    assert resolved.point.problem[CyclicParameterKernelPaths.RUNTIME_WRITABLE] is True
    assert requirements.weight_payload_kind is MVAUWeightPayloadKind.RUNTIME_WRITABLE_LOCAL_STATE
    assert requirements.weight_initializer is None


@pytest.mark.parametrize(
    "changed_fact",
    ["shape", "datatype", "weights", "target", "clock", "runtime_writable"],
)
def test_logical_mvau_stale_graph_and_build_facts_are_rejected(
    changed_fact: str,
) -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(
        _context(),
        _compute(MVAURegionDeclaration.STANDARD_STREAMED, MVAUParameterTopology.DIRECT),
    )
    context = _context()
    if changed_fact == "shape":
        model.set_tensor_shape("activation", [2, 4])
        model.set_tensor_shape("output", [2, 4])
    elif changed_fact == "datatype":
        model.set_tensor_datatype("activation", DataType["INT4"])
    elif changed_fact == "weights":
        model.set_initializer("weights", np.zeros((4, 4), dtype=np.float32))
    elif changed_fact == "target":
        context = _context(part=VERSAL_PART)
    elif changed_fact == "clock":
        context = _context(clock=3.0)
    else:
        context = _context(runtime_writable=True)
    with pytest.raises(DataflowOpError) as stale:
        operation.resolve_dataflow(context)
    assert {finding.code for finding in stale.value.findings} == {
        "dataflow-selection-problem-mismatch"
    }


def test_logical_mvau_assignments_survive_fresh_process_reload(tmp_path: Path) -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(
        _context(),
        _compute(MVAURegionDeclaration.STANDARD_STREAMED, MVAUParameterTopology.DIRECT),
    )
    expected = operation.resolve_dataflow(_context())
    model_path = tmp_path / "logical-mvau.onnx"
    model.save(model_path)
    code = """
import json
import sys
from dataclasses import dataclass
from qonnx.core.modelwrapper import ModelWrapper
from finn.dataflow.ops.mvau_op import MvauDataflowOp
@dataclass
class Config:
    synth_clk_period_ns: float = 5.0
    fpga_part: str = sys.argv[2]
    def _resolve_fpga_part(self):
        return self.fpga_part
model = ModelWrapper(sys.argv[1])
op = model.get_customop_wrapper(model.graph.node[0])
assert isinstance(op, MvauDataflowOp)
resolved = op.resolve_dataflow(Config())
print(json.dumps({
    'assignments': sorted(str(path) for path in resolved.point.assignments),
    'result': repr(resolved.result),
    'scope': resolved.source_scope_id,
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), str(Path.cwd() / "tests"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(model_path), PART],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = __import__("json").loads(completed.stdout)
    assert payload == {
        "assignments": sorted(str(path) for path in expected.point.assignments),
        "result": repr(expected.result),
        "scope": expected.source_scope_id,
    }


def test_logical_mvau_reference_execution() -> None:
    model = _model(repetitions=2)
    operation = _wrapped(model)
    activation = np.asarray([[1, 2, 3, 4], [-1, 0, 1, 2]], dtype=np.float32)
    weights = model.get_initializer("weights")
    assert weights is not None
    context = {
        "activation": activation,
        "weights": weights,
        "output": np.empty((2, 4), dtype=np.float32),
    }
    operation.execute_node(context, model.graph)
    np.testing.assert_array_equal(context["output"], np.matmul(activation, weights))


def test_logical_mvau_shape_datatype_and_verification_follow_source_semantics() -> None:
    model = _model(repetitions=2)
    operation = _wrapped(model)
    operation.verify_node()
    shape_node = operation.make_shape_compatible_op(model)
    shape_attribute = next(
        attribute for attribute in shape_node.attribute if attribute.name == "shape"
    )
    assert tuple(shape_attribute.ints) == (2, 4)
    model.set_tensor_datatype("output", DataType["INT8"])
    operation.infer_node_datatype(model)
    assert model.get_tensor_datatype("output") == DataType["INT16"]


def test_logical_mvau_fused_threshold_reference_execution() -> None:
    model = _model(repetitions=2, fused=True)
    operation = _wrapped(model)
    activation = np.ones((2, 4), dtype=np.float32)
    weights = np.ones((4, 4), dtype=np.float32)
    thresholds = np.zeros((4, 3), dtype=np.float32)
    context = {
        "activation": activation,
        "weights": weights,
        "thresholds": thresholds,
        "output": np.empty((2, 4), dtype=np.float32),
    }
    operation.execute_node(context, model.graph)
    np.testing.assert_array_equal(context["output"], np.full((2, 4), 3.0))


def test_partial_mvau_point_exposes_readiness_without_forcing_resolution() -> None:
    operation = _wrapped(_model())
    operation.commit_dataflow_assignments(_context(), {MVAUComputeKernelPaths.PE: 2})
    point = operation.hydrate_dataflow_point(_context())
    engine = Engine()
    readiness = engine.check_readiness(point, "mvau_op_structural")
    assert readiness.ready is None
    result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert not isinstance(result, Decided)


def test_logical_mvau_passes_shared_operation_conformance_harness(tmp_path: Path) -> None:
    assignments = _compute(
        MVAURegionDeclaration.STANDARD_STREAMED,
        MVAUParameterTopology.DIRECT,
    )
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_model(),
            operation_type=MvauDataflowOp,
            config=_context(),
            complete_assignments=assignments,
            rejected_assignments={MVAUComputeKernelPaths.PE: 3},
            reload_path=tmp_path / "mvau-conformance.onnx",
            stale_config=_context(clock=3.0),
            mutate_graph_problem=_change_mvau_shape,
        )
    )
    assert isinstance(result.original.result, RegionRef)
