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
from qonnx.core.onnx_exec import execute_onnx  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.analysis.verify_custom_nodes import verify_nodes
from finn.dataflow.design import Decided, Engine, QualifiedPath
from finn.dataflow.ops.mvau.problem import MVAUComputationProfile
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.artifacts._implementation import build_decomposed_artifact_requirements
from finn.dataflow.ops.mvau.input_supply import (
    EXTERNAL_SUPPLY,
    FINN_RTL_MEMSTREAM_SUPPLY,
)
from finn.dataflow.ops.mvau.elaboration import elaborate_mvau
from finn.dataflow.ops.mvau.source import (
    MVAUProjectionContext,
    MVAUResolvedDesign,
    project_mvau_source,
    start_mvau_projection,
)
from finn.dataflow.op import DataflowBuildConfigView, DataflowOpError
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAUDataflowOpPaths,
    NetworkRef,
)
from finn.dataflow.datatypes import is_qonnx_datatype
from finn.dataflow.ops.mvau.op import (
    MVAU_DATAFLOW_OP_FAMILY_VERSION,
    MVAUDataflowBuildContext,
    MvauDataflowOp,
)
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.region import BeatSequence
from finn.dataflow.testing import DataflowOpConformanceCase, assert_dataflow_op_conforms
from finn.dataflow.ops.mvau.problem import MVAUProblemPaths

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
    no_activation_attribute: int | None = None,
    binary_xnor_attribute: int | None = 0,
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
    attributes: dict[str, object] = {
        "accDataType": "INT16",
        "ActVal": 0,
    }
    if no_activation_attribute is not None:
        attributes["noActivation"] = no_activation_attribute
    elif fused:
        attributes["noActivation"] = 0
    if binary_xnor_attribute is not None:
        attributes["binaryXnorMode"] = binary_xnor_attribute
    node = helper.make_node(
        "MvauDataflowOp",
        inputs,
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.dataflow",
        dataflow_scope_id=f"{NODE_ID}_scope",
        **attributes,
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


def _dot_product(
    *,
    supply: str = EXTERNAL_SUPPLY,
    pe: int = 2,
    simd: int = 2,
) -> dict[QualifiedPath | str, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    assignments: dict[QualifiedPath | str, object] = {
        assembly.inventory.design_path: DotProductDesign.id,
        assembly.dot_product.pe.path: pe,
        assembly.dot_product.simd.path: simd,
        assembly.compute_pumping.path: False,
        assembly.input_supply.declaration.choice.path: supply,
    }
    if supply == FINN_RTL_MEMSTREAM_SUPPLY:
        assignments.update(
            {
                assembly.input_supply.settings.ram_style.path: CyclicRamStyle.BRAM,
                assembly.input_supply.settings.pumped_memory.path: False,
            }
        )
    return assignments


def _batch_interleaved() -> dict[QualifiedPath | str, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    return {
        assembly.inventory.design_path: BatchInterleavedDesign.id,
        assembly.batch_interleaved.pe.path: 2,
        assembly.batch_interleaved.simd.path: 2,
        assembly.batch_interleaved.interleave.path: 2,
        assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
    }


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
        source_scope_id=f"{NODE_ID}_scope",
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
    for set_name in ("mvau_op_structural", "mvau_op_feasibility"):
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
        source_scope_id=operation.dataflow_scope_id(),
    )

    assert operation._attached_model() is model
    assert type(operation).build_design_space_spec() is MVAU_DATAFLOW_OP_SPEC
    assert operation.problem_instance(context) == expected.problem_data
    assert operation.read_assignments() == {}


def test_logical_source_attributes_use_declared_defaults_and_reject_invalid_values() -> None:
    operation = _wrapped(_model(no_activation_attribute=None))
    assert operation.get_nodeattr("noActivation") == 1
    assert (
        operation.problem_instance(_context())[MVAUProblemPaths.COMPUTATION_PROFILE]
        is MVAUComputationProfile.ACCUMULATOR_INTEGER
    )

    for name in ("noActivation", "binaryXnorMode"):
        invalid = _model(
            no_activation_attribute=2 if name == "noActivation" else 1,
            binary_xnor_attribute=2 if name == "binaryXnorMode" else 0,
        )
        with pytest.raises(DataflowOpError) as malformed:
            _wrapped(invalid).problem_instance(_context())
        assert {finding.code for finding in malformed.value.findings} == {
            "mvau-logical-source-attribute-invalid"
        }


def _narrow_weights(operation: MvauDataflowOp, context: DataflowBuildConfigView) -> object:
    """Ask the operation for its narrow-weight decision, as elaboration does."""

    point = operation.hydrate_dataflow_point(context)
    answer = Engine().query_property(point, MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS)
    assert isinstance(answer, Decided)
    return answer.value


def test_graph_and_build_projection_keep_fact_ownership_explicit() -> None:
    model = _model(with_initializer=False)
    operation = _wrapped(model)
    context = _context(part=VERSAL_PART, clock=3.0, runtime_writable=True)
    graph_problem = operation.project_graph_problem()
    build_problem = operation.project_build_problem(context)
    problem = operation.problem_instance(context)

    assert MVAUDataflowOpPaths.TARGET_FPGA_PART not in graph_problem
    assert MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS not in graph_problem
    assert MVAUProblemPaths.RUNTIME_WRITABLE not in graph_problem
    assert build_problem[MVAUDataflowOpPaths.TARGET_FPGA_PART] == VERSAL_PART
    assert build_problem[MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS] == 3.0
    assert build_problem[MVAUProblemPaths.RUNTIME_WRITABLE] is True
    # Runtime-writable weights are governed by the caller's contract, and none
    # was declared, so the property is False without any projection overwriting
    # the graph's own analysis.
    assert MVAUProblemPaths.RUNTIME_WEIGHT_RANGE_CONTRACT not in problem
    assert _narrow_weights(operation, context) is False
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
    assert {codec.attribute_name for codec in codecs.values()} == {
        "dataflow_design",
        "dataflow_dot_product_pe",
        "dataflow_dot_product_simd",
        "dataflow_interleaved_pe",
        "dataflow_interleaved_simd",
        "dataflow_interleaved_batch",
        "dataflow_weight_supply",
        "dataflow_dotp_axi_pumping",
        "dataflow_finn_rtl_memstream_ram_style",
        "dataflow_finn_rtl_memstream_pumping",
    }
    assert dict(operation.get_nodeattr_types()) == dict(operation.get_nodeattr_types())


def test_dot_product_network_round_trips_through_node_persistence(tmp_path: Path) -> None:
    model = _model()
    operation = _wrapped(model)
    assignments = _dot_product()
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
        source_scope_id=operation.dataflow_scope_id(),
    )
    assert isinstance(resolved, MVAUResolvedDesign)
    assert isinstance(resolved.result, NetworkRef)
    assert {node.id for node in resolved.result.network.nodes} == {"compute", "replay"}
    assert resolved.point.problem == expected.problem_data
    assert resolved.point.assignments == committed.point.assignments
    assert resolved.result.source_association == resolved.source_association
    assert resolved.source_scope_id == operation.get_nodeattr(operation.SCOPE_ID_ATTR)
    _assert_resolution_parity(resolved, start_mvau_projection(expected, assignments))
    path = tmp_path / "dot-product.onnx"
    model.save(path)
    restored = _wrapped(ModelWrapper(str(path))).resolve_dataflow(_context())
    _assert_resolution_parity(restored, resolved)


def test_batch_interleaved_resolves_to_a_singleton_network(tmp_path: Path) -> None:
    model = _model()
    operation = _wrapped(model)
    context = _context(part=VERSAL_PART)
    assignments = _batch_interleaved()
    operation.commit_dataflow_assignments(context, assignments)
    resolved = operation.resolve_dataflow(context)
    assert isinstance(resolved.result, NetworkRef)
    assert tuple(node.id for node in resolved.result.network.nodes) == ("compute",)
    assessment = resolved.engine.evaluate_constraint_set(resolved.point, "mvau_op_structural")
    assert assessment.verdict is True
    feasibility = resolved.engine.evaluate_constraint_set(resolved.point, "mvau_op_feasibility")
    assert feasibility.verdict is None
    _assert_resolution_parity(resolved, _standalone(model, context, assignments))
    path = tmp_path / "batch-interleaved-direct.onnx"
    model.save(path)
    restored = _wrapped(ModelWrapper(str(path))).resolve_dataflow(context)
    _assert_resolution_parity(restored, resolved)


def test_memstream_supplied_dot_product_round_trips_without_an_adapter(tmp_path: Path) -> None:
    model = _model()
    operation = _wrapped(model)
    context = _context()
    assignments = _dot_product(supply=FINN_RTL_MEMSTREAM_SUPPLY)
    operation.commit_dataflow_assignments(context, assignments)
    original = operation.resolve_dataflow(context)
    assert isinstance(original.result, NetworkRef)
    assert {node.id for node in original.result.network.nodes} == {
        "compute",
        "delivery",
        "replay",
    }
    assert {edge.id for edge in original.result.network.edges} == {
        "activation_replay",
        "weight",
    }
    _assert_resolution_parity(original, _standalone(model, context, assignments))
    path = tmp_path / "cyclic.onnx"
    model.save(path)
    restored = _wrapped(ModelWrapper(str(path))).resolve_dataflow(context)
    assert restored.point.assignments == original.point.assignments
    assert restored.result == original.result
    assert restored.source_scope_id == original.source_scope_id


def test_node_backed_result_is_accepted_by_dot_product_physical_path() -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(_context(), _dot_product())
    resolved = operation.resolve_dataflow(_context())
    elaboration = elaborate_mvau(resolved)
    requirements = build_decomposed_artifact_requirements(resolved, elaboration, Path.cwd())
    assert elaboration.semantic_result == resolved.result
    assert requirements.elaboration.origin == elaboration.origin


def test_runtime_writable_policy_is_projected_from_build_context() -> None:
    model = _model(with_initializer=False)
    operation = _wrapped(model)
    context = _context(runtime_writable=True)
    point = operation.hydrate_dataflow_point(context)
    assert point.problem[MVAUProblemPaths.RUNTIME_WRITABLE] is True


@pytest.mark.parametrize(
    "changed_fact",
    ["shape", "datatype", "weights", "target", "clock", "runtime_writable"],
)
def test_logical_mvau_stale_graph_and_build_facts_are_rejected(
    changed_fact: str,
) -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(_context(), _dot_product())
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


def test_a_v5_node_is_rejected_for_its_family_version_not_incidentally() -> None:
    assert MVAU_DATAFLOW_OP_FAMILY_VERSION == "mvau-dataflow-op-v6"

    operation = _wrapped(_model())
    operation.commit_dataflow_assignments(_context(), _dot_product())
    operation.set_nodeattr(operation.FAMILY_VERSION_ATTR, "mvau-dataflow-op-v5")

    with pytest.raises(DataflowOpError) as stale:
        operation.hydrate_dataflow_point(_context())
    assert {finding.code for finding in stale.value.findings} == {
        "dataflow-selection-family-mismatch"
    }
    assert dict(stale.value.findings[0].values) == {
        "actual_family": "finn.dataflow.mvau",
        "actual_version": "mvau-dataflow-op-v5",
        "expected_family": "finn.dataflow.mvau",
        "expected_version": "mvau-dataflow-op-v6",
    }


def test_a_freshly_saved_v6_selection_reloads_with_its_datatypes_intact() -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(_context(), _dot_product())
    assert operation.get_nodeattr(operation.FAMILY_VERSION_ATTR) == "mvau-dataflow-op-v6"

    point = operation.hydrate_dataflow_point(_context())
    for path in (
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE,
    ):
        value = point.problem[path]
        assert is_qonnx_datatype(value), path
        assert value == DataType[value.name], path


def test_logical_mvau_assignments_survive_fresh_process_reload(tmp_path: Path) -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(_context(), _dot_product())
    expected = operation.resolve_dataflow(_context())
    model_path = tmp_path / "logical-mvau.onnx"
    model.save(model_path)
    code = """
import json
import sys
from dataclasses import dataclass
from qonnx.core.modelwrapper import ModelWrapper
from finn.dataflow.ops.mvau.op import MvauDataflowOp
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


def test_logical_mvau_node_rename_preserves_scope_selection_and_provider_lookup() -> None:
    model = _model()
    operation = _wrapped(model)
    operation.commit_dataflow_assignments(_context(), _dot_product())
    original = operation.resolve_dataflow(_context())
    operation.onnx_node.name = "renamed_mvau"

    renamed = operation.resolve_dataflow(_context())
    assert renamed.source_scope_id == original.source_scope_id
    assert renamed.result == original.result
    elaboration = elaborate_mvau(renamed)
    requirements = build_decomposed_artifact_requirements(renamed, elaboration, Path.cwd())
    assert requirements.elaboration.source_scope_id == original.source_scope_id


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


def test_logical_mvau_executes_and_verifies_through_normal_model_consumers() -> None:
    model = _model(repetitions=2)
    activation = np.asarray([[1, 2, 3, 4], [-1, 0, 1, 2]], dtype=np.float32)
    weights = model.get_initializer("weights")
    assert weights is not None

    output = execute_onnx(model, {"activation": activation})
    np.testing.assert_array_equal(output["output"], np.matmul(activation, weights))
    assert model.analysis(verify_nodes) == {"MvauDataflowOp": None}


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
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    operation.commit_dataflow_assignments(
        _context(),
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: DotProductDesign.id,
            MVAU_DESIGN_INVENTORY.dot_product.pe.path: 2,
        },
    )
    point = operation.hydrate_dataflow_point(_context())
    engine = Engine()
    readiness = engine.check_readiness(point, "mvau_op_structural")
    assert readiness.ready is None
    result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    assert not isinstance(result, Decided)


def test_logical_mvau_passes_shared_operation_conformance_harness(tmp_path: Path) -> None:
    assignments = _dot_product()
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_model(),
            node_name=NODE_ID,
            operation_type=MvauDataflowOp,
            config=_context(),
            complete_assignments=assignments,
            rejected_assignments={MVAU_DESIGN_INVENTORY.dot_product.pe.path: 3},
            reload_path=tmp_path / "mvau-conformance.onnx",
            stale_config=_context(clock=3.0),
            mutate_graph_problem=_change_mvau_shape,
        )
    )
    assert isinstance(result.original.result, NetworkRef)
