# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The final-acceptance chain of the Op/Kernel waterfall plan, end to end.

MvauDataflowOp class      -> static compute and supplier Kernel pools
MvauDataflowOp instance   -> live graph/build ProblemInstance
source inference          -> unresolved node admitted by Kernel constraints
selection policy          -> compute Kernel + supplier Kernel + locals
selected Kernels          -> Regions + demands
assembly                  -> RegionRef or NetworkRef
providers                 -> physical elaboration + artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.core.onnx_exec import execute_onnx  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Decided
from finn.dataflow.kernels import NO_KERNEL, SelectedKernel
from finn.dataflow.mvau.artifacts import build_mvau_rtl_artifact_requirements
from finn.dataflow.mvau.compute_kernels import (
    MVAU_COMPUTE_SELECTION,
    SOFT_VECTOR_PATHS,
    SOFT_VECTOR_PROVIDER_ID,
    WEIGHT_INTERFACE,
    MVAUComputeKernelId,
)
from finn.dataflow.mvau.elaboration import elaborate_mvau_rtl_softvec
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    NetworkRef,
)
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    MEMSTREAM_PROVIDER_ID,
    MVAUWeightSupplyKernelId,
    WeightOrganization,
)
from finn.transformation.fpgadataflow.infer_mvau_dataflow import (
    InferMVAUDataflowOp,
    source_nodes_of,
)
from finn.transformation.fpgadataflow.select_dataflow_design import (
    ExplicitAssignmentsPolicy,
    SelectDataflowDesign,
)

PART = "xczu3eg-sbva484-1-e"


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


def _source_model() -> ModelWrapper:
    width = height = rows = 4
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["activation", "weights"], ["output"], name="matmul0")],
        "acceptance",
        [
            helper.make_tensor_value_info("activation", TensorProto.FLOAT, [rows, width]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [width, height]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, height])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="acceptance"))
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT16"])
    model.set_initializer("weights", np.ones((width, height), dtype=np.float32))
    return model


def _operation(model: ModelWrapper) -> MvauDataflowOp:
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    return operation


def test_the_waterfall_runs_from_source_matmul_to_artifact_requirements(
    tmp_path: Path,
) -> None:
    source = _source_model()
    activation = np.arange(16, dtype=np.float32).reshape(4, 4)
    expected = execute_onnx(source, {"activation": activation})["output"]

    # source inference -> an unresolved logical node admitted by the pool
    inference = InferMVAUDataflowOp(_context())
    model = source.transform(inference, cleanup=False)
    assert len(inference.report.lowered) == 1
    operation = _operation(model)
    assert operation.read_assignments() == {}
    assert source_nodes_of(operation) == ("matmul0",)
    scope_id = operation.dataflow_scope_id()

    # the source function survives the lowering
    np.testing.assert_array_equal(
        execute_onnx(model, {"activation": activation})["output"], expected
    )

    # selection policy -> compute Kernel + supplier Kernel + local choices
    policy = ExplicitAssignmentsPolicy(
        {
            scope_id: {
                MVAU_COMPUTE_SELECTION.paths.kernel: MVAUComputeKernelId.SOFT_VECTOR.value,
                SOFT_VECTOR_PATHS.pe: 2,
                SOFT_VECTOR_PATHS.simd: 2,
                SOFT_VECTOR_PATHS.compute_pumping: False,
                MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: (
                    MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
                ),
                FINN_RTL_MEMSTREAM_PATHS.organization: WeightOrganization.AS_DEMANDED,
                FINN_RTL_MEMSTREAM_PATHS.ram_style: CyclicRamStyle.BRAM,
                FINN_RTL_MEMSTREAM_PATHS.pumped_memory: False,
                MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel: NO_KERNEL,
            }
        }
    )
    selection = SelectDataflowDesign(
        policy,
        _context(),
    )
    model = model.transform(selection, cleanup=False)
    report = selection.report.scope(scope_id)
    assert report.findings == ()
    assert report.structural_readiness is not None
    assert report.structural_readiness.ready is True
    assert report.artifact_readiness is not None
    assert report.artifact_readiness.ready is True
    for assessment in report.feasibility.values():
        assert assessment.verdict is True

    # selected Kernels -> Regions + demands
    operation = _operation(model)
    resolved = operation.resolve_dataflow(_context())
    engine = resolved.engine
    assert engine.query_property(
        resolved.point, MVAU_COMPUTE_SELECTION.paths.selected_kernel
    ) == Decided(
        SelectedKernel(MVAU_COMPUTE_SELECTION.name, MVAUComputeKernelId.SOFT_VECTOR.value, "1")
    )
    demand = engine.query_property(
        resolved.point, MVAU_COMPUTE_SELECTION.paths.demand(WEIGHT_INTERFACE)
    )
    supply_port = engine.query_property(
        resolved.point, MVAU_WEIGHT_SUPPLY_SELECTION.paths.export("output_port")
    )
    assert isinstance(demand, Decided) and isinstance(supply_port, Decided)
    assert demand.value == supply_port.value

    # assembly -> NetworkRef, with the topology read back rather than chosen
    assert isinstance(resolved.result, NetworkRef)
    association = resolved.result.source_association
    assert association.parameter_topology is MVAUParameterTopology.CYCLIC
    assert association.compute_kernel_id == MVAUComputeKernelId.SOFT_VECTOR.value
    assert association.supply_kernel_id == MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
    assert association.adapter_kernel_id is None
    assert association.fused_source_node_ids == ("matmul0",)

    # providers -> physical elaboration + artifact requirements
    elaboration = elaborate_mvau_rtl_softvec(resolved)
    assert elaboration.origin.kernel_ids == (
        MVAUComputeKernelId.SOFT_VECTOR.value,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value,
    )
    assert elaboration.origin.provider_ids == (
        SOFT_VECTOR_PROVIDER_ID,
        MEMSTREAM_PROVIDER_ID,
    )
    requirements = build_mvau_rtl_artifact_requirements(resolved, elaboration, model, Path.cwd())
    assert requirements.elaboration.origin == elaboration.origin

    # node-backed choices stay sparse, transactional, and reloadable
    path = tmp_path / "accepted.onnx"
    model.save(path)
    restored = _operation(ModelWrapper(str(path))).resolve_dataflow(_context())
    assert restored.point.assignments == resolved.point.assignments
    assert restored.result == resolved.result
    assert restored.source_scope_id == resolved.source_scope_id


def test_the_result_property_never_needs_a_topology_decision() -> None:
    """Equal Regions keep distinct Kernels, and no choice grid remains."""

    decisions = {str(item.path) for item in MVAU_DATAFLOW_OP_SPEC.decisions}
    assert str(MVAUDataflowOpPaths.PARAMETER_TOPOLOGY) not in decisions
    identity_decisions = {
        str(MVAU_COMPUTE_SELECTION.paths.kernel),
        str(MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel),
        str(MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel),
    }
    assert identity_decisions <= decisions
    for path in decisions - identity_decisions:
        # Everything else belongs to exactly one named Kernel.
        assert any(
            path.startswith(f"{selection.name}.{kernel.id}.")
            for selection in (
                MVAU_COMPUTE_SELECTION,
                MVAU_WEIGHT_SUPPLY_SELECTION,
                MVAU_WEIGHT_ADAPTER_SELECTION,
            )
            for kernel in selection.kernels
        ), path
