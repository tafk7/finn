# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The final-acceptance chain of the Op/Kernel waterfall plan, end to end.

MvauDataflowOp class      -> closed DataflowDesign inventory
MvauDataflowOp instance   -> live graph/build ProblemInstance
source inference          -> unresolved node admitted by design constraints
selection policy          -> design + supply + Kernel-local choices
selected design           -> one flat Network and configured Kernels
composition               -> physical elaboration + artifacts
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
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.artifacts.supplied import (
    build_supplied_artifact_requirements,
)
from finn.dataflow.ops.mvau.input_supply import FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.ops.mvau.elaboration import elaborate_mvau
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAUParameterTopology,
    NetworkRef,
)
from finn.dataflow.ops.mvau.op import MVAUDataflowBuildContext, MvauDataflowOp
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
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

    # source inference -> an unresolved logical node admitted by the inventory
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

    # selection policy -> design + common supply + Kernel-local choices
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    policy = ExplicitAssignmentsPolicy(
        {
            scope_id: {
                MVAU_DESIGN_INVENTORY.inventory.design_path: DotProductDesign.id,
                MVAU_DESIGN_INVENTORY.dot_product.pe.path: 2,
                MVAU_DESIGN_INVENTORY.dot_product.simd.path: 2,
                MVAU_DESIGN_INVENTORY.compute_pumping.path: False,
                MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: (
                    FINN_RTL_MEMSTREAM_SUPPLY
                ),
                MVAU_DESIGN_INVENTORY.input_supply.settings.ram_style.path: (CyclicRamStyle.BRAM),
                MVAU_DESIGN_INVENTORY.input_supply.settings.pumped_memory.path: False,
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

    # selected design -> Network and configured physical Kernels
    operation = _operation(model)
    resolved = operation.resolve_dataflow(_context())
    engine = resolved.engine
    realization = MVAU_DESIGN_INVENTORY.inventory.realize(engine, resolved.point)
    assert isinstance(realization, Decided)
    assert set(realization.value.kernels) == {"compute", "replay", "delivery"}

    # assembly -> NetworkRef, with the topology read back rather than chosen
    assert isinstance(resolved.result, NetworkRef)
    association = resolved.result.source_association
    assert association.parameter_topology is MVAUParameterTopology.CYCLIC
    assert association.design_id == DotProductDesign.id
    assert association.compute_kernel_id == "dotp_axi"
    assert association.supply_kernel_id == FINN_RTL_MEMSTREAM_SUPPLY
    assert association.adapter_kernel_id is None
    assert association.fused_source_node_ids == ("matmul0",)

    # configured Kernels -> physical elaboration + supplied artifact requirements
    elaboration = elaborate_mvau(resolved)
    assert elaboration.origin.kernel_ids == ("dotp_axi", "replay_buffer", "finn_rtl_memstream")
    assert elaboration.origin.provider_ids == ()
    weights = model.get_initializer("weights")
    assert weights is not None
    requirements = build_supplied_artifact_requirements(
        resolved,
        realization.value,
        elaboration,
        weights,
        Path.cwd(),
    )
    assert requirements.elaboration.origin == elaboration.origin

    # node-backed choices stay sparse, transactional, and reloadable
    path = tmp_path / "accepted.onnx"
    model.save(path)
    restored = _operation(ModelWrapper(str(path))).resolve_dataflow(_context())
    assert restored.point.assignments == resolved.point.assignments
    assert restored.result == resolved.result
    assert restored.source_scope_id == resolved.source_scope_id


def test_the_result_property_uses_only_the_frozen_v6_decisions() -> None:

    decisions = {str(item.path) for item in MVAU_DATAFLOW_OP_SPEC.decisions}
    assert "semantic.mvau.op.parameter_topology" not in decisions
    assert decisions == {str(path) for path in MvauDataflowOp.decision_nodeattrs()}
