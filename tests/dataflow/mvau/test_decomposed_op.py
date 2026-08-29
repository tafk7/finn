# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4's forcing case, through a real operation.

The Network constructor and its tests show the two Regions assemble.  This
shows an *operation* doing it: projecting its problem from a live
``ModelWrapper``, selecting both Kernels, returning the decomposed
``NetworkRef``, associating the source tensors with the two nodes that now
carry them, and reconstituting all of that after a save and reload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.mvau.decomposed import (
    ACTIVATION_EDGE,
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
    ActivationReplayKernel,
    DotProductKernel,
)
from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau_problem import MVAUProblemPaths
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import validate_network
from finn.dataflow.ops.mvau import NetworkRef, SemanticOperandDestination
from finn.dataflow.ops.mvau_decomposed import (
    DOT_PRODUCT_SELECTION,
    MVAU_DECOMPOSED_POOLS,
    REPLAY_SELECTION,
    DecomposedMvauDataflowOp,
)
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import validate_region

NODE_ID = "mvau_decomposed"
PART = "xcvc1902-vsva2197-2MP-e-S"
MATRIX_WIDTH = 4
MATRIX_HEIGHT = 4


class _BuildConfig:
    synth_clk_period_ns = 4.0
    fpga_part = PART

    def _resolve_fpga_part(self) -> str:
        return PART


def _context() -> MVAUDataflowBuildContext:
    return MVAUDataflowBuildContext(_BuildConfig())


def _model(*, repetitions: int = 4) -> ModelWrapper:
    node = helper.make_node(
        "DecomposedMvauDataflowOp",
        ["activation", "weights"],
        ["output"],
        name=NODE_ID,
        domain="finn.custom_op.dataflow",
        dataflow_scope_id=f"{NODE_ID}_scope",
        accDataType="INT16",
        ActVal=0,
        noActivation=1,
        binaryXnorMode=0,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [node],
                "decomposed-mvau",
                [
                    helper.make_tensor_value_info(
                        "activation", TensorProto.FLOAT, [repetitions, MATRIX_WIDTH]
                    ),
                    helper.make_tensor_value_info(
                        "weights", TensorProto.FLOAT, [MATRIX_WIDTH, MATRIX_HEIGHT]
                    ),
                ],
                [
                    helper.make_tensor_value_info(
                        "output", TensorProto.FLOAT, [repetitions, MATRIX_HEIGHT]
                    )
                ],
            ),
            producer_name="decomposed-mvau-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT16"])
    model.set_initializer(
        "weights",
        np.asarray(
            [[-128, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.float32
        ),
    )
    return model


def _wrapped(model: ModelWrapper) -> DecomposedMvauDataflowOp:
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, DecomposedMvauDataflowOp)
    return operation


def _choices(*, pe: int = 2, simd: int = 2, pumping: bool = False) -> dict[QualifiedPath, object]:
    pools = MVAU_DECOMPOSED_POOLS
    return {
        DOT_PRODUCT_SELECTION.paths.kernel: DotProductKernel.id,
        REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
        pools.pe.path: pe,
        pools.simd.path: simd,
        pools.compute_pumping.path: pumping,
    }


def _result(operation: DecomposedMvauDataflowOp) -> NetworkRef:
    result = operation.resolve_dataflow(_context()).result
    assert isinstance(result, NetworkRef)
    return result


def _network(operation: DecomposedMvauDataflowOp) -> DataflowNetwork:
    return _result(operation).network


def _committed(
    model: ModelWrapper, *, pe: int = 2, simd: int = 2, pumping: bool = False
) -> DecomposedMvauDataflowOp:
    operation = _wrapped(model)
    operation.initialize_dataflow_scope_id()
    # Raises DataflowOpError with findings if anything is rejected.
    operation.commit_dataflow_assignments(_context(), _choices(pe=pe, simd=simd, pumping=pumping))
    return operation


# -- the forcing case --------------------------------------------------------


def test_the_operation_selects_both_kernels_and_returns_their_network() -> None:
    model = _model()
    operation = _committed(model)

    resolved = operation.resolve_dataflow(_context())
    assert isinstance(resolved.result, NetworkRef)

    network = resolved.result.network
    assert {node.id for node in network.nodes} == {REPLAY_NODE, DOT_PRODUCT_NODE}
    assert [edge.id for edge in network.edges] == [ACTIVATION_EDGE]
    assert list(validate_network(network)) == []
    for node in network.nodes:
        assert list(validate_region(node.region)) == []
        assert isinstance(node.region, DataflowRegion)


def test_the_operation_projects_its_problem_from_the_live_graph() -> None:
    """No geometry is stored on the node; changing a shape changes the problem."""

    model = _model(repetitions=4)
    problem = _wrapped(model).problem_instance(_context())
    assert problem[MVAUProblemPaths.REPETITIONS] == 4
    assert problem[MVAUProblemPaths.MATRIX_WIDTH] == MATRIX_WIDTH
    assert problem[MVAUProblemPaths.MATRIX_HEIGHT] == MATRIX_HEIGHT

    attribute_names = {item.name for item in model.graph.node[0].attribute}
    assert not attribute_names & {"MW", "MH", "inputDataType", "weightDataType"}

    other = _model(repetitions=2)
    assert _wrapped(other).problem_instance(_context())[MVAUProblemPaths.REPETITIONS] == 2


def test_the_boundary_matches_what_the_source_tensors_require() -> None:
    model = _model()
    operation = _committed(model)
    network = _network(operation)
    boundaries = {item.id: item for item in network.boundaries}

    assert set(boundaries) == {"activation", "weight", "output"}
    assert boundaries["activation"].endpoint.node_id == REPLAY_NODE
    assert boundaries["weight"].endpoint.node_id == DOT_PRODUCT_NODE
    assert boundaries["output"].endpoint.node_id == DOT_PRODUCT_NODE


# -- source association ------------------------------------------------------


def test_the_source_tensors_are_associated_with_the_nodes_that_carry_them() -> None:
    model = _model()
    operation = _committed(model)
    association = _result(operation).source_association

    owners = {item.role: item.destination.owner_id for item in association.operands}
    assert owners == {
        "activation": REPLAY_NODE,
        "weight": DOT_PRODUCT_NODE,
        "output": DOT_PRODUCT_NODE,
    }
    tensors = {item.role: item.source_operand_id for item in association.operands}
    assert tensors == {"activation": "activation", "weight": "weights", "output": "output"}
    assert association.source_node_id == f"{NODE_ID}_scope"
    assert association.compute_kernel_id == DotProductKernel.id


def test_the_association_names_the_operands_the_regions_actually_declare() -> None:
    """An association that points at a name no Region has is a broken promise."""

    model = _model()
    operation = _committed(model)
    result = _result(operation)
    by_node = {node.id: node.region for node in result.network.nodes}

    for operand in result.source_association.operands:
        destination = operand.destination
        assert isinstance(destination, SemanticOperandDestination)
        region = by_node[destination.owner_id]
        declared = {interface.port.operand.id for interface in region.interfaces}
        assert destination.operand_id in declared, operand


# -- persistence -------------------------------------------------------------


def test_the_selection_survives_a_save_and_reload(tmp_path: Path) -> None:
    """Both Kernel choices and the folding come back from the node attributes."""

    model = _model()
    _committed(model, pe=2, simd=4)
    path = tmp_path / "decomposed.onnx"
    model.save(str(path))

    reloaded = _wrapped(ModelWrapper(str(path)))
    assignments = reloaded.read_assignments()
    pools = MVAU_DECOMPOSED_POOLS
    assert assignments[DOT_PRODUCT_SELECTION.paths.kernel] == DotProductKernel.id
    assert assignments[REPLAY_SELECTION.paths.kernel] == ActivationReplayKernel.id
    assert assignments[pools.pe.path] == 2
    assert assignments[pools.simd.path] == 4

    result = _result(reloaded)
    assert {node.id for node in result.network.nodes} == {REPLAY_NODE, DOT_PRODUCT_NODE}


def test_a_reloaded_network_equals_the_one_that_was_saved(tmp_path: Path) -> None:
    model = _model()
    _committed(model)
    before = _network(_wrapped(model))
    path = tmp_path / "decomposed.onnx"
    model.save(str(path))

    after = _network(_wrapped(ModelWrapper(str(path))))
    assert before == after


def test_the_persisted_attributes_are_named_for_what_they_choose() -> None:
    model = _model()
    _committed(model)
    names = {item.name for item in model.graph.node[0].attribute}

    assert {
        "dataflow_dot_product_kernel",
        "dataflow_replay_kernel",
        "dataflow_dot_product_pe",
        "dataflow_dot_product_simd",
    } <= names


def test_changing_the_graph_invalidates_the_saved_selection() -> None:
    """A fingerprint over the projected problem, not over the node attributes."""

    model = _model()
    operation = _committed(model)
    before = operation.problem_instance(_context())

    model.set_tensor_shape("activation", [2, MATRIX_WIDTH])
    model.set_tensor_shape("output", [2, MATRIX_HEIGHT])
    after = _wrapped(model).problem_instance(_context())

    assert before[MVAUProblemPaths.REPETITIONS] != after[MVAUProblemPaths.REPETITIONS]


# -- the boundary is the fused one -------------------------------------------


@pytest.mark.parametrize(("pe", "simd"), [(2, 2), (1, 4), (4, 4), (4, 1)])
def test_every_folding_assembles_a_valid_network(pe: int, simd: int) -> None:
    model = _model()
    operation = _committed(model, pe=pe, simd=simd)
    network = _network(operation)

    assert list(validate_network(network)) == []
    assert len(network.nodes) == 2


def test_the_op_declares_exactly_the_two_pools() -> None:
    assert DecomposedMvauDataflowOp.kernel_selections() == (
        DOT_PRODUCT_SELECTION,
        REPLAY_SELECTION,
    )
