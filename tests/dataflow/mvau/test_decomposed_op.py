# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4's forcing case, through the real MVAU operation.

The Network constructor and its tests show the two Regions assemble.  This
shows ``MvauDataflowOp`` doing it: projecting its problem from a live
``ModelWrapper``, selecting the decomposed compute member and its replay
Kernel, returning a ``NetworkRef``, associating the source tensors with the two
nodes that now carry them, and reconstituting all of that after a save and
reload.

There is one MVAU operation and one ONNX op type.  Choosing the decomposed
implementation is a Kernel selection inside its design space, the same kind of
choice as picking any other compute member -- not a different node type and not
a second axis beside the Kernels.
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
)
from finn.dataflow.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.input_supply import EXTERNAL_SUPPLY
from finn.dataflow.design import (
    Absent,
    ConstraintAssessment,
    Decided,
    Engine,
    QualifiedPath,
)
from finn.dataflow.mvau_problem import MVAUProblemPaths
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import validate_network
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.op import DataflowOpError
from finn.dataflow.ops.mvau import MVAUDataflowOpPaths, NetworkRef, SemanticOperandDestination
from finn.dataflow.mvau.compute_kernels import (
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp
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
        "MvauDataflowOp",
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


def _wrapped(model: ModelWrapper) -> MvauDataflowOp:
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    return operation


def _choices(*, pe: int = 2, simd: int = 2, pumping: bool = False) -> dict[QualifiedPath, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    return {
        assembly.inventory.design_path: DotProductDesign.id,
        assembly.dot_product.pe.path: pe,
        assembly.dot_product.simd.path: simd,
        assembly.compute_pumping.path: pumping,
        assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
    }


def _batch_interleaved_choices() -> dict[QualifiedPath, object]:
    assembly = MVAU_DESIGN_INVENTORY
    assert assembly.inventory.design_path is not None
    return {
        assembly.inventory.design_path: "batch_interleaved",
        assembly.batch_interleaved.pe.path: 2,
        assembly.batch_interleaved.simd.path: 2,
        assembly.batch_interleaved.interleave.path: 2,
        assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
    }


def _result(operation: MvauDataflowOp) -> NetworkRef:
    result = operation.resolve_dataflow(_context()).result
    assert isinstance(result, NetworkRef)
    return result


def _network(operation: MvauDataflowOp) -> DataflowNetwork:
    return _result(operation).network


def _committed(
    model: ModelWrapper, *, pe: int = 2, simd: int = 2, pumping: bool = False
) -> MvauDataflowOp:
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
    assert association.design_id == DotProductDesign.id
    assert association.compute_kernel_id == "dotp_axi"
    assert association.kernel_ids == ("dotp_axi", "replay_buffer")


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
    assert assignments[MVAUDataflowOpPaths.DESIGN] == DotProductDesign.id
    assert assignments[MVAU_DESIGN_INVENTORY.dot_product.pe.path] == 2
    assert assignments[MVAU_DESIGN_INVENTORY.dot_product.simd.path] == 4
    assert MVAU_COMPUTE_SELECTION.paths.kernel not in assignments
    assert MVAU_REPLAY_SELECTION.paths.kernel not in assignments

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
        "dataflow_design",
        "dataflow_weight_supply",
        "dataflow_dot_product_pe",
        "dataflow_dot_product_simd",
        "dataflow_dotp_axi_pumping",
    } <= names
    assert not {"dataflow_compute_kernel", "dataflow_replay_kernel"} & names


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


def test_dot_product_is_a_design_not_a_legacy_kernel_pool_member() -> None:
    assert MVAU_DESIGN_INVENTORY.inventory.design_ids == (
        "dot_product",
        "batch_interleaved",
    )
    assert MvauDataflowOp.kernel_selections() == ()


def test_batch_interleaved_resolves_to_a_singleton_network() -> None:
    model = _model()
    operation = _wrapped(model)
    operation.initialize_dataflow_scope_id()
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    operation.commit_dataflow_assignments(
        _context(),
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: "batch_interleaved",
            MVAU_DESIGN_INVENTORY.batch_interleaved.pe.path: 2,
            MVAU_DESIGN_INVENTORY.batch_interleaved.simd.path: 2,
            MVAU_DESIGN_INVENTORY.batch_interleaved.interleave.path: 2,
            MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    )
    result = operation.resolve_dataflow(_context()).result
    assert isinstance(result, NetworkRef)
    assert tuple(node.id for node in result.network.nodes) == ("compute",)


# -- the operation's own verdict ---------------------------------------------
#
# Validating the Network directly says the assembly is well formed.  It does
# not say the *operation* accepts it: its constraints are separate declarations
# that can disagree with what assembly produced, and did.  These ask the
# operation.


def _assessment(operation: MvauDataflowOp, name: str) -> ConstraintAssessment:
    return Engine().evaluate_constraint_set(operation.hydrate_dataflow_point(_context()), name)


@pytest.mark.parametrize("constraint_set", ["mvau_op_structural", "mvau_op_feasibility"])
def test_the_operation_accepts_the_decomposed_point(constraint_set: str) -> None:
    operation = _committed(_model())
    assessment = _assessment(operation, constraint_set)
    rejected = {
        str(path): answer for path, answer in assessment.answers.items() if answer == Decided(False)
    }
    assert rejected == {}
    assert assessment.verdict is True


def test_the_operation_is_structurally_ready() -> None:
    operation = _committed(_model())
    point = operation.hydrate_dataflow_point(_context())
    assert Engine().check_readiness(point, "mvau_op_structural").ready is True


def test_the_association_and_the_network_are_both_checked() -> None:
    """Neither constraint may quietly sit out on a decomposed point.

    ``source_association_valid`` once assumed ownership followed from the
    parameter topology alone and rejected the replay node; the Network check
    once applied only when a supplier had produced the Network.  Both are
    ``Absent`` failures rather than ``False`` ones, so a verdict alone would
    not have caught either.
    """

    assessment = _assessment(_committed(_model()), "mvau_op_structural")
    assert assessment.answers[MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED] == Decided(True)
    point = _committed(_model()).hydrate_dataflow_point(_context())
    assert isinstance(
        Engine().query_property(point, MVAUDataflowOpPaths.SOURCE_ASSOCIATION), Decided
    )


# -- replay is not optional --------------------------------------------------


def test_the_replay_kernel_cannot_be_declined() -> None:
    """Without it the dot product's expanded activation reaches the boundary.

    ``R x NF x SF`` beats where the source operation presents ``R x SF``.  That
    is a different contract, so ``none`` is not in this pool's domain when the
    compute is decomposed -- the choice is withheld, not defaulted.
    """

    decisions = set(MvauDataflowOp.validated_design_space().decisions)
    assert MVAU_REPLAY_SELECTION.paths.kernel not in decisions

    operation = _wrapped(_model())
    operation.initialize_dataflow_scope_id()
    choices = dict(_choices())
    choices[MVAU_REPLAY_SELECTION.paths.kernel] = NO_KERNEL
    with pytest.raises(DataflowOpError):
        operation.commit_dataflow_assignments(_context(), choices)


def test_the_replay_choice_does_not_apply_to_a_fused_member() -> None:
    """DotProduct declarations are inactive for BatchInterleavedDesign."""

    operation = _wrapped(_model())
    operation.initialize_dataflow_scope_id()
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    operation.commit_dataflow_assignments(_context(), _batch_interleaved_choices())
    point = operation.hydrate_dataflow_point(_context())
    assert isinstance(
        Engine().query_property(point, MVAU_DESIGN_INVENTORY.dot_product.replay_region.path),
        Absent,
    )
    assert Engine().evaluate_constraint_set(point, "mvau_op_structural").verdict is True


def test_structural_readiness_does_not_wait_on_a_physical_choice() -> None:
    """Pumping changes no beat, so a Region-complete point is structurally ready.

    This regressed the moment the physical decisions joined the structural
    profile: a point with every semantic choice committed and its Network fully
    derived reported ``ready=None`` purely because nobody had said whether to
    pump the datapath.  That defeats the separation the physical layer exists to
    draw -- structural readiness is a question about logical dataflow.
    """

    operation = _wrapped(_model())
    operation.initialize_dataflow_scope_id()
    semantic = {
        path: value
        for path, value in _choices().items()
        if path != MVAU_DESIGN_INVENTORY.compute_pumping.path
    }
    operation.commit_dataflow_assignments(_context(), semantic)
    resolved = operation.resolve_dataflow(_context())
    engine, point = resolved.engine, resolved.point

    assert engine.check_readiness(point, "mvau_op_structural").ready is True
    # ...and the artifact profile correctly still says no, for that exact reason.
    assert engine.check_readiness(point, "artifact_inputs").ready is not True

    committed = engine.commit_assignments(
        point, {MVAU_DESIGN_INVENTORY.compute_pumping.path: False}
    ).point
    assert engine.check_readiness(committed, "artifact_inputs").ready is True
