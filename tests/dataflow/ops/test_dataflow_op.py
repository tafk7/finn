# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U4: the operation is the root Space, and the graph is read exactly once.

Two operations run through the same tests wherever the question is generic --
binding, freezing, persistence, staleness, association -- because the claim
being checked is that the layer is not MVAU-shaped.  Where MVAU has a Design
alternative and a matrix and the replay op has neither, the tests say so
separately rather than pretending the difference away.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.model.declarations import (
    AuthoringError,
    ConstraintGroup,
    Problem,
    constraint,
    declared_members,
)
from finn.dataflow.model.occurrence import ProjectionAssessment
from finn.dataflow.ops.association import (
    BoundaryDestination,
    CoordinateMapping,
    RegionStateDestination,
    StreamDestination,
)
from finn.dataflow.ops.base import (
    DATAFLOW_DOMAIN,
    FAMILY_ATTRIBUTE,
    FAMILY_VERSION_ATTRIBUTE,
    FINGERPRINT_ATTRIBUTE,
    SCOPE_ID_ATTRIBUTE,
    DataflowOp,
    DataflowOpError,
    source_declarations,
)
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.supplied_dot_product import (
    SuppliedDotProductDesign,
    WeightSupply,
)
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import (
    AssignDataflowScopeIds,
    CommitmentStage,
    apply_graph_effects,
    assign_dataflow_scope_ids,
)
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.schema import InputTensor, OutputTensor
from finn.dataflow.ops.state import (
    STATE_ATTRIBUTE,
    DecodeError,
    decode_dataflow_state,
    decode_state,
    format_dataflow_state,
)
from finn.dataflow.ops.source import SourceError


@dataclass(frozen=True)
class Build:
    """The build facts the generic boundary reads.  FINN's config satisfies it."""

    synth_clk_period_ns: float = 4.0
    target_dsp: DspBlock = DspBlock.DSP58


def _tensor(name: str, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _mvau_model(
    *,
    repetitions: int = 2,
    matrix_width: int = 8,
    matrix_height: int = 4,
    initializer: bool = True,
    narrow: bool = False,
) -> ModelWrapper:
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weight"],
        ["output"],
        domain=DATAFLOW_DOMAIN,
        name="mvau0",
        narrow_weights=int(narrow),
    )
    graph = helper.make_graph(
        [node],
        "mvau",
        [_tensor("activation", (repetitions, matrix_width))],
        [_tensor("output", (repetitions, matrix_height))],
        value_info=[_tensor("weight", (matrix_width, matrix_height))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("weight", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT32"])
    if initializer:
        model.set_initializer("weight", np.zeros((matrix_width, matrix_height), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _replay_model(*, repetitions: int = 2, matrix_width: int = 8, folds: int = 4) -> ModelWrapper:
    node = helper.make_node(
        "ActivationReplayOp",
        ["activation"],
        ["expanded"],
        domain=DATAFLOW_DOMAIN,
        name="replay0",
        neuron_folds=folds,
    )
    graph = helper.make_graph(
        [node],
        "replay",
        [_tensor("activation", (repetitions, matrix_width))],
        [_tensor("expanded", (repetitions * folds, matrix_width))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("expanded", DataType["INT8"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _chained_mvau_model() -> ModelWrapper:
    """An MVAU whose output is internal, so shape inference has something to do."""

    mvau = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weight"],
        ["hidden"],
        domain=DATAFLOW_DOMAIN,
        name="mvau0",
        narrow_weights=0,
    )
    tail = helper.make_node("Identity", ["hidden"], ["output"], name="tail")
    graph = helper.make_graph(
        [mvau, tail],
        "mvau",
        [_tensor("activation", (2, 8))],
        [_tensor("output", (2, 4))],
        value_info=[_tensor("weight", (8, 4))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    for name, kind in (("activation", "INT8"), ("weight", "INT8")):
        model.set_tensor_datatype(name, DataType[kind])
    model.set_initializer("weight", np.zeros((8, 4), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _chained_replay_model() -> ModelWrapper:
    replay = helper.make_node(
        "ActivationReplayOp",
        ["activation"],
        ["hidden"],
        domain=DATAFLOW_DOMAIN,
        name="replay0",
        neuron_folds=4,
    )
    tail = helper.make_node("Identity", ["hidden"], ["expanded"], name="tail")
    graph = helper.make_graph(
        [replay, tail],
        "replay",
        [_tensor("activation", (2, 8))],
        [_tensor("expanded", (8, 8))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _unbound(model: ModelWrapper, name: str) -> DataflowOp:
    node = next(item for item in model.graph.node if item.name == name)
    operation = model.get_customop_wrapper(node)
    assert isinstance(operation, DataflowOp)
    return operation


def _configured_mvau(
    model: ModelWrapper | None = None,
    *,
    design: str = "dot_product",
    supply: WeightSupply | None = None,
    pe: int = 2,
    simd: int = 2,
    pumped: bool = False,
) -> tuple[ModelWrapper, MvauDataflowOp]:
    """Bind, choose, and record -- the whole lifecycle a caller actually runs."""

    model = model or _mvau_model()
    operation = _unbound(model, "mvau0")
    chosen = operation.bind(model, Build())
    chosen = chosen.design.select(design).root
    if design == "supplied":
        chosen = (
            chosen.design.alternative(design)
            .assign(SuppliedDotProductDesign.weight_supply, supply or WeightSupply.EXTERNAL)
            .root
        )
        chosen = (
            chosen.design.alternative(design)
            .compute.select("dotp_axi_embedded" if supply is WeightSupply.EMBEDDED else "dotp_axi")
            .root
        )
    for declaration, value in (
        (WeightedDotProductDesign.pe, pe),
        (WeightedDotProductDesign.simd, simd),
    ):
        chosen = chosen.design.alternative(design).assign(declaration, value).root
    kernel = chosen.design.alternative(design).kernel("compute")
    assert isinstance(kernel, Decided)
    chosen = kernel.value.assign(DotpAxiKernel.compute_pumping, pumped).root
    committed = chosen.commit(model, Build())
    assert isinstance(committed, MvauDataflowOp)
    return model, committed


def _configured_replay(
    model: ModelWrapper | None = None, *, pe: int = 2, simd: int = 4
) -> tuple[ModelWrapper, ActivationReplayOp]:
    model = model or _replay_model()
    chosen = _unbound(model, "replay0").bind(model, Build())
    for declaration, value in (
        (ActivationReplayDesign.pe, pe),
        (ActivationReplayDesign.simd, simd),
    ):
        chosen = chosen.design.assign(declaration, value).root
    committed = chosen.commit(model, Build())
    assert isinstance(committed, ActivationReplayOp)
    return model, committed


# -- U4a: the operation is the root Space --------------------------------------


def test_the_operation_class_is_the_space() -> None:
    """No wrapper, no separate source Space, no ``.occurrence()``."""

    declarations = dict(declared_members(MvauDataflowOp))

    assert isinstance(declarations["activation"], Problem)
    assert isinstance(declarations["weight"], Problem)
    assert "design" in declarations
    assert not hasattr(MvauDataflowOp, "source_space")
    assert not hasattr(MvauDataflowOp, "occurrence")


def test_qonnx_constructs_the_unbound_instance_through_its_own_registry() -> None:
    from qonnx.custom_op.registry import getCustomOp  # noqa: PLC0415 - the bare path

    model = _mvau_model()
    bare = getCustomOp(model.graph.node[0])

    assert isinstance(bare, MvauDataflowOp)
    assert not bare.is_bound


def test_wants_model_is_a_class_attribute_and_attach_is_the_qonnx_protocol() -> None:
    """A method here would be accidentally truthy for an operation meaning no."""

    assert MvauDataflowOp.wants_model is True
    assert not callable(MvauDataflowOp.__dict__.get("wants_model", True))

    model = _mvau_model()
    operation = _unbound(model, "mvau0")
    before = model.graph.node[0].SerializeToString(deterministic=True)

    assert operation.attach_model(model) is operation
    assert model.graph.node[0].SerializeToString(deterministic=True) == before


def test_binding_returns_the_same_concrete_class_attached() -> None:
    model = _mvau_model()

    bound = _unbound(model, "mvau0").bind(model, Build())

    assert type(bound) is MvauDataflowOp
    assert bound.is_bound
    assert bound.root is bound


def test_a_successor_carries_the_same_frozen_binding() -> None:
    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    successor = bound.design.select("dot_product").root

    assert type(successor) is MvauDataflowOp
    assert successor is not bound
    assert successor.binding is bound.binding
    assert successor.problem_fingerprint == bound.problem_fingerprint


def test_a_child_design_and_kernel_cannot_reach_the_binding() -> None:
    """The capability boundary the root factory exists to keep."""

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())
    chosen = bound.design.select("dot_product").root

    design = chosen.design.alternative("dot_product")
    kernel = design.kernel("compute")

    assert not isinstance(design, DataflowOp)
    assert not hasattr(design, "binding")
    assert isinstance(kernel, Decided)
    assert not hasattr(kernel.value, "binding")


def test_a_live_node_edit_does_not_change_a_bound_answer() -> None:
    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())
    fingerprint = bound.problem_fingerprint

    model.set_tensor_datatype("activation", DataType["INT4"])
    model.graph.node[0].name = "renamed"

    assert bound.problem_fingerprint == fingerprint
    assert bound.source.node_name == "mvau0"
    assert bound.onnx_node.name == "mvau0"


def test_rebinding_observes_the_edit() -> None:
    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())
    fingerprint = bound.problem_fingerprint

    model.set_tensor_datatype("activation", DataType["INT4"])

    assert bound.rebind(model, Build()).problem_fingerprint != fingerprint


def test_a_missing_shape_or_datatype_is_a_source_refusal() -> None:
    model = _mvau_model()
    model.graph.node[0].input[1] = "unknown_tensor"

    with pytest.raises(SourceError, match="has no shape"):
        _unbound(model, "mvau0").bind(model, Build())


def test_binding_without_a_scope_id_names_the_upgrade_transaction() -> None:
    """Identity belongs to whoever constructs the node, never to a read."""

    model = _mvau_model()
    node = model.graph.node[0]
    kept = [item for item in node.attribute if item.name != SCOPE_ID_ATTRIBUTE]
    del node.attribute[:]
    node.attribute.extend(kept)

    operation = _unbound(model, "mvau0")

    assert operation.recorded_scope_id() is None
    with pytest.raises(SourceError, match="AssignDataflowScopeIds"):
        operation.bind(model, Build())


def test_the_two_operations_read_entirely_different_operand_sets() -> None:
    mvau = _unbound(_mvau_model(), "mvau0")
    replay = _unbound(_replay_model(), "replay0")

    assert [name for name, _ in source_declarations(type(mvau))] == [
        "activation",
        "weight",
        "output",
        "narrow_weights",
        "accumulator_type",
        "target_dsp",
        "clock_period_ns",
    ]
    assert [name for name, _ in source_declarations(type(replay))] == [
        "activation",
        "expanded",
        "neuron_folds",
    ]


# -- U4b: the compound source schema -------------------------------------------


def test_a_tensor_lowers_to_explicit_named_facets() -> None:
    declarations = dict(declared_members(MvauDataflowOp))

    assert "weight__shape" in declarations
    assert "weight__rank" in declarations
    assert "weight__datatype" in declarations
    assert "weight__initializer_present" in declarations
    # Only where the declaration asked for it.
    assert "weight__initializer_digest" in declarations
    assert "activation__initializer_digest" not in declarations
    # Only for an optional operand.
    assert "weight__present" not in declarations


def test_a_facet_accessor_returns_the_generated_declaration() -> None:
    declarations = dict(declared_members(MvauDataflowOp))

    assert MvauDataflowOp.__dict__["weight"].shape is declarations["weight__shape"]

    with pytest.raises(AttributeError, match="no facet"):
        _ = MvauDataflowOp.__dict__["weight"].nonsense


def test_an_optional_operand_gets_a_presence_facet_and_absent_facets_refuse() -> None:
    optional = InputTensor(index=3, optional=True)

    assert "present" in optional.facets
    assert optional.required is False

    refusal = optional.facets["shape"].evaluate(operand=object())
    assert refusal.findings[0].code == "source-operand-absent"


def test_an_output_observation_never_reaches_the_point() -> None:
    """An observation is not a Problem at all.

    A value that could reach a Decision domain or a constraint while staying
    out of the problem identity is a value whose recorded choices can be
    silently wrong -- so rather than an escape hatch on Problem, an output
    simply is not one.  MVAU therefore derives its output datatype from a
    declared accumulator attribute instead of reading it back off the tensor it
    writes.
    """

    assert isinstance(InputTensor(index=0), Problem)
    assert not isinstance(OutputTensor(index=0), Problem)
    assert "output" not in dict(declared_members(MvauDataflowOp))
    assert isinstance(dict(declared_members(MvauDataflowOp))["accumulator_type"], Problem)

    model = _mvau_model()
    before = _unbound(model, "mvau0").bind(model, Build()).problem_fingerprint
    model.set_tensor_datatype("output", DataType["INT16"])

    assert _unbound(model, "mvau0").bind(model, Build()).problem_fingerprint == before


def test_a_tensor_rename_does_not_change_the_problem_identity() -> None:
    """An operation does not depend on what its input is called."""

    model = _mvau_model()
    before = _unbound(model, "mvau0").bind(model, Build()).problem_fingerprint

    model.rename_tensor("weight", "the_matrix")

    assert _unbound(model, "mvau0").bind(model, Build()).problem_fingerprint == before


def test_the_initializer_values_move_the_identity_where_declared() -> None:
    left = _mvau_model()
    right = _mvau_model()
    right.set_initializer("weight", np.ones((8, 4), dtype=np.float32))

    assert (
        _unbound(left, "mvau0").bind(left, Build()).problem_fingerprint
        != _unbound(right, "mvau0").bind(right, Build()).problem_fingerprint
    )


def test_a_generated_facet_name_may_not_be_shadowed() -> None:
    with pytest.raises(AuthoringError, match="collides"):

        class Shadowing(DataflowOp):
            family = "test.shadowing"
            weight = InputTensor(index=0)
            weight__shape = InputTensor(index=1)


# -- U4c: the projections ------------------------------------------------------


def test_the_projection_is_a_full_assessment_not_a_bare_answer() -> None:
    """Readiness, validity and availability stay three questions."""

    _model, operation = _configured_mvau()

    assessment = operation.dataflow

    assert isinstance(assessment, ProjectionAssessment)
    assert assessment.readiness.ready is True
    assert all(item.verdict is True for item in assessment.constraints)
    assert isinstance(assessment.accepted_answer, Decided)
    assert operation.network == assessment.accepted_answer


def test_the_mvau_source_projects_folding_facts_from_its_tensors() -> None:
    _model, operation = _configured_mvau()

    network = operation.network

    assert isinstance(network, Decided), network
    assert {node.id for node in network.value.nodes} == {"replay", "compute"}
    assert {item.id for item in network.value.boundaries} == {"activation", "weight", "output"}


def test_the_replay_source_projects_one_node_and_no_selector() -> None:
    _model, operation = _configured_replay()

    network = operation.network

    assert isinstance(network, Decided), network
    assert {node.id for node in network.value.nodes} == {"replay"}
    assert network.value.edges == ()
    assert {item.id for item in network.value.boundaries} == {"activation", "expanded"}


def test_an_unselected_design_is_unresolved_and_not_an_exception() -> None:
    """A structural choice not yet made is a point state, not a defect."""

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    assessment = bound.dataflow

    assert isinstance(assessment, ProjectionAssessment)
    assert isinstance(assessment.accepted_answer, Unresolved)
    assert assessment.readiness.ready is None
    codes = {finding.code for finding in assessment.accepted_answer.findings}
    assert "projection-alternative-unselected" in codes


def test_the_operations_own_constraints_gate_its_projection() -> None:
    """source_accepts must not be compiled and consulted by nothing."""

    model = _mvau_model()
    # A rank-1 matrix: mathematically not a matrix-vector multiplication.
    model.set_tensor_shape("weight", [8])
    model.set_initializer("weight", np.zeros((8,), dtype=np.float32))
    bound = _unbound(model, "mvau0").bind(model, Build())

    assessment = bound.dataflow

    codes = {
        finding.code
        for answer in (assessment.accepted_answer,)
        for finding in getattr(answer, "findings", ())
    }
    assert any(code.startswith("mvau-weight-not-a-matrix") for code in codes), codes


def test_a_malformed_shape_gives_a_finding_not_a_crash() -> None:
    """A Derived cannot assume a sibling constraint ran first."""

    model = _mvau_model()
    model.set_tensor_shape("weight", [8])
    model.set_initializer("weight", np.zeros((8,), dtype=np.float32))
    bound = _unbound(model, "mvau0").bind(model, Build())

    answer = bound.answer(MvauDataflowOp.matrix_height)

    assert not isinstance(answer, Decided)
    assert {finding.code for finding in answer.findings} >= {"mvau-weight-not-a-matrix"}


def test_a_repairable_output_annotation_is_a_difference_not_a_rejection() -> None:
    """Otherwise the repair can never be committed and the node stays wrong."""

    model = _mvau_model()
    model.set_tensor_shape("output", [2, 99])
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    chosen = chosen.design.alternative("dot_product").assign(WeightedDotProductDesign.pe, 2).root
    chosen = chosen.design.alternative("dot_product").assign(WeightedDotProductDesign.simd, 2).root

    assert isinstance(chosen.dataflow.accepted_answer, Decided)
    assert any("output" in item for item in chosen.reconciliation())

    committed = chosen.commit(model, Build())

    assert tuple(model.get_tensor_shape("output")) == (2, 4)
    assert committed.reconciliation() == ()


def test_an_uncommitted_folding_leaves_the_network_unresolved() -> None:
    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root

    assert isinstance(chosen.network, Unresolved)


# -- U4d: persistence ----------------------------------------------------------


def test_choices_survive_a_save_and_reload_exactly(tmp_path: Path) -> None:
    model, operation = _configured_mvau(pe=2, simd=4)
    before = operation.network
    assert isinstance(before, Decided)

    path = tmp_path / "mvau.onnx"
    model.save(str(path))
    reloaded = ModelWrapper(str(path))
    restored = _unbound(reloaded, "mvau0").bind(reloaded, Build())

    assert dict(restored.recorded()) == dict(operation.recorded())
    after = restored.network
    assert isinstance(after, Decided)
    assert after.value == before.value


def test_a_structural_choice_can_be_changed_without_touching_the_node() -> None:
    """bind, commit supplied+embedded, rebind, switch, commit, rebind -- no clearing.

    An immutable point does not rebase a committed selector, so switching
    alternatives means a point that never had the first one.  ``reconstruct()``
    is that point: the same frozen binding and therefore the same problem
    identity, with no assignments.  Nothing reaches into the node.
    """

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())
    supplied = bound.design.select("supplied").root
    supplied = (
        supplied.design.alternative("supplied")
        .assign(SuppliedDotProductDesign.weight_supply, WeightSupply.EMBEDDED)
        .root
    )
    committed = supplied.commit(model, Build())
    assert dict(committed.recorded())["design.supplied.weight_supply"] is WeightSupply.EMBEDDED

    reloaded = _unbound(model, "mvau0").bind(model, Build())
    switched = reloaded.reconstruct().design.select("dot_product").root
    final = switched.commit(model, Build())

    assert dict(final.recorded())["design.case"] == "dot_product"
    # Nothing belonging to the alternative left behind survived.
    assert "design.supplied.weight_supply" not in dict(final.recorded())
    assert "design.supplied.weight_supply" not in dict(_unbound(model, "mvau0").recorded())
    # And it rebinds cleanly: nothing left over refuses to replay.
    assert _unbound(model, "mvau0").bind(model, Build()).recorded()["design.case"] == (
        "dot_product"
    )


def test_a_partial_point_may_be_saved(tmp_path: Path) -> None:
    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    chosen.commit(model, Build())

    path = tmp_path / "partial.onnx"
    model.save(str(path))
    reloaded = ModelWrapper(str(path))
    restored = _unbound(reloaded, "mvau0").bind(reloaded, Build())

    # A singleton segment generates no selector, so there is nothing to record
    # for it -- adding a second candidate later adds a path, it does not rename
    # an existing one.
    assert set(restored.recorded()) == {"design.case"}
    assert isinstance(restored.network, Unresolved)


def test_a_refused_point_may_not_be_saved() -> None:
    """Unresolved is a legitimate thing to record; refused is not."""

    model = _mvau_model(matrix_height=4)
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root

    # 3 does not divide a matrix height of 4, so the assignment itself refuses.
    with pytest.raises(Exception):  # noqa: B017 - RequestError from the engine
        chosen.design.alternative("dot_product").assign(WeightedDotProductDesign.pe, 3)


def test_a_stale_plan_is_refused_rather_than_applied() -> None:
    model = _mvau_model()
    _model, operation = _configured_mvau(model)
    effects = operation.graph_effects()

    model.graph.node[0].name = "renamed"

    with pytest.raises(DataflowOpError, match="changed since this change was planned"):
        apply_graph_effects(model, effects)


def test_a_plan_addressed_to_another_graph_is_refused() -> None:
    _left, operation = _configured_mvau()
    effects = operation.graph_effects()
    other = _mvau_model()

    with pytest.raises(DataflowOpError, match="no node in this graph carries"):
        apply_graph_effects(other, effects)


def test_the_design_state_is_one_canonical_document() -> None:
    """One authority, and no flat copy of any value that could disagree with it."""

    model, operation = _configured_mvau(pe=2, simd=4)
    node = model.graph.node[0]

    names = {item.name for item in node.attribute}
    assert {SCOPE_ID_ATTRIBUTE, STATE_ATTRIBUTE} <= names
    # The source-semantic attributes stay their own thing; they define the
    # mathematical operation, not the implementation chosen for it.
    assert "narrow_weights" in names
    # No flat copy of a Decision.
    assert not ({"PE", "SIMD", "dataflow_design", "weight_supply"} & names)
    # Family, version and fingerprint moved into the document, so they stop
    # having two persisted homes.
    assert not ({FAMILY_ATTRIBUTE, FAMILY_VERSION_ATTRIBUTE, FINGERPRINT_ATTRIBUTE} & names)

    state = decode_dataflow_state(node)
    assert state is not None
    assert state.family == "finn.dataflow.mvau"
    assert state.problem_fingerprint == operation.problem_fingerprint
    assert state.assignments["design.dot_product.pe"].value == 2
    assert state.assignments["design.dot_product.simd"].value == 4


def test_the_document_is_canonical_and_readable() -> None:
    """Two equal points produce equal bytes, and a human can read them."""

    left = _configured_mvau(pe=2, simd=4)[0]
    right = _configured_mvau(_mvau_model(), pe=2, simd=4)[0]

    def document(model: ModelWrapper) -> str:
        return next(
            item.s.decode("utf-8")
            for item in model.graph.node[0].attribute
            if item.name == STATE_ATTRIBUTE
        )

    assert document(left) == document(right)
    assert '"schema":"finn.dataflow.state/1"' in document(left)

    rendered = format_dataflow_state(left.graph.node[0])
    assert "design.dot_product.pe = 2" in rendered
    assert "finn.dataflow.mvau" in rendered


def test_a_state_document_this_build_does_not_know_is_refused() -> None:
    """Not reinterpreted: an unknown schema is a refusal, not a best effort."""

    model, _operation = _configured_mvau()
    node = model.graph.node[0]
    for item in node.attribute:
        if item.name == STATE_ATTRIBUTE:
            item.s = b'{"schema":"finn.dataflow.state/99","assignments":{}}'

    with pytest.raises(DataflowOpError, match="this build writes"):
        _unbound(model, "mvau0").bind(model, Build())


def test_reconstruct_keeps_the_identity_and_drops_every_choice() -> None:
    model, operation = _configured_mvau()
    scope = operation.binding.node_identity

    fresh = operation.reconstruct()

    assert fresh.recorded() == {}
    assert fresh.binding.node_identity == scope
    assert fresh.problem_fingerprint == operation.problem_fingerprint


def test_reconstruct_refuses_an_arbitrary_problem() -> None:
    """Its Problem values and its frozen binding would describe different nodes."""

    _model, operation = _configured_mvau()

    with pytest.raises(DataflowOpError, match="not from an arbitrary problem mapping"):
        operation.reconstruct({})


def test_a_changed_problem_makes_the_recorded_choices_stale() -> None:
    model, operation = _configured_mvau()
    assert not operation.is_stale((model, Build()))

    model.set_tensor_datatype("activation", DataType["INT4"])

    assert operation.is_stale((model, Build()))
    with pytest.raises(DataflowOpError, match="different problem"):
        _unbound(model, "mvau0").bind(model, Build())


def test_a_saved_family_that_this_build_does_not_offer_is_refused() -> None:
    model, _operation = _configured_mvau()
    node = model.graph.node[0]
    for item in node.attribute:
        if item.name == STATE_ATTRIBUTE:
            document = json.loads(item.s.decode("utf-8"))
            document["family"] = "finn.dataflow.something_else"
            item.s = json.dumps(document, sort_keys=True).encode("utf-8")

    with pytest.raises(DataflowOpError, match="stores choices for family"):
        _unbound(model, "mvau0").bind(model, Build())


def test_a_partly_applied_transaction_restores_the_whole_model() -> None:
    """Restoring the node alone would leave tensor metadata half-written."""

    model, operation = _configured_mvau()
    effects = operation.graph_effects()

    class Exploding:
        """A datatype whose write fails *after* the node has been rewritten."""

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("this tensor write fails")

    broken = replace(effects, tensor_datatypes={"output": Exploding()})
    before = model.model.SerializeToString(deterministic=True)

    with pytest.raises(Exception):  # noqa: B017 - whatever the writer raises
        apply_graph_effects(model, broken)

    assert model.model.SerializeToString(deterministic=True) == before


def test_commit_returns_a_bound_operation_over_the_committed_graph() -> None:
    """The lifecycle continues across the mutation boundary."""

    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root

    committed = chosen.commit(model, Build())

    assert type(committed) is MvauDataflowOp
    assert committed.is_bound
    assert dict(committed.recorded())["design.case"] == "dot_product"


def test_the_commitment_stage_is_recorded_with_the_plan() -> None:
    model, operation = _configured_mvau()

    effects = operation.graph_effects(require=CommitmentStage.DATAFLOW)

    assert effects.commitment_stage is CommitmentStage.DATAFLOW
    assert effects.expected_source_fingerprint == operation.problem_fingerprint
    # Recorded, and named for what it actually promises: no required projection
    # was finally rejected.  Not that every Decision was made.
    state = decode_dataflow_state(model.graph.node[0])
    assert state is not None and state.commitment_stage == "dataflow"


# -- U4e: source association ---------------------------------------------------


def test_every_source_operand_is_associated_with_where_its_data_crosses() -> None:
    _model, operation = _configured_mvau()

    answer = operation.association

    assert isinstance(answer, Decided)
    association = answer.value
    assert association.family == "finn.dataflow.mvau"
    assert {item.operand for item in association.operands} == {"activation", "weight", "output"}
    activation = association.operand("activation")
    assert activation.boundary == "activation"
    assert (activation.node_id, activation.port_id) == ("replay", "activation_in")
    assert activation.correspondence is CoordinateMapping.FLATTEN_LEADING
    assert isinstance(activation.destination, BoundaryDestination)


def test_an_association_names_no_kernel_component_or_artifact() -> None:
    _model, operation = _configured_mvau()
    answer = operation.association
    assert isinstance(answer, Decided)

    rendered = repr(answer.value)

    for forbidden in ("dotp_axi", "replay_buffer", "ComponentABI", "Derivation", "ArtifactRef"):
        assert forbidden not in rendered


def test_a_decoupled_matrix_is_traffic_and_an_embedded_one_is_state() -> None:
    """Three destinations, because they are three different facts.

    Reporting embedded state as a port named ``"embedded"`` names a port that
    does not exist: a consumer resolving it finds nothing, and the empty shape
    that came with it reads as a zero-element tensor.
    """

    external = _configured_mvau(design="supplied", supply=WeightSupply.EXTERNAL)[1]
    decoupled = _configured_mvau(_mvau_model(), design="supplied", supply=WeightSupply.DECOUPLED)[1]
    embedded = _configured_mvau(_mvau_model(), design="supplied", supply=WeightSupply.EMBEDDED)[1]

    outside = external.association
    inside = decoupled.association
    baked = embedded.association
    assert isinstance(outside, Decided) and isinstance(inside, Decided)
    assert isinstance(baked, Decided)

    assert isinstance(outside.value.operand("weight").destination, BoundaryDestination)
    assert isinstance(inside.value.operand("weight").destination, StreamDestination)
    assert isinstance(baked.value.operand("weight").destination, RegionStateDestination)

    assert inside.value.operand("weight").boundary is None
    assert inside.value.operand("weight").node_id == "compute"
    assert inside.value.operand("weight").port_id == "weight"
    # No port at all, and no fabricated shape to go with one.
    assert baked.value.operand("weight").port_id is None
    assert baked.value.operand("weight").selected_shape is None


def test_a_physical_choice_does_not_change_the_association() -> None:
    plain = _configured_mvau(simd=4)[1].association
    pumped = _configured_mvau(_mvau_model(), simd=4, pumped=True)[1].association

    assert isinstance(plain, Decided) and isinstance(pumped, Decided)
    assert [item.boundary for item in plain.value.operands] == [
        item.boundary for item in pumped.value.operands
    ]
    assert [item.node_id for item in plain.value.operands] == [
        item.node_id for item in pumped.value.operands
    ]


def test_the_second_operation_associates_its_own_two_operands() -> None:
    _model, operation = _configured_replay()

    answer = operation.association

    assert isinstance(answer, Decided)
    assert {item.operand for item in answer.value.operands} == {"activation", "expanded"}
    assert answer.value.family == "finn.dataflow.activation_replay"
    assert answer.value.operand("expanded").node_id == "replay"


def test_the_scope_id_survives_a_rename_of_the_node() -> None:
    model, operation = _configured_mvau()
    scope = operation.binding.node_identity

    model.graph.node[0].name = "renamed"
    renamed = _unbound(model, "renamed").bind(model, Build())

    assert renamed.binding.node_identity == scope
    answer = renamed.association
    assert isinstance(answer, Decided)
    assert answer.value.scope_id == scope
    assert answer.value.source_node == "renamed"


# -- U4f: the two operations do not share an implementation --------------------


def test_persistence_is_discovered_from_the_model_not_declared_by_the_operation() -> None:
    """Neither operation lists its own choices; both are walked from the point.

    A Decision added to a Design three levels down is persisted without anyone
    editing the operation, and two Decisions called the same thing in different
    subspaces cannot collide, because compiled paths cannot.
    """

    assert not hasattr(MvauDataflowOp, "attributes")
    assert not hasattr(ActivationReplayOp, "attributes")

    _model, replay = _configured_replay()
    _other, mvau = _configured_mvau()

    replay_paths = set(replay.recorded())
    mvau_paths = set(mvau.recorded())

    # No selector at all in the replay op: it has a fixed Subspace.
    assert replay_paths == {"design.pe", "design.simd"}
    # The MVAU op has two nested selectors and Decisions beneath them, all
    # named by compiled path rather than by a friendly alias.
    assert "design.case" in mvau_paths
    assert "design.dot_product.pe" in mvau_paths
    # Nested three levels down, inside the selected alternative's selected
    # candidate.  No alias could have named it, and nobody listed it.
    assert "design.dot_product.compute.dotp_axi.compute_pumping" in mvau_paths

    # And a segment with a real choice does contribute its selector.
    _third, supplied = _configured_mvau(
        _mvau_model(), design="supplied", supply=WeightSupply.EMBEDDED
    )
    assert "design.supplied.compute.kernel" in set(supplied.recorded())


def test_a_choice_the_model_no_longer_declares_is_a_refusal() -> None:
    """A renamed declaration is a schema change, not an assignment to discard."""

    model, _operation = _configured_mvau()
    node = model.graph.node[0]
    for item in node.attribute:
        if item.name == STATE_ATTRIBUTE:
            document = json.loads(item.s.decode("utf-8"))
            document["assignments"]["design.dot_product.renamed_away"] = {
                "codec": "dataflow.int@1",
                "value": 4,
            }
            item.s = json.dumps(document, sort_keys=True).encode("utf-8")

    with pytest.raises(DataflowOpError, match="has no declaration for"):
        _unbound(model, "mvau0").bind(model, Build())


def test_a_changed_codec_is_a_refusal_not_a_reinterpretation() -> None:
    model, _operation = _configured_mvau()
    node = model.graph.node[0]
    for item in node.attribute:
        if item.name == STATE_ATTRIBUTE:
            document = json.loads(item.s.decode("utf-8"))
            document["assignments"]["design.dot_product.pe"]["codec"] = "dataflow.int@99"
            item.s = json.dumps(document, sort_keys=True).encode("utf-8")

    with pytest.raises(DataflowOpError, match="changed encoding is not reinterpreted"):
        _unbound(model, "mvau0").bind(model, Build())


@pytest.mark.parametrize(
    "document",
    [
        '{"schema":"finn.dataflow.state/1"}',
        (
            '{"schema":"finn.dataflow.state/1","family":"f","family_version":"1",'
            '"problem_fingerprint":"x","commitment_stage":"nonsense","assignments":{}}'
        ),
        (
            '{"schema":"finn.dataflow.state/1","family":"f","family_version":"1",'
            '"problem_fingerprint":"x","commitment_stage":"dataflow","assignments":{},'
            '"surprise":1}'
        ),
        (
            '{"schema":"finn.dataflow.state/1","family":"f","family_version":"1",'
            '"problem_fingerprint":"x","commitment_stage":"dataflow",'
            '"assignments":{"a.b":2}}'
        ),
    ],
    ids=["incomplete", "unknown-stage", "unknown-member", "untagged-value"],
)
def test_state_decoding_is_strict(document: str) -> None:
    """A permissive reader turns a schema change into a silently lost design."""

    with pytest.raises(DecodeError):
        decode_state(document)


def test_both_operations_use_the_same_persistence_authority(tmp_path: Path) -> None:
    for build, name in ((_configured_mvau, "mvau0"), (_configured_replay, "replay0")):
        model, operation = build()
        path = tmp_path / f"{name}.onnx"
        model.save(str(path))
        reloaded = ModelWrapper(str(path))
        restored = _unbound(reloaded, name).bind(reloaded, Build())
        assert dict(restored.recorded()) == dict(operation.recorded())


def test_every_refusal_survives_python_o() -> None:
    """Transactionality must not be carried by an assert.

    ``python -O`` strips assertions, and a persistence layer whose "the node
    was not touched" guarantee lived in one would silently start writing
    half-committed graphs in exactly the configuration a production build uses.
    """

    script = (
        "from dataflow.ops.test_dataflow_op import Build, _configured_mvau, _mvau_model\n"
        "from finn.dataflow.ops.base import DataflowOpError\n"
        "from finn.dataflow.ops.persistence import apply_graph_effects\n"
        "model, operation = _configured_mvau()\n"
        "effects = operation.graph_effects()\n"
        "model.graph.node[0].name = 'renamed'\n"
        "before = model.graph.node[0].SerializeToString(deterministic=True)\n"
        "try:\n"
        "    apply_graph_effects(model, effects)\n"
        "except DataflowOpError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('a stale plan was applied')\n"
        "after = model.graph.node[0].SerializeToString(deterministic=True)\n"
        "raise SystemExit(0 if after == before else 'the node changed')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout


# -- U4g: addressing identity --------------------------------------------------


def test_normalization_assigns_a_missing_scope_id() -> None:
    model = _mvau_model()
    _strip_scope(model.graph.node[0])

    assigned = AssignDataflowScopeIds(DATAFLOW_DOMAIN).normalize(model)

    assert len(assigned) == 1
    assert _unbound(model, "mvau0").recorded_scope_id() == assigned[0]


def test_normalization_is_idempotent() -> None:
    model = _mvau_model()
    before = _unbound(model, "mvau0").recorded_scope_id()

    assert AssignDataflowScopeIds(DATAFLOW_DOMAIN).normalize(model) == ()
    assert _unbound(model, "mvau0").recorded_scope_id() == before


def test_a_cloned_node_is_given_a_distinct_identity() -> None:
    """Copying a NodeProto copies its scope attribute; two nodes must not share one.

    Worse than a missing id: a change addressed to a shared id finds two
    candidates, and a recorded choice is attributable to either.
    """

    model = _mvau_model()
    original = model.graph.node[0]
    clone = model.graph.node.add()
    clone.CopyFrom(original)
    clone.name = "mvau1"
    shared = _unbound(model, "mvau0").recorded_scope_id()

    assigned = AssignDataflowScopeIds(DATAFLOW_DOMAIN).normalize(model)

    assert len(assigned) == 1
    # The first in graph order keeps it; the clone is reallocated.
    assert _unbound(model, "mvau0").recorded_scope_id() == shared
    assert _unbound(model, "mvau1").recorded_scope_id() == assigned[0]
    assert _unbound(model, "mvau1").recorded_scope_id() != shared


def test_binding_refuses_a_graph_that_still_contains_duplicates() -> None:
    model = _mvau_model()
    clone = model.graph.node.add()
    clone.CopyFrom(model.graph.node[0])
    clone.name = "mvau1"

    with pytest.raises(SourceError, match="carry dataflow scope id"):
        _unbound(model, "mvau0").bind(model, Build())


def test_a_rename_preserves_identity() -> None:
    model = _mvau_model()
    before = _unbound(model, "mvau0").recorded_scope_id()

    model.graph.node[0].name = "renamed"

    assert AssignDataflowScopeIds(DATAFLOW_DOMAIN).normalize(model) == ()
    assert _unbound(model, "renamed").recorded_scope_id() == before


def test_rebinding_reads_the_live_node_not_the_frozen_copy() -> None:
    """The defect a datatype-only test cannot catch.

    ``bind`` on ``self`` would re-read the frozen copy, so the tensor names,
    the node attributes, the operand list and the node name would all be the
    ones captured at binding time.
    """

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    model.graph.node[0].name = "renamed"
    model.rename_tensor("activation", "a2")
    _set_attribute(model.graph.node[0], "narrow_weights", 1)

    fresh = bound.rebind(model, Build())

    assert fresh.source.node_name == "renamed"
    assert fresh.source.operand("activation").tensor == "a2"
    assert fresh.source.attributes["narrow_weights"] is True
    # The original is untouched: that is what freezing means.
    assert bound.source.node_name == "mvau0"
    assert bound.source.operand("activation").tensor == "activation"
    assert bound.source.attributes["narrow_weights"] is False


def test_rebinding_notices_a_changed_build_fact() -> None:
    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    fresh = bound.rebind(model, Build(synth_clk_period_ns=10.0))

    assert fresh.problem_fingerprint != bound.problem_fingerprint


def test_declared_operand_indices_must_be_contiguous() -> None:
    """An index is an index, not a hint about ordering."""

    with pytest.raises(AuthoringError, match="contiguous from zero"):

        class Sparse(DataflowOp):
            family = "test.sparse"
            first = InputTensor(index=0)
            third = InputTensor(index=2)
            out = OutputTensor(index=0)


def _strip_scope(node: Any) -> None:
    kept = [item for item in node.attribute if item.name != SCOPE_ID_ATTRIBUTE]
    del node.attribute[:]
    node.attribute.extend(kept)


def _set_attribute(node: Any, name: str, value: int) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


# -- U4h: the QONNX boundary ---------------------------------------------------


def test_qonnx_shape_inference_runs_on_the_unbound_wrapper() -> None:
    """Through QONNX's real transformation, with no build configuration.

    An output shape is a fact about the *operation*, not about the target it
    will be built for, so shape inference must not need a synthesis config --
    and QONNX has none to give.  The MVAU output is an internal tensor here,
    because a graph output's shape is fixed by the graph and inference would
    have nothing to decide.
    """

    from qonnx.transformation.infer_shapes import InferShapes  # noqa: PLC0415

    model = _chained_mvau_model()

    model = model.transform(InferShapes())

    assert tuple(model.get_tensor_shape("hidden")) == (2, 4)


def test_qonnx_datatype_inference_runs_on_the_unbound_wrapper() -> None:
    from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: PLC0415

    model = _mvau_model()
    model.set_tensor_datatype("output", DataType["FLOAT32"])

    model = model.transform(InferDataTypes())

    assert model.get_tensor_datatype("output") == DataType["INT32"]


def test_the_replay_operation_infers_through_the_same_passes() -> None:
    from qonnx.transformation.infer_shapes import InferShapes  # noqa: PLC0415

    model = _chained_replay_model()

    model = model.transform(InferShapes())

    assert tuple(model.get_tensor_shape("hidden")) == (8, 8)


def test_the_bound_and_unbound_paths_use_one_formula() -> None:
    model = _mvau_model()
    unbound = _unbound(model, "mvau0")
    bound = unbound.bind(model, Build())

    assert unbound.expected_for(unbound.source_snapshot(model)) == bound.expected_outputs()


# -- U4i: the frozen snapshot is actually frozen --------------------------------


def test_successors_do_not_share_one_mutable_node() -> None:
    """A frozen dataclass holding a live protobuf is not frozen."""

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())
    successor = bound.design.select("dot_product").root

    assert successor.onnx_node is not bound.onnx_node
    assert successor.onnx_node.SerializeToString(
        deterministic=True
    ) == bound.onnx_node.SerializeToString(deterministic=True)


def test_a_bound_occurrence_refuses_node_mutation() -> None:
    """The write would appear to succeed and change nothing that matters."""

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    with pytest.raises(DataflowOpError, match="frozen snapshot"):
        bound.set_nodeattr("narrow_weights", 1)

    # The unbound wrapper is an ordinary CustomOp and still writes.
    _unbound(model, "mvau0").set_nodeattr("narrow_weights", 1)
    assert bound.source.attributes["narrow_weights"] is False


def test_binding_resolves_the_node_from_the_supplied_model() -> None:
    """A wrapper from one graph must not read structure into another's facts."""

    left = _mvau_model()
    right = _mvau_model(matrix_width=16)
    # Give the second graph the first's identity, as a copy would.
    scope = _unbound(left, "mvau0").recorded_scope_id()
    for item in right.graph.node[0].attribute:
        if item.name == SCOPE_ID_ATTRIBUTE:
            item.s = scope.encode("utf-8")

    bound = _unbound(left, "mvau0").bind(right, Build())

    assert bound.source.operand("activation").shape == (2, 16)


# -- U4j: the normalization transformation in its real pipeline -----------------


def test_the_scope_transformation_runs_through_qonnx() -> None:
    model = _mvau_model()
    _strip_scope(model.graph.node[0])
    # A non-dataflow node, to prove it is left alone.
    other = model.graph.node.add()
    other.CopyFrom(model.graph.node[0])
    other.name = "elsewhere"
    other.domain = "some.other.domain"

    model = model.transform(AssignDataflowScopeIds(DATAFLOW_DOMAIN))

    assert _unbound(model, "mvau0").recorded_scope_id() is not None
    assert not [item for item in other.attribute if item.name == SCOPE_ID_ATTRIBUTE]

    # Idempotent: a second pass reports no modification and changes nothing.
    before = model.model.SerializeToString(deterministic=True)
    assert AssignDataflowScopeIds(DATAFLOW_DOMAIN).apply(model)[1] is False
    assert model.model.SerializeToString(deterministic=True) == before


def test_the_transformation_reports_that_it_modified_the_graph() -> None:
    model = _mvau_model()
    _strip_scope(model.graph.node[0])

    _returned, modified = AssignDataflowScopeIds(DATAFLOW_DOMAIN).apply(model)

    assert modified is True


# -- U4k: malformed extents ------------------------------------------------------


def test_a_zero_extent_gives_a_finding_not_a_division_by_zero() -> None:
    """A Derived cannot assume a sibling constraint ran first."""

    model = _mvau_model()
    model.set_tensor_shape("weight", [0, 4])
    model.set_initializer("weight", np.zeros((0, 4), dtype=np.float32))
    bound = _unbound(model, "mvau0").bind(model, Build())

    answer = bound.answer(MvauDataflowOp.repetitions)

    assert not isinstance(answer, Decided)
    assert {finding.code for finding in answer.findings} >= {"mvau-degenerate-extent"}


def test_a_zero_extent_replay_activation_gives_a_finding() -> None:
    model = _replay_model()
    model.set_tensor_shape("activation", [2, 0])
    bound = _unbound(model, "replay0").bind(model, Build())

    answer = bound.answer(ActivationReplayOp.repetitions)

    assert not isinstance(answer, Decided)
    assert {finding.code for finding in answer.findings} >= {"replay-degenerate-extent"}


# -- the commit boundary: one build, and it is the frozen one ------------------


def test_commit_needs_no_build_because_the_occurrence_already_froze_one() -> None:
    """The effects were derived from the frozen facts; so is the rebinding."""

    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root

    committed = chosen.commit(model)

    assert type(committed) is MvauDataflowOp
    assert committed.problem_fingerprint == chosen.problem_fingerprint
    assert dict(committed.recorded())["design.case"] == "dot_product"


def test_a_mismatched_build_at_commit_leaves_the_graph_unchanged() -> None:
    """Equivalence is proved before anything is written, not after.

    The failure this forbids is specific: apply effects derived from build A,
    then fail while rebinding under build B, and the graph is left holding
    choices whose author has already been told the operation failed.
    """

    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    before = model.model.SerializeToString(deterministic=True)

    with pytest.raises(DataflowOpError, match="clock_period_ns"):
        chosen.commit(model, Build(synth_clk_period_ns=10.0))

    assert model.model.SerializeToString(deterministic=True) == before


def test_an_equivalent_build_at_commit_is_accepted() -> None:
    """A caller that passes the same configuration is not being punished."""

    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root

    committed = chosen.commit(model, Build())

    assert dict(committed.recorded())["design.case"] == "dot_product"


def test_changing_the_build_context_is_a_rebinding_not_a_commit() -> None:
    """And it reads the graph rather than writing it."""

    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())
    before = model.model.SerializeToString(deterministic=True)

    fresh = bound.rebind(model, Build(synth_clk_period_ns=10.0))

    assert fresh.problem_fingerprint != bound.problem_fingerprint
    assert model.model.SerializeToString(deterministic=True) == before


def test_rebinding_without_a_build_reuses_the_frozen_facts() -> None:
    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    assert bound.rebind(model).problem_fingerprint == bound.problem_fingerprint


# -- constraint classification -------------------------------------------------


def test_an_operation_constraint_must_be_classified() -> None:
    """A constraint in no group is compiled, evaluated, and consulted by nothing."""

    with pytest.raises(AuthoringError, match="outside source_accepts"):

        class Unclassified(DataflowOp):
            family = "test.unclassified"
            activation = InputTensor(index=0)
            result = OutputTensor(index=0)

            @constraint(shape=activation.shape)
            def rank_is_two(*, shape: tuple[int, ...]) -> object:
                return len(shape) == 2


def test_a_classified_operation_constraint_is_accepted() -> None:
    class Classified(DataflowOp):
        family = "test.classified"
        activation = InputTensor(index=0)
        result = OutputTensor(index=0)

        @constraint(shape=activation.shape)
        def rank_is_two(*, shape: tuple[int, ...]) -> object:
            return len(shape) == 2

        source_accepts = ConstraintGroup(rank_is_two)

    assert Classified.source_accepts.constraints == (Classified.rank_is_two,)


def test_source_accepts_must_be_a_constraint_group() -> None:
    with pytest.raises(AuthoringError, match="names one ConstraintGroup"):

        class Broken(DataflowOp):
            family = "test.broken-group"
            activation = InputTensor(index=0)
            result = OutputTensor(index=0)
            source_accepts = "everything"


# -- rank ----------------------------------------------------------------------


def _rank_one_mvau_model(*, matrix_width: int = 8, matrix_height: int = 4) -> ModelWrapper:
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weight"],
        ["output"],
        domain=DATAFLOW_DOMAIN,
        name="mvau0",
    )
    graph = helper.make_graph(
        [node],
        "mvau",
        [_tensor("activation", (matrix_width,))],
        [_tensor("output", (matrix_height,))],
        value_info=[_tensor("weight", (matrix_width, matrix_height))],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    for name in ("activation", "weight"):
        model.set_tensor_datatype(name, DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT32"])
    model.set_initializer("weight", np.zeros((matrix_width, matrix_height), dtype=np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def test_a_rank_one_activation_is_one_repetition_and_has_an_applicable_design() -> None:
    """The docstring used to claim the opposite; nothing enforced it.

    A restriction has to be argued from the mathematics or from a Design's
    structure.  Neither argues for one here: a vector through a matrix is a
    single repetition, and every Design builds it with no special case.
    """

    model = _rank_one_mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    assert bound.answer(MvauDataflowOp.repetitions) == Decided(1)

    _model, operation = _configured_mvau(model)
    assert isinstance(operation.network, Decided)
    assert operation.expected_outputs()["output"][0] == (4,)
    assert operation.reconciliation() == ()
