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

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.model.declarations import AuthoringError, Problem, declared_members
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
    CommitmentStage,
    apply_graph_effects,
    assign_dataflow_scope_ids,
    drop_recorded_choices,
)
from finn.dataflow.ops.replay.design import ActivationReplayDesign
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.schema import InputTensor, OutputTensor
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
        helper.make_model(graph, opset_imports=[helper.make_opsetid(DATAFLOW_DOMAIN, 1)])
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
        helper.make_model(graph, opset_imports=[helper.make_opsetid(DATAFLOW_DOMAIN, 1)])
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype("expanded", DataType["INT8"])
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
    apply_graph_effects(model, chosen.graph_effects())
    assert isinstance(chosen, MvauDataflowOp)
    return model, chosen


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
    apply_graph_effects(model, chosen.graph_effects())
    assert isinstance(chosen, ActivationReplayOp)
    return model, chosen


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


def test_an_output_observation_is_not_part_of_the_problem_identity() -> None:
    """Otherwise applying a repair invalidates the identity it was committed under."""

    assert OutputTensor(index=0).fingerprint is False
    assert InputTensor(index=0).fingerprint is True

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


def test_an_uncommitted_design_alternative_is_named_as_the_reason() -> None:
    model = _mvau_model()
    bound = _unbound(model, "mvau0").bind(model, Build())

    with pytest.raises(DataflowOpError, match="Design alternative is not chosen"):
        _ = bound.dataflow


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


def test_only_the_reachable_point_is_recorded() -> None:
    """A choice with nowhere to go is dropped, not written and then unreplayable.

    The recorded state is a serialization of the *point*, not a patch built up
    from what was written before.  Switching alternatives goes through a clear,
    because an immutable point does not rebase a committed selector -- and the
    node that results must carry nothing belonging to the alternative left
    behind.
    """

    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("supplied").root
    chosen = (
        chosen.design.alternative("supplied")
        .assign(SuppliedDotProductDesign.weight_supply, WeightSupply.EMBEDDED)
        .root
    )
    apply_graph_effects(model, chosen.graph_effects())
    assert "weight_supply" in dict(_unbound(model, "mvau0").recorded())

    scope = chosen.binding.node_identity
    drop_recorded_choices(model, scope, [item.name for item in MvauDataflowOp.attributes])
    switched = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    apply_graph_effects(model, switched.graph_effects())

    reloaded = _unbound(model, "mvau0")
    assert "weight_supply" not in dict(reloaded.recorded())
    assert reloaded.recorded()["dataflow_design"] == "dot_product"
    # And it reloads: nothing left over refuses to replay.
    assert reloaded.bind(model, Build()).recorded()["dataflow_design"] == "dot_product"


def test_a_partial_point_may_be_saved(tmp_path: Path) -> None:
    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    apply_graph_effects(model, chosen.graph_effects())

    path = tmp_path / "partial.onnx"
    model.save(str(path))
    reloaded = ModelWrapper(str(path))
    restored = _unbound(reloaded, "mvau0").bind(reloaded, Build())

    assert set(restored.recorded()) == {"dataflow_design", "dataflow_compute"}
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


def test_the_layer_owns_its_own_attribute_names() -> None:
    model, _operation = _configured_mvau()
    node = model.graph.node[0]

    names = {item.name for item in node.attribute}

    assert {SCOPE_ID_ATTRIBUTE, FAMILY_ATTRIBUTE, FINGERPRINT_ATTRIBUTE} <= names
    family = next(item for item in node.attribute if item.name == FAMILY_ATTRIBUTE)
    assert family.s.decode("utf-8") == "finn.dataflow.mvau"


def test_clearing_keeps_the_identity_and_drops_every_choice() -> None:
    model, operation = _configured_mvau()
    scope = operation.binding.node_identity

    drop_recorded_choices(model, scope, [item.name for item in MvauDataflowOp.attributes])

    reloaded = _unbound(model, "mvau0")
    assert reloaded.recorded() == {}
    assert reloaded.recorded_scope_id() == scope


def test_a_changed_problem_makes_the_recorded_choices_stale() -> None:
    model, _operation = _configured_mvau()

    model.set_tensor_datatype("activation", DataType["INT4"])

    with pytest.raises(DataflowOpError, match="different problem"):
        _unbound(model, "mvau0").bind(model, Build())


def test_a_saved_family_that_this_build_does_not_offer_is_refused() -> None:
    model, _operation = _configured_mvau()
    _unbound(model, "mvau0").set_nodeattr(FAMILY_ATTRIBUTE, "finn.dataflow.something_else")

    with pytest.raises(DataflowOpError, match="stores choices for family"):
        _unbound(model, "mvau0").bind(model, Build())


def test_the_commitment_stage_is_recorded_with_the_plan() -> None:
    _model, operation = _configured_mvau()

    effects = operation.graph_effects(require=CommitmentStage.DATAFLOW)

    assert effects.validated_stage is CommitmentStage.DATAFLOW
    assert effects.expected_source_fingerprint == operation.problem_fingerprint


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
    plain = _configured_mvau()[1].association
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


def test_the_replay_operation_has_no_selector_and_needs_none() -> None:
    assert all(type(item).__name__ == "DecisionAttribute" for item in ActivationReplayOp.attributes)
    assert {item.name for item in ActivationReplayOp.attributes} == {"PE", "SIMD"}
    assert {item.name for item in MvauDataflowOp.attributes} >= {
        "dataflow_design",
        "dataflow_compute",
    }


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
