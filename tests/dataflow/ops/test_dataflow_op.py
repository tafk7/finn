# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U4: one source node, one frozen problem, one persisted set of choices.

Two operations run through the same tests wherever the question is generic --
source projection, freezing, persistence, staleness, association -- because the
claim being checked is that the layer is not MVAU-shaped.  Where MVAU has a
Design alternative and a matrix and the replay op has neither, the tests say so
separately rather than pretending the difference away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.ops.association import CoordinateMapping
from finn.dataflow.ops.base import (
    DATAFLOW_DOMAIN,
    FAMILY_ATTRIBUTE,
    FINGERPRINT_ATTRIBUTE,
    SCOPE_ID_ATTRIBUTE,
    DataflowOp,
    DataflowOpError,
)
from finn.dataflow.ops.mvau.designs.supplied_dot_product import WeightSupply
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.replay.op import ActivationReplayOp
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
    return model


def _op(model: ModelWrapper, name: str) -> DataflowOp:
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
) -> tuple[ModelWrapper, MvauDataflowOp]:
    model = model or _mvau_model()
    operation = _op(model, "mvau0")
    assert isinstance(operation, MvauDataflowOp)
    values: dict[str, object] = {"dataflow_design": design, "PE": pe, "SIMD": simd}
    if design == "supplied":
        values["weight_supply"] = supply or WeightSupply.EXTERNAL
        values["dataflow_compute"] = (
            "dotp_axi_embedded" if supply is WeightSupply.EMBEDDED else "dotp_axi"
        )
    values["pumpedCompute"] = False
    operation.commit(Build(), values)
    return model, operation


# -- U4a: the frozen source reading -------------------------------------------


def test_the_source_node_is_read_once_and_the_graph_is_not_reread() -> None:
    model = _mvau_model()
    operation = _op(model, "mvau0")
    build = Build()

    first = operation.start(build)
    fingerprint = first.problem_fingerprint

    # Change something the design space reads.  The occurrence already started
    # does not move; only a *new* reading sees it.
    model.set_tensor_datatype("activation", DataType["INT4"])
    assert first.problem_fingerprint == fingerprint
    assert operation.start(build).problem_fingerprint != fingerprint


def test_an_unattached_operation_refuses_rather_than_guessing() -> None:
    from qonnx.custom_op.registry import getCustomOp  # noqa: PLC0415 - the bare path

    model = _mvau_model()
    bare = getCustomOp(model.graph.node[0])
    assert isinstance(bare, MvauDataflowOp)
    with pytest.raises(DataflowOpError, match="needs its ModelWrapper"):
        bare.start(Build())


def test_a_missing_shape_or_datatype_is_a_source_refusal() -> None:
    model = _mvau_model()
    model.graph.node[0].input[1] = "unknown_tensor"
    with pytest.raises(SourceError, match="has no shape"):
        _op(model, "mvau0").start(Build())


def test_the_two_operations_read_entirely_different_operand_sets() -> None:
    mvau = _op(_mvau_model(), "mvau0").read_source()
    replay = _op(_replay_model(), "replay0").read_source()
    assert tuple(item.id for item in mvau.inputs) == ("activation", "weight")
    assert tuple(item.id for item in replay.inputs) == ("activation",)
    assert mvau.operand("weight").initializer is True
    assert replay.operand("activation").initializer is False


# -- U4b: the source projection -----------------------------------------------


def test_the_mvau_source_projects_folding_facts_from_its_tensors() -> None:
    _model, operation = _configured_mvau()
    network = operation.network(Build())
    assert isinstance(network, Decided), network
    assert {node.id for node in network.value.nodes} == {"replay", "compute"}
    assert {item.id for item in network.value.boundaries} == {
        "activation",
        "weight",
        "output",
    }


def test_the_replay_source_projects_one_node_and_no_selector() -> None:
    model = _replay_model()
    operation = _op(model, "replay0")
    operation.commit(Build(), {"PE": 2, "SIMD": 4})
    network = operation.network(Build())
    assert isinstance(network, Decided), network
    assert {node.id for node in network.value.nodes} == {"replay"}
    assert network.value.edges == ()
    assert {item.id for item in network.value.boundaries} == {"activation", "expanded"}


def test_an_uncommitted_design_alternative_is_named_as_the_reason() -> None:
    operation = _op(_mvau_model(), "mvau0")
    with pytest.raises(DataflowOpError, match="Design alternative is not chosen"):
        operation.network(Build())


def test_an_uncommitted_folding_leaves_the_network_unresolved() -> None:
    model = _mvau_model()
    operation = _op(model, "mvau0")
    operation.commit(Build(), {"dataflow_design": "dot_product"})
    assert isinstance(operation.network(Build()), Unresolved)


# -- U4c: persistence ---------------------------------------------------------


def test_choices_survive_a_save_and_reload_exactly(tmp_path: Path) -> None:
    model, operation = _configured_mvau(pe=2, simd=4)
    before = operation.network(Build())
    assert isinstance(before, Decided)

    path = tmp_path / "mvau.onnx"
    model.save(str(path))
    reloaded = ModelWrapper(str(path))
    restored = _op(reloaded, "mvau0")
    assert dict(restored.recorded()) == dict(operation.recorded())
    after = restored.network(Build())
    assert isinstance(after, Decided)
    assert after.value == before.value


def test_only_the_choices_physically_present_are_replayed(tmp_path: Path) -> None:
    model = _mvau_model()
    operation = _op(model, "mvau0")
    operation.commit(Build(), {"dataflow_design": "dot_product", "PE": 2})
    assert set(operation.recorded()) == {"dataflow_design", "PE"}
    path = tmp_path / "partial.onnx"
    model.save(str(path))
    restored = _op(ModelWrapper(str(path)), "mvau0")
    assert set(restored.recorded()) == {"dataflow_design", "PE"}
    assert isinstance(restored.network(Build()), Unresolved)


def test_a_refused_choice_leaves_the_node_byte_for_byte_unchanged() -> None:
    model = _mvau_model()
    operation = _op(model, "mvau0")
    operation.commit(Build(), {"dataflow_design": "dot_product"})
    before = model.graph.node[0].SerializeToString(deterministic=True)

    # 3 does not divide a matrix height of 4.
    with pytest.raises(DataflowOpError, match="refuses PE=3"):
        operation.commit(Build(), {"PE": 3})
    assert model.graph.node[0].SerializeToString(deterministic=True) == before


def test_an_attribute_the_operation_does_not_persist_is_refused() -> None:
    operation = _op(_mvau_model(), "mvau0")
    with pytest.raises(DataflowOpError, match="does not persist"):
        operation.commit(Build(), {"nonsense": 1})


def test_the_layer_owns_its_own_attribute_names() -> None:
    _model, operation = _configured_mvau()
    node = operation.onnx_node
    names = {item.name for item in node.attribute}
    assert {SCOPE_ID_ATTRIBUTE, FAMILY_ATTRIBUTE, FINGERPRINT_ATTRIBUTE} <= names
    family = next(item for item in node.attribute if item.name == FAMILY_ATTRIBUTE)
    assert family.s.decode("utf-8") == "finn.dataflow.mvau"


def test_clearing_keeps_the_identity_and_drops_every_choice() -> None:
    _model, operation = _configured_mvau()
    scope = operation.scope_id()
    operation.clear()
    assert operation.recorded() == {}
    assert operation.scope_id() == scope
    assert not operation.is_stale(Build())


def test_a_changed_problem_makes_the_recorded_choices_stale() -> None:
    model, operation = _configured_mvau()
    build = Build()
    assert not operation.is_stale(build)

    model.set_tensor_datatype("activation", DataType["INT4"])
    assert operation.is_stale(build)
    with pytest.raises(DataflowOpError, match="different problem"):
        operation.occurrence(build)


def test_a_stale_operation_is_reconstructed_and_never_rebased() -> None:
    model, operation = _configured_mvau()
    model.set_tensor_datatype("activation", DataType["INT4"])
    operation.clear()
    assert not operation.is_stale(Build())
    # Nothing carried over: the choices were made against other facts.
    assert operation.recorded() == {}


def test_a_saved_family_that_this_build_does_not_offer_is_refused() -> None:
    _model, operation = _configured_mvau()
    operation.set_nodeattr(FAMILY_ATTRIBUTE, "finn.dataflow.something_else")
    with pytest.raises(DataflowOpError, match="stores choices for family"):
        operation.occurrence(Build())


# -- U4d: source association ---------------------------------------------------


def test_every_source_operand_is_associated_with_where_its_data_crosses() -> None:
    _model, operation = _configured_mvau()
    answer = operation.association(Build())
    assert isinstance(answer, Decided)
    association = answer.value
    assert association.family == "finn.dataflow.mvau"
    assert {item.operand for item in association.operands} == {
        "activation",
        "weight",
        "output",
    }
    activation = association.operand("activation")
    assert activation.boundary == "activation"
    assert (activation.node_id, activation.port_id) == ("replay", "activation_in")
    assert activation.correspondence is CoordinateMapping.FLATTEN_LEADING
    weight = association.operand("weight")
    assert (weight.node_id, weight.port_id) == ("compute", "weight")


def test_an_association_names_no_kernel_component_or_artifact() -> None:
    _model, operation = _configured_mvau()
    answer = operation.association(Build())
    assert isinstance(answer, Decided)
    rendered = repr(answer.value)
    for forbidden in ("dotp_axi", "replay_buffer", "ComponentABI", "Derivation", "ArtifactRef"):
        assert forbidden not in rendered


def test_the_association_follows_the_network_not_the_declaration() -> None:
    """A matrix produced inside crosses no boundary, and the record says so."""

    external = _configured_mvau(design="supplied", supply=WeightSupply.EXTERNAL)[1]
    decoupled = _configured_mvau(_mvau_model(), design="supplied", supply=WeightSupply.DECOUPLED)[1]

    outside = external.association(Build())
    inside = decoupled.association(Build())
    assert isinstance(outside, Decided) and isinstance(inside, Decided)
    assert outside.value.operand("weight").boundary == "weight"
    assert inside.value.operand("weight").boundary is None
    # It still reaches the same port; only how it got there changed.
    assert inside.value.operand("weight").node_id == "compute"
    assert inside.value.operand("weight").port_id == "weight"


def test_a_physical_choice_does_not_change_the_association() -> None:
    plain = _configured_mvau()[1].association(Build())
    model = _mvau_model()
    operation = _op(model, "mvau0")
    operation.commit(
        Build(),
        {"dataflow_design": "dot_product", "PE": 2, "SIMD": 4, "pumpedCompute": True},
    )
    pumped = operation.association(Build())
    assert isinstance(plain, Decided) and isinstance(pumped, Decided)
    assert [item.boundary for item in plain.value.operands] == [
        item.boundary for item in pumped.value.operands
    ]
    assert [item.node_id for item in plain.value.operands] == [
        item.node_id for item in pumped.value.operands
    ]


def test_the_second_operation_associates_its_own_two_operands() -> None:
    model = _replay_model()
    operation = _op(model, "replay0")
    operation.commit(Build(), {"PE": 2, "SIMD": 4})
    answer = operation.association(Build())
    assert isinstance(answer, Decided)
    assert {item.operand for item in answer.value.operands} == {"activation", "expanded"}
    assert answer.value.family == "finn.dataflow.activation_replay"
    assert answer.value.operand("expanded").node_id == "replay"


def test_the_scope_id_survives_a_rename_of_the_node() -> None:
    model, operation = _configured_mvau()
    scope = operation.scope_id()
    model.graph.node[0].name = "renamed"
    renamed = _op(model, "renamed")
    assert renamed.scope_id() == scope
    answer = renamed.association(Build())
    assert isinstance(answer, Decided)
    assert answer.value.scope_id == scope
    assert answer.value.source_node == "renamed"


# -- U4e: the two operations do not share an implementation --------------------


def test_the_replay_operation_has_no_selector_and_needs_none() -> None:
    assert all(type(item).__name__ == "DecisionAttribute" for item in ActivationReplayOp.attributes)
    assert {item.name for item in ActivationReplayOp.attributes} == {"PE", "SIMD"}
    assert {item.name for item in MvauDataflowOp.attributes} >= {
        "dataflow_design",
        "dataflow_compute",
    }


def test_both_operations_use_the_same_persistence_authority(tmp_path: Path) -> None:
    for model, name, values in (
        (_mvau_model(), "mvau0", {"dataflow_design": "dot_product", "PE": 2, "SIMD": 2}),
        (_replay_model(), "replay0", {"PE": 2, "SIMD": 4}),
    ):
        operation = _op(model, name)
        operation.commit(Build(), values)
        path = tmp_path / f"{name}.onnx"
        model.save(str(path))
        restored = _op(ModelWrapper(str(path)), name)
        assert dict(restored.recorded()) == dict(operation.recorded())
        assert not restored.is_stale(Build())
