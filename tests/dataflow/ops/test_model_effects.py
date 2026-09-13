from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from onnx import TensorAnnotation, TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from dataflow.ops.test_selected_graph import _fixture as _selected_fixture
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.model_effects import (
    MODEL_READ_PRESENT,
    ModelEffects,
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    apply_model_effects,
    model_snapshot_digest,
)
from finn.dataflow.ops.native import NativeAttribute, read_attributes
from finn.dataflow.ops.selected import build_selected_snapshot
from finn.dataflow.ops.tensor_summary import FrozenInitializer

TEST_DOMAIN = "test.model_effects"


def _tensor(name: str, shape: tuple[int, ...]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _node(
    name: str,
    inputs: tuple[str, ...],
    target: str,
    scope: str,
    *,
    op_type: str = "Identity",
):
    return helper.make_node(
        op_type,
        list(inputs),
        [target],
        domain=TEST_DOMAIN,
        name=name,
        dataflow_scope_id=scope,
        hint="old",
    )


def _model() -> ModelWrapper:
    producer = _node("display-producer", ("X",), "Y", "producer")
    unused = _node("display-unused", ("X", "W"), "unused", "unused", op_type="Add")
    graph = helper.make_graph(
        [producer, unused],
        "effects",
        [_tensor("X", (2,)), _tensor("W", (2,))],
        [_tensor("Y", (2,))],
        value_info=[_tensor("unused", (2,)), _tensor("orphan", (1,))],
        initializer=[
            numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), name="W"),
            numpy_helper.from_array(np.array([7.0], dtype=np.float32), name="orphan_init"),
        ],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(TEST_DOMAIN, 1)],
        )
    )
    for name in ("X", "W", "Y", "unused"):
        model.set_tensor_datatype(name, DataType["INT8"])
    model.set_metadata_prop("old", "value")
    return model


def _annotation(name: str, datatype: str) -> bytes:
    annotation = TensorAnnotation(tensor_name=name)
    entry = annotation.quant_parameter_tensor_names.add()
    entry.key = "finn_datatype"
    entry.value = datatype
    return annotation.SerializeToString(deterministic=True)


def _add_capturing_subgraph(model: ModelWrapper) -> None:
    body = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["branch_value"])],
        "body",
        [],
        [_tensor("branch_value", (2,))],
    )
    model.graph.node[1].attribute.append(helper.make_attribute("body", body))


def _effects_cases(model: ModelWrapper):
    replacement = _node("replacement-display", ("X", "W"), "unused", "unused", op_type="Add")
    inserted = _node("inserted-display", ("X",), "extra", "inserted")
    new_w = numpy_helper.from_array(np.array([11.0, 12.0], dtype=np.float32), name="W")
    return (
        (
            "remove_nodes",
            ModelEffects(remove_nodes=("unused",)),
            lambda current: len(current.graph.node) == 1,
        ),
        (
            "replace_nodes",
            ModelEffects(
                replace_nodes=(("unused", replacement.SerializeToString(deterministic=True)),)
            ),
            lambda current: current.graph.node[1].name == "replacement-display",
        ),
        (
            "insert_nodes",
            ModelEffects(insert_nodes=((1, inserted.SerializeToString(deterministic=True)),)),
            lambda current: len(current.graph.node) == 3,
        ),
        (
            "node_order",
            ModelEffects(node_order=("unused", "producer")),
            lambda current: current.graph.node[0].name == "display-unused",
        ),
        (
            "rename_values",
            ModelEffects(rename_values=(("X", "renamed_X"),)),
            lambda current: current.graph.input[0].name == "renamed_X",
        ),
        (
            "remove_graph_inputs",
            ModelEffects(remove_graph_inputs=("W",)),
            lambda current: all(item.name != "W" for item in current.graph.input),
        ),
        (
            "remove_value_info",
            ModelEffects(remove_value_info=("orphan",)),
            lambda current: all(item.name != "orphan" for item in current.graph.value_info),
        ),
        (
            "remove_initializers",
            ModelEffects(remove_initializers=("orphan_init",)),
            lambda current: all(item.name != "orphan_init" for item in current.graph.initializer),
        ),
        (
            "set_initializers",
            ModelEffects(set_initializers=(("W", new_w.SerializeToString(deterministic=True)),)),
            lambda current: float(current.get_initializer("W")[0]) == 11.0,
        ),
        (
            "replace_quantization_annotations",
            ModelEffects(replace_quantization_annotations=(_annotation("Y", "INT4"),)),
            lambda current: current.get_tensor_datatype("Y") == DataType["INT4"],
        ),
        (
            "remove_metadata",
            ModelEffects(remove_metadata=("old",)),
            lambda current: current.get_metadata_prop("old") is None,
        ),
        (
            "set_metadata",
            ModelEffects(set_metadata=(("new", "value"),)),
            lambda current: current.get_metadata_prop("new") == "value",
        ),
        (
            "remove_attributes",
            ModelEffects(remove_attributes=(("unused", "hint"),)),
            lambda current: all(item.name != "hint" for item in current.graph.node[1].attribute),
        ),
        (
            "set_attributes",
            ModelEffects(set_attributes=(("unused", "new_hint", NativeAttribute("s", "v")),)),
            lambda current: any(
                item.name == "new_hint" for item in current.graph.node[1].attribute
            ),
        ),
        (
            "tensor_datatypes",
            ModelEffects(tensor_datatypes=(("Y", DataType["INT4"]),)),
            lambda current: current.get_tensor_datatype("Y") == DataType["INT4"],
        ),
        (
            "tensor_shapes",
            ModelEffects(tensor_shapes=(("Y", (1, 2)),)),
            lambda current: current.get_tensor_shape("Y") == [1, 2],
        ),
    )


@pytest.mark.parametrize("phase", ["validate", "finish"])
@pytest.mark.parametrize("index", range(16))
def test_every_write_class_has_a_failure_control(index: int, phase: str) -> None:
    model = _model()
    _name, effects, assertion = _effects_cases(model)[index]
    before = model.model.SerializeToString(deterministic=True)

    def validate(candidate: ModelWrapper) -> None:
        assert assertion(candidate)
        if phase == "validate":
            raise RuntimeError("fault after prepared write class")

    def finish(candidate: ModelWrapper) -> None:
        assert assertion(candidate)
        if phase == "finish":
            raise RuntimeError("fault after live write class")

    message = "prepared" if phase == "validate" else "live"
    with pytest.raises(RuntimeError, match=f"fault after {message} write class"):
        apply_model_effects(model, effects, validate=validate, finish=finish)
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("callback", ["validate", "finish"])
def test_callbacks_cannot_escape_the_declared_write_surface(callback: str) -> None:
    model = _model()
    before = model.model.SerializeToString(deterministic=True)

    def mutating(candidate: ModelWrapper) -> None:
        candidate.model.opset_import[0].version = 99

    kwargs = {
        "validate": mutating if callback == "validate" else lambda _model: None,
        "finish": mutating if callback == "finish" else lambda _model: None,
    }
    with pytest.raises(DataflowOpError, match=f"{callback} callback mutated"):
        apply_model_effects(
            model,
            ModelEffects(set_metadata=(("declared", "write"),)),
            **kwargs,
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_validate_closure_cannot_leak_a_live_model_edit_when_it_raises() -> None:
    model = _model()
    before = model.model.SerializeToString(deterministic=True)

    def fail(_candidate: ModelWrapper) -> None:
        model.set_metadata_prop("leaked", "yes")
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        apply_model_effects(
            model,
            ModelEffects(),
            validate=fail,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before
    assert model.get_metadata_prop("leaked") is None


def test_validate_closure_live_edit_is_refused_even_when_it_returns() -> None:
    model = _model()
    before = model.model.SerializeToString(deterministic=True)

    def leak(_candidate: ModelWrapper) -> None:
        model.set_metadata_prop("leaked", "yes")

    with pytest.raises(DataflowOpError, match="validate callback mutated the live model"):
        apply_model_effects(
            model,
            ModelEffects(),
            validate=leak,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before
    assert model.get_metadata_prop("leaked") is None


def test_nested_mutable_write_records_are_detached_at_plan_construction() -> None:
    metadata = ["reviewed", "original"]
    native_values = [1, 2]
    shape = [1, 2]
    datatype = DataType["INT4"]
    effects = ModelEffects(
        set_metadata=(metadata,),
        set_attributes=(("unused", "frozen_values", NativeAttribute("ints", native_values)),),
        tensor_shapes=(("Y", shape),),
        tensor_datatypes=(("Y", datatype),),
    )
    metadata[1] = "changed-after-plan"
    native_values.append(3)
    shape[0] = 99
    datatype._bitwidth = 2

    model = _model()
    apply_model_effects(
        model,
        effects,
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("reviewed") == "original"
    assert read_attributes(model.graph.node[1])["frozen_values"] == NativeAttribute("ints", (1, 2))
    assert model.get_tensor_shape("Y") == [1, 2]
    assert model.get_tensor_datatype("Y") == DataType["INT4"]


def test_finish_exception_restores_bytes_and_requires_fresh_node_lookup() -> None:
    model = _model()
    stale_node = model.graph.node[0]
    before = model.model.SerializeToString(deterministic=True)

    def finish(_candidate: ModelWrapper) -> None:
        raise RuntimeError("finish failed")

    with pytest.raises(RuntimeError, match="finish failed"):
        apply_model_effects(
            model,
            ModelEffects(set_metadata=(("temporary", "value"),)),
            validate=lambda _model: None,
            finish=finish,
        )
    assert model.model.SerializeToString(deterministic=True) == before
    assert model.graph.node[0] is not stale_node
    assert model.graph.node[0].name == "display-producer"


def test_read_set_uses_stable_id_and_ignores_display_name() -> None:
    model = _model()
    hint = next(item for item in model.graph.node[0].attribute if item.name == "hint")
    effects = ModelEffects(
        read_set=ModelReadSet(
            (
                ModelReadExpectation(
                    ModelReadKind.NODE,
                    "producer",
                    "operator",
                    b'["test.model_effects","Identity"]',
                ),
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    "producer",
                    "input:0",
                    MODEL_READ_PRESENT,
                ),
                ModelReadExpectation(
                    ModelReadKind.ATTRIBUTE,
                    "producer",
                    "hint",
                    hint.SerializeToString(deterministic=True),
                ),
            )
        ),
        set_metadata=(("applied", "yes"),),
    )
    model.graph.node[0].name = "renamed-display"
    apply_model_effects(
        model,
        effects,
        validate=lambda _model: None,
        finish=lambda current: current.get_metadata_prop("applied"),
    )
    assert model.get_metadata_prop("applied") == "yes"

    stale = replace(effects, set_metadata=(("second", "no"),))
    current_hint = next(item for item in model.graph.node[0].attribute if item.name == "hint")
    current_hint.s = b"changed"
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="Decision attribute"):
        apply_model_effects(model, stale, validate=lambda _model: None, finish=lambda _model: None)
    assert model.model.SerializeToString(deterministic=True) == before


def test_ordinal_tensor_reads_survive_a_coherent_input_rename() -> None:
    model = _model()
    digest = FrozenInitializer.from_tensor_proto(model.graph.initializer[0]).summary.content_digest
    effects = ModelEffects(
        read_set=ModelReadSet(
            (
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    "producer",
                    "input:0",
                    MODEL_READ_PRESENT,
                ),
                ModelReadExpectation(ModelReadKind.TENSOR_FACT, "producer", "input:0:shape", "[2]"),
                ModelReadExpectation(
                    ModelReadKind.TENSOR_FACT, "producer", "input:0:carrier_dtype", 1
                ),
                ModelReadExpectation(
                    ModelReadKind.TENSOR_FACT,
                    "producer",
                    "input:0:logical_datatype",
                    "INT8",
                ),
                ModelReadExpectation(
                    ModelReadKind.INITIALIZER_CONTENT, "unused", "input:1", digest
                ),
            )
        ),
        set_metadata=(("ordinal", "accepted"),),
    )
    model.rename_tensor("X", "activation")
    model.rename_tensor("W", "weights")

    apply_model_effects(
        model,
        effects,
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("ordinal") == "accepted"


def test_literal_old_presence_marker_spelling_is_an_exact_slot_name() -> None:
    model = _model()
    literal = "__finn_model_read_present_v1__"
    model.rename_tensor("X", literal)
    effects = ModelEffects(
        read_set=ModelReadSet(
            (
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    "producer",
                    "input:0",
                    literal,
                ),
            )
        )
    )
    model.graph.node[0].input[0] = "W"
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="operand_slot"):
        apply_model_effects(
            model,
            effects,
            validate=lambda _model: None,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_presence_marker_accepts_any_nonempty_operand_spelling() -> None:
    model = _model()
    effects = ModelEffects(
        read_set=ModelReadSet(
            (
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    "producer",
                    "input:0",
                    MODEL_READ_PRESENT,
                ),
            )
        ),
        set_metadata=(("presence", "accepted"),),
    )
    model.graph.node[0].input[0] = "W"
    apply_model_effects(
        model,
        effects,
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("presence") == "accepted"


def test_ordinal_initializer_content_change_stales_the_plan() -> None:
    model = _model()
    digest = FrozenInitializer.from_tensor_proto(model.graph.initializer[0]).summary.content_digest
    effects = ModelEffects(
        read_set=ModelReadSet(
            (ModelReadExpectation(ModelReadKind.INITIALIZER_CONTENT, "unused", "input:1", digest),)
        ),
        set_metadata=(("not", "applied"),),
    )
    model.set_initializer("W", np.array([99.0, 100.0], dtype=np.float32))
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="initializer_content"):
        apply_model_effects(
            model, effects, validate=lambda _model: None, finish=lambda _model: None
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_exact_output_target_reads_reject_retargeting() -> None:
    model = _model()
    output = model.graph.output[0]
    annotation = next(
        item for item in model.graph.quantization_annotation if item.tensor_name == "Y"
    )
    effects = ModelEffects(
        read_set=ModelReadSet(
            (
                ModelReadExpectation(ModelReadKind.OPERAND_SLOT, "producer", "output:0", "Y"),
                ModelReadExpectation(
                    ModelReadKind.VALUE_INFO,
                    "Y",
                    "output",
                    output.SerializeToString(deterministic=True),
                ),
                ModelReadExpectation(
                    ModelReadKind.QUANTIZATION_ANNOTATION,
                    "Y",
                    None,
                    annotation.SerializeToString(deterministic=True),
                ),
            )
        ),
        set_metadata=(("not", "applied"),),
    )
    model.rename_tensor("Y", "renamed_Y")
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="operand_slot"):
        apply_model_effects(
            model, effects, validate=lambda _model: None, finish=lambda _model: None
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_remaining_read_kinds_check_exact_facts() -> None:
    model = _model()
    graph_input = model.graph.input[0]
    orphan = next(item for item in model.graph.value_info if item.name == "orphan")
    effects = ModelEffects(
        read_set=ModelReadSet(
            (
                ModelReadExpectation(
                    ModelReadKind.NODE_ORDER,
                    model.graph.name,
                    None,
                    json.dumps(["producer", "unused"], separators=(",", ":")),
                ),
                ModelReadExpectation(
                    ModelReadKind.GRAPH_INPUT,
                    "X",
                    0,
                    graph_input.SerializeToString(deterministic=True),
                ),
                ModelReadExpectation(
                    ModelReadKind.VALUE_INFO,
                    "orphan",
                    "value_info",
                    orphan.SerializeToString(deterministic=True),
                ),
                ModelReadExpectation(ModelReadKind.METADATA, "old", None, "value"),
                ModelReadExpectation(ModelReadKind.OPSET, TEST_DOMAIN, None, 1),
            )
        ),
        set_metadata=(("checked", "yes"),),
    )
    apply_model_effects(
        model,
        effects,
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("checked") == "yes"


def test_snapshot_digest_is_a_stale_guard_not_write_authorization() -> None:
    model = _model()
    digest = model_snapshot_digest(model)
    effects = ModelEffects(
        read_set=ModelReadSet(
            (ModelReadExpectation(ModelReadKind.SNAPSHOT_DIGEST, "", None, digest),)
        ),
        set_metadata=(("snapshot", "matched"),),
    )
    apply_model_effects(
        model,
        effects,
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("snapshot") == "matched"

    stale_model = _model()
    stale_effects = replace(effects, read_set=effects.read_set)
    stale_model.graph.node[0].doc_string = "changed"
    before = stale_model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="snapshot_digest"):
        apply_model_effects(
            stale_model,
            stale_effects,
            validate=lambda _model: None,
            finish=lambda _model: None,
        )
    assert stale_model.model.SerializeToString(deterministic=True) == before


def test_duplicate_read_addresses_are_refused() -> None:
    item = ModelReadExpectation(ModelReadKind.OPSET, TEST_DOMAIN, None, 1)
    with pytest.raises(ValueError, match="unique"):
        ModelReadSet((item, item))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"kind": "opset", "owner": TEST_DOMAIN, "field": None, "expected": 1},
        {"kind": ModelReadKind.OPSET, "owner": 1, "field": None, "expected": 1},
        {"kind": ModelReadKind.OPSET, "owner": TEST_DOMAIN, "field": [], "expected": 1},
        {"kind": ModelReadKind.OPSET, "owner": TEST_DOMAIN, "field": None, "expected": True},
    ),
)
def test_read_expectations_reject_non_nominal_record_fields(kwargs) -> None:
    with pytest.raises(TypeError, match="model read"):
        ModelReadExpectation(**kwargs)


def test_display_name_is_not_a_node_identity() -> None:
    model = _model()
    with pytest.raises(DataflowOpError, match="unknown stable node id"):
        apply_model_effects(
            model,
            ModelEffects(set_attributes=(("display-producer", "x", NativeAttribute("i", 1)),)),
            validate=lambda _model: None,
            finish=lambda _model: None,
        )


def test_selected_v2_node_id_resolves_only_through_binding_index() -> None:
    _network, declaration, selected_model = _selected_fixture()
    selected = build_selected_snapshot(selected_model, declaration).model_copy()
    replacement = helper.make_node("Identity", ["X"], ["Y"], name="new-display")
    apply_model_effects(
        selected,
        ModelEffects(
            replace_nodes=(("identity", replacement.SerializeToString(deterministic=True)),)
        ),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert selected.graph.node[0].name == "new-display"


def test_graph_output_producer_change_refuses_before_live_mutation() -> None:
    model = _model()
    before = model.model.SerializeToString(deterministic=True)
    replacement = _node("replacement", ("X",), "other", "producer")
    called = []
    with pytest.raises(DataflowOpError, match="graph-output producer"):
        apply_model_effects(
            model,
            ModelEffects(
                replace_nodes=(("producer", replacement.SerializeToString(deterministic=True)),)
            ),
            validate=lambda _model: called.append(True),
            finish=lambda _model: None,
        )
    assert not called
    assert model.model.SerializeToString(deterministic=True) == before


def test_nonstructural_effect_does_not_require_unrelated_standard_node_ids() -> None:
    node = helper.make_node("Identity", ["X"], ["Y"], name="ordinary-display")
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph([node], "ordinary", [_tensor("X", (1,))], [_tensor("Y", (1,))]),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    apply_model_effects(
        model,
        ModelEffects(set_metadata=(("allowed", "yes"),)),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("allowed") == "yes"


def test_private_wrapper_startup_repairs_do_not_become_undeclared_writes() -> None:
    model = _model()
    kept = [item for item in model.graph.value_info if item.name != "orphan_init"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    apply_model_effects(
        model,
        ModelEffects(set_metadata=(("only", "declared"),)),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert all(item.name != "orphan_init" for item in model.graph.value_info)


@pytest.mark.parametrize(
    ("effects", "message"),
    (
        (ModelEffects(remove_graph_inputs=("X",)), "static initializer-backed"),
        (ModelEffects(remove_value_info=("unused",)), "final-state-unused value_info"),
    ),
)
def test_bounded_removals_refuse_unsupported_targets(effects: ModelEffects, message: str) -> None:
    model = _model()
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match=message):
        apply_model_effects(
            model, effects, validate=lambda _model: None, finish=lambda _model: None
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_value_info_made_unused_by_same_transaction_is_removed_atomically() -> None:
    model = _model()
    apply_model_effects(
        model,
        ModelEffects(remove_nodes=("unused",), remove_value_info=("unused",)),
        validate=lambda candidate: all(
            item.name != "unused" for item in candidate.graph.value_info
        ),
        finish=lambda _model: None,
    )
    assert len(model.graph.node) == 1
    assert all(item.name != "unused" for item in model.graph.value_info)


@pytest.mark.parametrize("structural", ["replace", "insert"])
def test_structural_writes_cannot_create_new_external_inputs(structural: str) -> None:
    model = _model()
    if structural == "replace":
        node = _node("replacement", ("X", "NEW"), "unused", "unused", op_type="Add")
        effects = ModelEffects(
            replace_nodes=(("unused", node.SerializeToString(deterministic=True)),)
        )
    else:
        node = _node("inserted", ("NEW",), "extra", "inserted")
        effects = ModelEffects(insert_nodes=((1, node.SerializeToString(deterministic=True)),))
    before = model.model.SerializeToString(deterministic=True)
    called = []
    with pytest.raises(DataflowOpError, match="new external inputs"):
        apply_model_effects(
            model,
            effects,
            validate=lambda _model: called.append(True),
            finish=lambda _model: None,
        )
    assert not called
    assert model.model.SerializeToString(deterministic=True) == before


def test_source_node_replacement_must_preserve_its_scope_id() -> None:
    model = _model()
    replacement = _node("replacement", ("X", "W"), "unused", "different", op_type="Add")
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="preserve its stable scope id"):
        apply_model_effects(
            model,
            ModelEffects(
                replace_nodes=(("unused", replacement.SerializeToString(deterministic=True)),)
            ),
            validate=lambda _model: None,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_total_value_rename_is_simultaneous_and_repairs_annotations() -> None:
    model = _model()
    apply_model_effects(
        model,
        ModelEffects(rename_values=(("X", "W"), ("W", "X"))),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert [item.name for item in model.graph.input] == ["W", "X"]
    assert [item.name for item in model.graph.initializer if item.name in {"X", "W"}] == ["X"]
    assert list(model.graph.node[0].input) == ["W"]
    assert list(model.graph.node[1].input) == ["W", "X"]
    annotation_names = {item.tensor_name for item in model.graph.quantization_annotation}
    assert {"X", "W"} <= annotation_names


def test_value_rename_does_not_rewrite_logical_datatype_values() -> None:
    model = _model()
    model.rename_tensor("X", "INT8")
    apply_model_effects(
        model,
        ModelEffects(rename_values=(("INT8", "activation"),)),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_tensor_datatype("activation") == DataType["INT8"]
    assert {
        entry.value
        for annotation in model.graph.quantization_annotation
        for entry in annotation.quant_parameter_tensor_names
        if entry.key == "finn_datatype"
    } == {"INT8"}


def test_logical_datatype_removal_drops_an_empty_annotation() -> None:
    model = _model()
    model.set_tensor_datatype("orphan", DataType["INT4"])
    apply_model_effects(
        model,
        ModelEffects(tensor_datatypes=(("orphan", None),)),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert all(item.tensor_name != "orphan" for item in model.graph.quantization_annotation)


@pytest.mark.parametrize("field", ["tensor_shapes", "tensor_datatypes"])
def test_tensor_fact_writes_refuse_unknown_values_without_creating_value_info(field: str) -> None:
    model = _model()
    effects = (
        ModelEffects(tensor_shapes=(("missing", (1,)),))
        if field == "tensor_shapes"
        else ModelEffects(tensor_datatypes=(("missing", DataType["INT4"]),))
    )
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="target 'missing'"):
        apply_model_effects(
            model, effects, validate=lambda _model: None, finish=lambda _model: None
        )
    assert model.model.SerializeToString(deterministic=True) == before
    assert all(item.name != "missing" for item in model.graph.value_info)


def test_subgraph_node_edits_are_refused_before_live_mutation() -> None:
    model = _model()
    branch = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Z"])],
        "branch",
        [_tensor("X", (2,))],
        [_tensor("Z", (2,))],
    )
    replacement = helper.make_node(
        "If",
        ["X"],
        ["Y"],
        domain=TEST_DOMAIN,
        name="replacement",
        dataflow_scope_id="producer",
        then_branch=branch,
        else_branch=branch,
    )
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="subgraphs"):
        apply_model_effects(
            model,
            ModelEffects(
                replace_nodes=(("producer", replacement.SerializeToString(deterministic=True)),)
            ),
            validate=lambda _model: None,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before


def test_unrelated_metadata_and_native_attribute_writes_allow_untouched_subgraphs() -> None:
    model = _model()
    _add_capturing_subgraph(model)
    apply_model_effects(
        model,
        ModelEffects(
            set_metadata=(("unrelated", "accepted"),),
            set_attributes=(("producer", "unrelated", NativeAttribute("i", 1)),),
        ),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.get_metadata_prop("unrelated") == "accepted"
    assert read_attributes(model.graph.node[0])["unrelated"] == NativeAttribute("i", 1)


def test_rename_unrelated_to_nested_values_preserves_untouched_subgraph() -> None:
    model = _model()
    _add_capturing_subgraph(model)
    apply_model_effects(
        model,
        ModelEffects(rename_values=(("Y", "renamed_Y"),)),
        validate=lambda _model: None,
        finish=lambda _model: None,
    )
    assert model.graph.output[0].name == "renamed_Y"
    body = next(item for item in model.graph.node[1].attribute if item.name == "body").g
    assert body.node[0].input[0] == "X"


@pytest.mark.parametrize(
    "effects",
    (
        ModelEffects(remove_nodes=("unused",)),
        ModelEffects(rename_values=(("X", "renamed_X"),)),
        ModelEffects(remove_attributes=(("unused", "body"),)),
        ModelEffects(set_attributes=(("unused", "body", NativeAttribute("s", "replacement")),)),
    ),
)
def test_subgraph_topology_capture_and_graph_attribute_edits_refuse_before_callbacks(
    effects: ModelEffects,
) -> None:
    model = _model()
    _add_capturing_subgraph(model)
    before = model.model.SerializeToString(deterministic=True)
    calls = []
    with pytest.raises(DataflowOpError, match="subgraph|GRAPH"):
        apply_model_effects(
            model,
            effects,
            validate=lambda _model: calls.append("validate"),
            finish=lambda _model: calls.append("finish"),
        )
    assert not calls
    assert model.model.SerializeToString(deterministic=True) == before


def test_node_order_must_cover_every_final_stable_node() -> None:
    model = _model()
    before = model.model.SerializeToString(deterministic=True)
    with pytest.raises(DataflowOpError, match="cover every final stable node"):
        apply_model_effects(
            model,
            ModelEffects(node_order=("producer",)),
            validate=lambda _model: None,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize(
    "mutation",
    ("opset", "membership", "order"),
)
def test_callbacks_cannot_change_opsets_or_output_membership_or_order(mutation: str) -> None:
    model = _model()
    if mutation == "order":
        kept = [item for item in model.graph.value_info if item.name != "unused"]
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept)
        model.graph.output.add().CopyFrom(_tensor("unused", (2,)))
    before = model.model.SerializeToString(deterministic=True)

    def validate(candidate: ModelWrapper) -> None:
        if mutation == "opset":
            candidate.model.opset_import[0].version = 99
        elif mutation == "membership":
            candidate.graph.output.add().CopyFrom(_tensor("extra", (1,)))
        else:
            reversed_outputs = list(reversed(candidate.graph.output))
            del candidate.graph.output[:]
            candidate.graph.output.extend(reversed_outputs)

    with pytest.raises(DataflowOpError, match="validate callback mutated"):
        apply_model_effects(
            model,
            ModelEffects(set_metadata=(("declared", "write"),)),
            validate=validate,
            finish=lambda _model: None,
        )
    assert model.model.SerializeToString(deterministic=True) == before
