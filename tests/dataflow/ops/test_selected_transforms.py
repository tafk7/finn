from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from qonnx.core.datatype import DataType

import finn.dataflow.ops.selected_registry as selected_registry_module
from dataflow.ops.mvau.test_selected_mvau import _facts as _mvau_facts
from dataflow.ops.replay.test_selected_replay import _facts as _replay_facts
from dataflow.ops.selected_transform_fixtures import (
    IDENTITY_CHAIN_CONSTRUCTIONS,
    IDENTITY_CHAIN_TRANSFORMS,
    configured_identity_chain,
)
from dataflow.ops.test_selected_graph import CONSTRUCTION as IDENTITY_CONSTRUCTION
from dataflow.ops.test_selected_graph import _fixture as _identity_fixture
from finn.dataflow.model.maps import CoordinateSet, RectangularDomain
from finn.dataflow.model.region import BeatSequence, InputInterface, ScheduledInputRequirements
from finn.dataflow.ops.mvau.computation import AccumulationMode
from finn.dataflow.ops.mvau.designs.supply import WeightSupply
from finn.dataflow.ops.mvau.selected import construct_mvau_snapshot
from finn.dataflow.ops.replay.selected import construct_replay_snapshot
from finn.dataflow.ops.selected import (
    ConstructionInputs,
    ConstructionRegistry,
    GraphSlotKind,
    GraphSlotRef,
    SelectedGraphError,
    build_selected_snapshot,
    reconstruct_selected_graph,
)
from finn.dataflow.ops.selected_transforms import (
    BOUNDED_CLEANUP_TRANSFORM,
    ELIDED_IDENTITY_FORM,
    READABLE_NAMES_TRANSFORM,
    TRANSFORM_VERSION,
    SelectedTransformAuthorization,
    SelectedTransformError,
    SelectedTransformRegistry,
    apply_selected_transform,
    plan_bounded_cleanup,
    plan_equal_width_identity_elision,
    plan_readable_names,
)
from finn.dataflow.ops.selected_verification import verify_normalized_selected_snapshot


def _run(snapshot, outputs, feeds):
    return ReferenceEvaluator(snapshot.model_copy().model).run(outputs, feeds)


@pytest.mark.parametrize(
    "planner",
    (plan_readable_names, plan_bounded_cleanup),
    ids=("readable_names", "bounded_cleanup"),
)
def test_production_replay_naming_cleanup_numerics_and_reload(planner, tmp_path: Path) -> None:
    facts = _replay_facts(source_shape=(2, 3), folds=2, simd=3)
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    original = snapshot.model_bytes
    plan = planner(snapshot)
    decoded = apply_selected_transform(snapshot, plan)
    values = dict(plan.value_correspondence)
    activation = np.arange(6, dtype=np.float32).reshape(2, 3)
    (actual,) = _run(decoded.snapshot, None, {values["X"]: activation})
    assert np.array_equal(actual, np.repeat(activation, 2, axis=0))
    assert snapshot.model_bytes == original
    assert decoded.declaration.construction == snapshot.declaration.construction

    path = tmp_path / f"replay-{planner.__name__}.onnx"
    path.write_bytes(decoded.snapshot.model_bytes)
    restored = reconstruct_selected_graph(path)
    assert restored.network == decoded.network
    assert restored.selection_facts == decoded.selection_facts

    repeated = apply_selected_transform(decoded.snapshot, planner(decoded.snapshot))
    assert repeated.snapshot.content_digest == decoded.snapshot.content_digest


@pytest.mark.parametrize(
    "planner",
    (plan_readable_names, plan_bounded_cleanup),
    ids=("readable_names", "bounded_cleanup"),
)
@pytest.mark.parametrize("mode", tuple(AccumulationMode))
@pytest.mark.parametrize("supply", tuple(WeightSupply))
def test_production_mvau_naming_cleanup_full_numerics_and_reload(
    planner,
    mode: AccumulationMode,
    supply: WeightSupply,
    tmp_path: Path,
) -> None:
    facts, inputs, activation, weight, expected = _mvau_facts(mode, supply)
    snapshot = construct_mvau_snapshot(facts, inputs)
    original = snapshot.model_bytes
    plan = planner(snapshot)
    decoded = apply_selected_transform(snapshot, plan)
    values = dict(plan.value_correspondence)
    feeds = {values["X"]: activation}
    if supply is WeightSupply.EXTERNAL:
        feeds[values["W_source"]] = weight
    actual_xr, actual = _run(
        decoded.snapshot,
        [values["XR"], values["Y"]],
        feeds,
    )
    assert np.array_equal(actual_xr, np.repeat(activation, 2, axis=0))
    assert np.array_equal(actual, expected)
    assert snapshot.model_bytes == original
    assert decoded.declaration.construction == snapshot.declaration.construction

    path = tmp_path / f"mvau-{mode.value}-{supply.value}-{planner.__name__}.onnx"
    path.write_bytes(decoded.snapshot.model_bytes)
    restored = reconstruct_selected_graph(path)
    assert restored.network == decoded.network
    assert restored.selection_facts == decoded.selection_facts


def _repeated_constant_snapshot():
    network, declaration, model = _identity_fixture()
    node = model.graph.node[0]
    node.op_type = "Sum"
    del node.input[:]
    node.input.extend(("X", "X", "P", "P"))
    model.set_initializer("P", np.asarray((2.0, 3.0), dtype=np.float32))
    static_input = next(item for item in model.graph.value_info if item.name == "P")
    model.graph.input.append(static_input)
    kept = [item for item in model.graph.value_info if item.name != "P"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    model.set_initializer("unused", np.asarray((9.0,), dtype=np.float32))
    model.graph.value_info.append(helper.make_tensor_value_info("orphan", TensorProto.FLOAT, [1]))
    model.set_tensor_datatype("P", DataType["INT8"])
    model.set_tensor_datatype("orphan", DataType["INT8"])
    input_binding = replace(
        declaration.interface_bindings[0],
        anchors=(
            GraphSlotRef(GraphSlotKind.GRAPH_INPUT, model.graph.name, 0),
            GraphSlotRef(GraphSlotKind.NODE_INPUT, "identity", 0),
            GraphSlotRef(GraphSlotKind.NODE_INPUT, "identity", 1),
        ),
    )
    declaration = replace(
        declaration,
        interface_bindings=(input_binding, declaration.interface_bindings[1]),
    )
    construction = replace(IDENTITY_CONSTRUCTION, verify=lambda _snapshot, _facts: ())
    constructions = ConstructionRegistry({("test.selected", "1"): construction})
    transforms = SelectedTransformRegistry(
        {
            ("test.selected", "1"): (
                SelectedTransformAuthorization(
                    READABLE_NAMES_TRANSFORM,
                    TRANSFORM_VERSION,
                    ("canonical",),
                    None,
                    ("replace_nodes", "rename_values"),
                ),
                SelectedTransformAuthorization(
                    BOUNDED_CLEANUP_TRANSFORM,
                    TRANSFORM_VERSION,
                    ("canonical",),
                    None,
                    (
                        "replace_nodes",
                        "node_order",
                        "remove_graph_inputs",
                        "remove_value_info",
                        "remove_initializers",
                        "set_initializers",
                        "replace_quantization_annotations",
                    ),
                ),
            )
        }
    )
    return build_selected_snapshot(model, declaration), network, constructions, transforms


def test_readable_names_repairs_repeated_slots_and_total_correspondence() -> None:
    snapshot, _network, constructions, transforms = _repeated_constant_snapshot()
    old_values = {
        value
        for node in snapshot.model_copy().graph.node
        for value in (*node.input, *node.output)
        if value
    }
    plan = plan_readable_names(snapshot, constructions=constructions, transforms=transforms)
    decoded = apply_selected_transform(
        snapshot, plan, constructions=constructions, transforms=transforms
    )
    model = decoded.snapshot.model_copy()
    node = model.graph.node[0]
    assert node.input[0] == node.input[1]
    binding = decoded.declaration.interface_bindings[0]
    assert {(item.owner, item.index) for item in binding.anchors} >= {
        ("identity", 0),
        ("identity", 1),
    }
    assert {old for old, _new in plan.value_correspondence} >= old_values
    assert len({new for _old, new in plan.value_correspondence}) == len(plan.value_correspondence)


def test_cleanup_repairs_repeated_slots_and_copies_only_construction_parameters() -> None:
    snapshot, _network, constructions, transforms = _repeated_constant_snapshot()
    expected = _run(
        snapshot,
        None,
        {"X": np.asarray((4.0, 5.0), dtype=np.float32)},
    )[0]
    plan = plan_bounded_cleanup(snapshot, constructions=constructions, transforms=transforms)
    decoded = apply_selected_transform(
        snapshot, plan, constructions=constructions, transforms=transforms
    )
    model = decoded.snapshot.model_copy()
    node = model.graph.node[0]
    assert tuple(node.input[:2]) == ("X", "X")
    assert node.input[2] == "P"
    assert node.input[3].startswith("P_unique_")
    assert ("P", node.input[3]) in plan.value_correspondence
    assert "P" not in {item.name for item in model.graph.input}
    assert "unused" not in {item.name for item in model.graph.initializer}
    assert "orphan" not in {item.name for item in model.graph.value_info}
    assert "orphan" not in {item.tensor_name for item in model.graph.quantization_annotation}
    actual = _run(
        decoded.snapshot,
        None,
        {"X": np.asarray((4.0, 5.0), dtype=np.float32)},
    )[0]
    assert np.array_equal(actual, expected)
    assert not verify_normalized_selected_snapshot(
        decoded.snapshot,
        snapshot,
        finding_code="selected-test-mismatch",
        path="selected.test",
        message="selected graph differs",
    )


def test_authorization_staleness_and_complete_correspondence_refuse() -> None:
    facts = _replay_facts(source_shape=(2, 3), folds=2, simd=3)
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())
    plan = plan_readable_names(snapshot)
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, replace(plan, transform_id="selected.unknown"))
    assert error.value.code == "selected.transform.unsupported"
    with pytest.raises(TypeError, match="positive integer"):
        replace(plan, transform_version=True)
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, replace(plan, expected_snapshot_digest="0" * 64))
    assert error.value.code == "selected.transform.snapshot_stale"

    unsupported = replace(
        plan,
        effects=replace(plan.effects, remove_initializers=("not-authorized",)),
    )
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, unsupported)
    assert error.value.code == "selected.transform.effects_unsupported"
    extra_metadata = replace(
        plan,
        effects=replace(
            plan.effects,
            set_metadata=(*plan.effects.set_metadata, ("review.extra", "bad")),
        ),
    )
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, extra_metadata)
    assert error.value.code == "selected.transform.metadata_unauthorized"
    incomplete = replace(plan, value_correspondence=plan.value_correspondence[:-1])
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, incomplete)
    assert error.value.code == "selected.transform.value_correspondence"

    nodes = list(plan.node_correspondence)
    nodes[0] = (nodes[0][0], nodes[1][1])
    nodes[1] = (nodes[1][0], plan.node_correspondence[0][1])
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, replace(plan, node_correspondence=tuple(nodes)))
    assert error.value.code == "selected.transform.node_correspondence"

    values = list(plan.value_correspondence)
    values[0] = (values[0][0], values[1][1])
    values[1] = (values[1][0], plan.value_correspondence[0][1])
    with pytest.raises(SelectedTransformError) as error:
        apply_selected_transform(snapshot, replace(plan, value_correspondence=tuple(values)))
    assert error.value.code == "selected.transform.value_correspondence"

    with pytest.raises(TypeError, match="two-item"):
        replace(plan, value_correspondence=(["mutable"],))  # type: ignore[arg-type]


def test_transformed_mvau_still_rejects_graph_corruption() -> None:
    facts, inputs, _activation, _weight, _expected = _mvau_facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    transformed = apply_selected_transform(snapshot, plan_readable_names(snapshot))
    values = dict(plan_readable_names(snapshot).value_correspondence)

    mutations = []
    operator = transformed.snapshot.model_copy()
    operator.graph.node[0].op_type = "Sub"
    mutations.append(operator)

    attribute = transformed.snapshot.model_copy()
    transpose_record = next(
        item
        for item in transformed.declaration.graph_nodes
        if item.node_id == "compute.weight.transpose"
    )
    transpose = attribute.graph.node[transpose_record.index]
    del transpose.attribute[:]
    transpose.attribute.extend((helper.make_attribute("perm", (1, 0, 2)),))
    mutations.append(attribute)

    constant = transformed.snapshot.model_copy()
    constant.set_initializer(values["shape_Y"], np.asarray((1, 8), dtype=np.int64))
    mutations.append(constant)

    order = transformed.snapshot.model_copy()
    matmul_record = next(
        item for item in transformed.declaration.graph_nodes if item.node_id == "compute.matmul"
    )
    matmul = order.graph.node[matmul_record.index]
    matmul.input[0], matmul.input[1] = matmul.input[1], matmul.input[0]
    mutations.append(order)

    for model in mutations:
        broken = build_selected_snapshot(model, transformed.declaration)
        with pytest.raises(SelectedGraphError) as error:
            reconstruct_selected_graph(broken.model_bytes)
        assert error.value.code in {
            "selected.construction.verification",
            "selected.graph.onnx_invalid",
        }


def _published_identity_snapshot(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        selected_registry_module,
        "DEFAULT_SELECTED_CONSTRUCTIONS",
        IDENTITY_CHAIN_CONSTRUCTIONS,
    )
    model, operation = configured_identity_chain(grouping=1)
    return operation.publish_selected(model).selected.snapshot


def _identity_construction_registry(construction) -> ConstructionRegistry:
    entries = dict(IDENTITY_CHAIN_CONSTRUCTIONS.entries)
    entries[(construction.family, construction.version)] = construction
    return ConstructionRegistry(entries, IDENTITY_CHAIN_CONSTRUCTIONS.choice_schemas)


def test_identity_elision_refuses_changed_surviving_region_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _published_identity_snapshot(monkeypatch)
    key = (snapshot.declaration.construction.family, snapshot.declaration.construction.version)
    construction = IDENTITY_CHAIN_CONSTRUCTIONS.entries[key]

    def changed_project(facts):
        network = construction.project(facts)
        if facts.construction.form != ELIDED_IDENTITY_FORM:
            return network
        consumer = network.node("consumer")
        main_input = consumer.region.input_interface("in")
        changed_input = replace(
            main_input,
            requirements=ScheduledInputRequirements(
                {
                    ((), (0,)): 2,
                    ((), (1,)): 2,
                }
            ),
        )
        changed_region = replace(
            consumer.region,
            inputs=tuple(
                changed_input if isinstance(item, InputInterface) and item.port.id == "in" else item
                for item in consumer.region.inputs
            ),
        )
        return replace(
            network,
            nodes=tuple(
                replace(node, region=changed_region) if node.id == "consumer" else node
                for node in network.nodes
            ),
        )

    constructions = _identity_construction_registry(replace(construction, project=changed_project))
    with pytest.raises(SelectedTransformError) as error:
        plan_equal_width_identity_elision(
            snapshot,
            "left",
            constructions=constructions,
            transforms=IDENTITY_CHAIN_TRANSFORMS,
        )
    assert error.value.code == "selected.transform.network_changed"


def test_identity_elision_refuses_changed_derived_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _published_identity_snapshot(monkeypatch)
    key = (snapshot.declaration.construction.family, snapshot.declaration.construction.version)
    construction = IDENTITY_CHAIN_CONSTRUCTIONS.entries[key]

    def changed_derive(identity, source, semantics, choices):
        facts = construction.derive_facts(identity, source, semantics, choices)
        if identity.form != ELIDED_IDENTITY_FORM:
            return facts
        return replace(
            facts,
            parameters=replace(facts.parameters, carrier_dtype=TensorProto.INT32),
        )

    constructions = _identity_construction_registry(
        replace(construction, derive_facts=changed_derive)
    )
    with pytest.raises(SelectedTransformError) as error:
        plan_equal_width_identity_elision(
            snapshot,
            "left",
            constructions=constructions,
            transforms=IDENTITY_CHAIN_TRANSFORMS,
        )
    assert error.value.code == "selected.transform.parameters_changed"


@pytest.mark.parametrize(
    "planner",
    (plan_readable_names, plan_bounded_cleanup),
    ids=("readable_names", "bounded_cleanup"),
)
def test_million_fold_replay_transform_stays_compact(planner, monkeypatch) -> None:
    facts = _replay_facts(source_shape=(2, 3), folds=1_048_576, simd=3)
    snapshot = construct_replay_snapshot(facts, ConstructionInputs())

    def refuse(*_args, **_kwargs):
        raise AssertionError("selected transform entered a compact-map iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)
    plan = planner(snapshot)
    decoded = apply_selected_transform(snapshot, plan)
    output = decoded.network.node("replay").region.output_interface("activation_out")
    assert output.port.beat_sequence.beat_count == 2 * 1_048_576
    assert output.port.beat_sequence.position_at(output.port.beat_sequence.beat_count - 1, 2) == (
        2 * 1_048_576 - 1,
        2,
    )
    assert len(decoded.snapshot.model_bytes) < 20_000
