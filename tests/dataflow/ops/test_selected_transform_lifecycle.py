# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle coverage for production and test-only selected transforms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx.reference import ReferenceEvaluator  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

import finn.dataflow.ops.selected_registry as selected_registry_module
import finn.dataflow.ops.selected_transform_registry as transform_registry_module
from finn.dataflow._engine import Decided
from finn.dataflow.ops.mvau.computation import AccumulationMode
from finn.dataflow.ops.mvau.designs.supply import WeightSupply
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.reconstruction import rebind_selected_graph
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.selected import (
    ConstructionIdentity,
    ConstructionRegistry,
    InterfaceDirection,
    SelectedGraphSnapshot,
    build_selected_snapshot,
    reconstruct_selected_graph,
)
from finn.dataflow.ops.selected_transforms import (
    SelectedTransformError,
    SelectedTransformPlan,
    SelectedTransformRegistry,
    apply_selected_transform,
    plan_bounded_cleanup,
    plan_equal_width_identity_elision,
    plan_readable_names,
)

from dataflow.ops.selected_transform_fixtures import (
    ACTIVATION_KEY,
    IDENTITY_ELIDED_FORM,
    IdentityChainOp,
    combined_construction_registry,
    combined_transform_registry,
    configured_identity_chain,
)

TransformPlanner = Callable[[SelectedGraphSnapshot], SelectedTransformPlan]

_SOURCE_SEMANTICS_FIXTURES = import_module("dataflow.ops.mvau.test_source_semantics")
_OP_FIXTURES = import_module("dataflow.ops.test_dataflow_op")


def _configured_replay_source() -> tuple[ModelWrapper, ActivationReplayOp]:
    return cast(
        "tuple[ModelWrapper, ActivationReplayOp]",
        _OP_FIXTURES._configured_replay(simd=4),
    )


def _install_fixture_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ConstructionRegistry, SelectedTransformRegistry]:
    constructions = combined_construction_registry()
    transforms = combined_transform_registry()
    monkeypatch.setattr(
        selected_registry_module,
        "DEFAULT_SELECTED_CONSTRUCTIONS",
        constructions,
    )
    monkeypatch.setattr(
        transform_registry_module,
        "DEFAULT_SELECTED_TRANSFORMS",
        transforms,
    )
    return constructions, transforms


def _run(snapshot: SelectedGraphSnapshot, values: np.ndarray) -> np.ndarray:
    return ReferenceEvaluator(snapshot.model_copy().model).run(None, {"X": values})[0]


@pytest.mark.parametrize("region_id", ("left", "right"))
def test_real_identity_chain_elision_survives_the_complete_lifecycle(
    region_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions, transforms = _install_fixture_registries(monkeypatch)
    source_model, operation = configured_identity_chain(grouping=1)
    published = operation.publish_selected(source_model)
    assert isinstance(published.operation, IdentityChainOp)
    original = published.selected.snapshot
    original_bytes = original.model_bytes

    plan = plan_equal_width_identity_elision(
        original,
        region_id,
        constructions=constructions,
        transforms=transforms,
    )
    transformed = apply_selected_transform(
        original,
        plan,
        constructions=constructions,
        transforms=transforms,
    )

    values = np.asarray([3.0, -2.0], dtype=np.float32)
    assert np.array_equal(_run(original, values), values)
    assert np.array_equal(_run(transformed.snapshot, values), values)
    assert original.model_bytes == original_bytes
    assert transformed.declaration.construction.form == IDENTITY_ELIDED_FORM
    assert transformed.declaration.construction.form_version == 1
    assert transformed.declaration.construction.form_arguments == (("region_id", region_id),)
    assert region_id not in {node.id for node in transformed.network.nodes}
    assert transformed.network.boundaries[-1].endpoint.node_id == "consumer"

    bypass_target = "right" if region_id == "left" else "consumer"
    bypass_source = "producer" if region_id == "left" else "left"
    bypass = next(
        edge
        for edge in transformed.network.edges
        if edge.source.node_id == bypass_source and edge.sinks[0].endpoint.node_id == bypass_target
    )
    assert bypass.source.port_id == "out"
    assert bypass.sinks[0].endpoint.port_id == "in"
    target_binding = next(
        item
        for item in transformed.declaration.interface_bindings
        if item.interface.node_id == bypass_target and item.interface.direction.value == "input"
    )
    assert target_binding.graph_value == ("P" if region_id == "left" else "L")
    assert all(anchor.owner != f"{region_id}.identity" for anchor in target_binding.anchors)

    path = tmp_path / f"identity-chain-{region_id}.onnx"
    path.write_bytes(transformed.snapshot.model_bytes)
    restored = reconstruct_selected_graph(path, constructions=constructions)
    assert restored == transformed
    rebound = rebind_selected_graph(
        published.operation,
        restored.snapshot,
        constructions=constructions,
    )
    assert rebound.network == transformed.network
    assert rebound.selection_facts == transformed.selection_facts


def test_width_changing_identity_is_not_elided_but_equal_width_peer_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions, transforms = _install_fixture_registries(monkeypatch)
    source_model, operation = configured_identity_chain(grouping=2)
    published = operation.publish_selected(source_model)
    original = published.selected.snapshot

    with pytest.raises(
        SelectedTransformError,
        match="identity_width_change_requires_adapter",
    ):
        plan_equal_width_identity_elision(
            original,
            "left",
            constructions=constructions,
            transforms=transforms,
        )

    plan = plan_equal_width_identity_elision(
        original,
        "right",
        constructions=constructions,
        transforms=transforms,
    )
    transformed = apply_selected_transform(
        original,
        plan,
        constructions=constructions,
        transforms=transforms,
    )
    assert {node.id for node in transformed.network.nodes} == {
        "producer",
        "left",
        "consumer",
    }
    assert (
        rebind_selected_graph(
            published.operation,
            transformed.snapshot,
            constructions=constructions,
        ).network
        == transformed.network
    )


@pytest.mark.parametrize(
    (
        "region_id",
        "before_value",
        "after_value",
        "after_owner",
        "before_path",
        "after_path",
    ),
    (
        (
            "left",
            "L",
            "P",
            "producer.identity",
            ("producer.identity", "left.identity"),
            ("producer.identity",),
        ),
        (
            "right",
            "R",
            "L",
            "left.identity",
            ("producer.identity", "left.identity", "right.identity"),
            ("producer.identity", "left.identity"),
        ),
    ),
)
def test_identity_elision_repairs_initializer_source_binding_and_supply_path(
    region_id: str,
    before_value: str,
    after_value: str,
    after_owner: str,
    before_path: tuple[str, ...],
    after_path: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions, transforms = _install_fixture_registries(monkeypatch)
    expected = np.asarray([3.0, -2.0], dtype=np.float32)
    source_model, operation = configured_identity_chain(
        grouping=1,
        reference_region=region_id,
        activation=expected,
    )
    published = operation.publish_selected(source_model)
    original = published.selected
    source_binding = next(
        item for item in original.declaration.source_bindings if item.source == ACTIVATION_KEY
    )
    assert source_binding.graph_value == before_value
    assert len(source_binding.anchors) == 1
    assert source_binding.anchors[0].owner == f"{region_id}.identity"
    assert original.declaration.supplies[0].root_graph_value == "X_initializer"
    assert original.declaration.supplies[0].derivation_nodes == before_path
    assert np.array_equal(
        ReferenceEvaluator(original.snapshot.model_copy().model).run(None, {})[0],
        expected,
    )

    transformed = apply_selected_transform(
        original.snapshot,
        plan_equal_width_identity_elision(
            original.snapshot,
            region_id,
            constructions=constructions,
            transforms=transforms,
        ),
        constructions=constructions,
        transforms=transforms,
    )
    repaired = next(
        item for item in transformed.declaration.source_bindings if item.source == ACTIVATION_KEY
    )
    assert repaired.graph_value == after_value
    assert repaired.anchors[0].owner == after_owner
    assert transformed.declaration.supplies[0].root_graph_value == "X_initializer"
    assert transformed.declaration.supplies[0].derivation_nodes == after_path
    reference = next(
        item
        for item in transformed.declaration.interface_bindings
        if item.interface.node_id == "consumer" and item.interface.interface_id is None
    )
    assert reference.graph_value == after_value
    assert np.array_equal(
        ReferenceEvaluator(transformed.snapshot.model_copy().model).run(None, {})[0],
        expected,
    )

    path = tmp_path / f"identity-chain-source-{region_id}.onnx"
    path.write_bytes(transformed.snapshot.model_bytes)
    restored = reconstruct_selected_graph(path, constructions=constructions)
    rebound = rebind_selected_graph(
        published.operation,
        restored.snapshot,
        constructions=constructions,
    )
    assert rebound.network == transformed.network
    assert rebound.selection_facts == transformed.selection_facts


@pytest.mark.parametrize(
    "construction",
    (
        lambda value: replace(value, form_version=2),
        lambda value: replace(value, form_arguments=(("region_id", "consumer"),)),
        lambda value: replace(
            value,
            form_arguments=(("extra", 1), ("region_id", "left")),
        ),
    ),
    ids=("wrong_version", "unsupported_region", "extra_argument"),
)
def test_identity_elision_form_arguments_remain_bounded(
    construction: Callable[[ConstructionIdentity], ConstructionIdentity],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions, transforms = _install_fixture_registries(monkeypatch)
    source_model, operation = configured_identity_chain(grouping=1)
    original = operation.publish_selected(source_model).selected.snapshot
    transformed = apply_selected_transform(
        original,
        plan_equal_width_identity_elision(
            original,
            "left",
            constructions=constructions,
            transforms=transforms,
        ),
        constructions=constructions,
        transforms=transforms,
    )
    malformed = build_selected_snapshot(
        transformed.snapshot.model_copy(),
        replace(
            transformed.declaration,
            construction=construction(transformed.declaration.construction),
        ),
    )
    with pytest.raises(ValueError, match="identity_elided"):
        reconstruct_selected_graph(malformed.model_bytes, constructions=constructions)


def _mvau_for_mode(
    mode: AccumulationMode,
    supply: WeightSupply,
) -> tuple[ModelWrapper, MvauDataflowOp]:
    if mode is AccumulationMode.INTEGER:
        model = cast(ModelWrapper, _SOURCE_SEMANTICS_FIXTURES._model())
    elif mode is AccumulationMode.XNOR_POPCOUNT:
        model = cast(
            ModelWrapper,
            _SOURCE_SEMANTICS_FIXTURES._model(
                binary_xnor=True,
                activation_type="BINARY",
                weight_type="BINARY",
            ),
        )
    else:
        model = cast(
            ModelWrapper,
            _SOURCE_SEMANTICS_FIXTURES._model(
                activation_type="BIPOLAR",
                weight_type="BIPOLAR",
            ),
        )
    return cast(
        "tuple[ModelWrapper, MvauDataflowOp]",
        _OP_FIXTURES._configured_mvau(model, supply=supply, pe=2, simd=2),
    )


@pytest.mark.parametrize(
    "planner",
    (plan_readable_names, plan_bounded_cleanup),
    ids=("readable_names", "bounded_cleanup"),
)
def test_transformed_replay_rebinds_to_the_real_source(
    planner: TransformPlanner,
) -> None:
    _model, operation = _configured_replay_source()
    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    transformed = apply_selected_transform(answer.value, planner(answer.value))
    rebound = rebind_selected_graph(operation, transformed.snapshot)
    assert rebound.network == transformed.network
    assert rebound.selection_facts == transformed.selection_facts


@pytest.mark.parametrize("mode", tuple(AccumulationMode))
@pytest.mark.parametrize("supply", tuple(WeightSupply))
@pytest.mark.parametrize(
    "planner",
    (plan_readable_names, plan_bounded_cleanup),
    ids=("readable_names", "bounded_cleanup"),
)
def test_transformed_mvau_profiles_and_supplies_rebind_to_the_real_source(
    mode: AccumulationMode,
    supply: WeightSupply,
    planner: TransformPlanner,
) -> None:
    _model, operation = _mvau_for_mode(mode, supply)
    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    transformed = apply_selected_transform(answer.value, planner(answer.value))
    rebound = rebind_selected_graph(operation, transformed.snapshot)
    assert rebound.network == transformed.network
    assert rebound.selection_facts == transformed.selection_facts


def test_transformed_replay_runs_two_activation_basis_invocations_after_reload(
    tmp_path: Path,
) -> None:
    _model, operation = _configured_replay_source()
    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    transformed = apply_selected_transform(answer.value, plan_readable_names(answer.value))
    path = tmp_path / "transformed-replay.onnx"
    path.write_bytes(transformed.snapshot.model_bytes)
    restored = reconstruct_selected_graph(path)
    assert rebind_selected_graph(operation, restored.snapshot).network == restored.network

    model = restored.snapshot.model_copy()
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    evaluator = ReferenceEvaluator(model.model)
    activations = (
        np.eye(8, dtype=np.float32)[[0, 1]],
        np.eye(8, dtype=np.float32)[[3, 7]],
    )
    for activation in activations:
        actual = evaluator.run([output_name], {input_name: activation})[0]
        assert np.array_equal(actual, np.repeat(activation, 4, axis=0))


def _basis_mvau_values(
    mode: AccumulationMode,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    if mode is AccumulationMode.INTEGER:
        weight = np.asarray(
            [
                [1, -2, 3, 0],
                [0, 2, -1, 4],
                [-3, 1, 2, 1],
                [2, 0, -2, 3],
                [1, 1, 0, -1],
                [4, -1, 1, 2],
                [-2, 3, 2, 0],
                [3, 2, -4, 1],
            ],
            dtype=np.float32,
        )
        activations = (
            np.eye(8, dtype=np.float32)[[0, 3]],
            np.eye(8, dtype=np.float32)[[5, 7]],
        )
    elif mode is AccumulationMode.XNOR_POPCOUNT:
        weight = np.asarray(
            [[(row + column) % 2 for column in range(4)] for row in range(8)],
            dtype=np.float32,
        )
        first = np.zeros((2, 8), dtype=np.float32)
        first[0, 0] = 1
        first[1, 3] = 1
        second = np.zeros((2, 8), dtype=np.float32)
        second[0, 5] = 1
        second[1, 7] = 1
        activations = (first, second)
    else:
        weight = np.asarray(
            [[1 if (row + column) % 3 else -1 for column in range(4)] for row in range(8)],
            dtype=np.float32,
        )
        first = -np.ones((2, 8), dtype=np.float32)
        first[0, 0] = 1
        first[1, 3] = 1
        second = -np.ones((2, 8), dtype=np.float32)
        second[0, 5] = 1
        second[1, 7] = 1
        activations = (first, second)
    return weight, activations


@pytest.mark.parametrize("mode", tuple(AccumulationMode))
def test_transformed_embedded_mvau_runs_two_basis_invocations_with_full_xr_and_y(
    mode: AccumulationMode,
    tmp_path: Path,
) -> None:
    weight, activations = _basis_mvau_values(mode)
    if mode is AccumulationMode.INTEGER:
        source_model = cast(
            ModelWrapper,
            _SOURCE_SEMANTICS_FIXTURES._model(weights=weight),
        )
    elif mode is AccumulationMode.XNOR_POPCOUNT:
        source_model = cast(
            ModelWrapper,
            _SOURCE_SEMANTICS_FIXTURES._model(
                binary_xnor=True,
                activation_type="BINARY",
                weight_type="BINARY",
                weights=weight,
            ),
        )
    else:
        source_model = cast(
            ModelWrapper,
            _SOURCE_SEMANTICS_FIXTURES._model(
                activation_type="BIPOLAR",
                weight_type="BIPOLAR",
                weights=weight,
            ),
        )
    _model, operation = cast(
        "tuple[ModelWrapper, MvauDataflowOp]",
        _OP_FIXTURES._configured_mvau(
            source_model,
            supply=WeightSupply.EMBEDDED,
            pe=2,
            simd=2,
        ),
    )
    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    transformed = apply_selected_transform(answer.value, plan_readable_names(answer.value))
    path = tmp_path / f"transformed-mvau-{mode.value}.onnx"
    path.write_bytes(transformed.snapshot.model_bytes)
    restored = reconstruct_selected_graph(path)
    assert rebind_selected_graph(operation, restored.snapshot).network == restored.network

    model = restored.snapshot.model_copy()
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    xr_name = next(
        item.graph_value
        for item in restored.declaration.interface_bindings
        if item.interface.node_id == "replay"
        and item.interface.direction is InterfaceDirection.OUTPUT
    )
    evaluator = ReferenceEvaluator(model.model)
    for activation in activations:
        actual_xr, actual_y = evaluator.run(
            [xr_name, output_name],
            {input_name: activation},
        )
        expected_y = (
            activation @ weight
            if mode is AccumulationMode.INTEGER
            else np.sum(
                activation[:, :, None] == weight[None, :, :],
                axis=1,
            ).astype(np.float32)
        )
        assert np.array_equal(actual_xr, np.repeat(activation, 2, axis=0))
        assert np.array_equal(actual_y, expected_y)


def test_production_registry_does_not_advertise_identity_elision() -> None:
    registry = transform_registry_module.DEFAULT_SELECTED_TRANSFORMS
    for authorizations in registry.entries.values():
        assert all(
            authorization.transform_id != "selected.elide_equal_width_identity"
            for authorization in authorizations
        )
