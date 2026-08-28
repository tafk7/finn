# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 gate: Op -> KernelDeclaration pool -> selected Kernel -> Region."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import (
    Absent,
    Decided,
    DesignPoint,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.kernels import (
    NO_KERNEL,
    KernelDeclaration,
    KernelDemand,
    KernelSelection,
    SelectedKernel,
    bind_kernel,
    selected_kernel,
)
from finn.dataflow.region import DataflowRegion, Port
from finn.dataflow.resolution import RegionRef
from finn.dataflow.testing import DataflowOpConformanceCase, assert_dataflow_op_conforms
from finn.dataflow.spec_algebra import SpecAuthoringError, spec_declaration_paths
from dataflow.synthetic_kernel_op import (
    PAIRED_SELECTION,
    PairedKernelDataflowOp,
    PairedPaths,
    build_paired_kernel_op_spec,
)

_PATHS = PAIRED_SELECTION.paths


def _point(kernel_id: str, lanes: int, *, extent: int = 8) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    point = engine.start(
        engine.validate(build_paired_kernel_op_spec()),
        {
            PairedPaths.SOURCE_ID: "scope",
            PairedPaths.EXTENT: extent,
            PairedPaths.CLOCK: 5.0,
        },
    )
    result = engine.commit_assignments(
        point,
        {
            _PATHS.kernel: kernel_id,
            QualifiedPath(f"paired.{kernel_id}.lanes"): lanes,
        },
    )
    assert all(item.disposition in {"committed", "unchanged"} for item in result.outcomes)
    return engine, result.point


def test_either_kernel_derives_an_equal_region() -> None:
    even_engine, even = _point("even", 1)
    any_engine, other = _point("any", 1)
    left = even_engine.query_property(even, _PATHS.region)
    right = any_engine.query_property(other, _PATHS.region)
    assert isinstance(left, Decided) and isinstance(right, Decided)
    assert cast(DataflowRegion, left.value) == cast(DataflowRegion, right.value)


def test_equal_regions_retain_distinct_kernel_identities() -> None:
    even_engine, even = _point("even", 1)
    any_engine, other = _point("any", 1)
    left = selected_kernel(even_engine, PAIRED_SELECTION, even)
    right = selected_kernel(any_engine, PAIRED_SELECTION, other)
    assert isinstance(left, Decided) and isinstance(right, Decided)
    assert left.value == SelectedKernel("paired.compute", "even", "1")
    assert right.value == SelectedKernel("paired.compute", "any", "1")
    assert left.value != right.value


def test_kernel_local_decisions_apply_only_to_the_selected_kernel() -> None:
    engine, point = _point("even", 2)
    selected = engine.decision_state(point, QualifiedPath("paired.even.lanes"))
    unselected = engine.decision_state(point, QualifiedPath("paired.any.lanes"))
    assert isinstance(selected, Decided)
    assert selected.value.status == "committed"
    assert isinstance(unselected, Absent)


def test_unselected_kernel_domains_are_not_reachable() -> None:
    engine = Engine()
    space = engine.validate(build_paired_kernel_op_spec())
    point = engine.start(
        space,
        {
            PairedPaths.SOURCE_ID: "scope",
            PairedPaths.EXTENT: 8,
            PairedPaths.CLOCK: 5.0,
        },
    )
    committed = engine.commit_assignments(point, {_PATHS.kernel: "any"})
    # 3 belongs to the "any" pool member only; it is unavailable under "even".
    accepted = engine.commit_assignments(committed.point, {QualifiedPath("paired.any.lanes"): 3})
    assert accepted.outcomes[0].disposition in {"committed", "unchanged"}
    rejected = engine.commit_assignments(point, {QualifiedPath("paired.even.lanes"): 3})
    assert rejected.outcomes[0].disposition not in {"committed", "unchanged"}


def _constraint(engine: Engine, point: DesignPoint, path: str) -> object:
    assessment = engine.evaluate_constraints(point, (QualifiedPath(path),))
    return assessment.answers[QualifiedPath(path)]


def test_the_pool_owns_source_admission() -> None:
    engine, point = _point("any", 1, extent=7)
    admitted = _constraint(engine, point, "constraint.paired.any.extent_admitted")
    assert isinstance(admitted, Decided) and admitted.value is True
    engine, point = _point("even", 1, extent=7)
    refused = _constraint(engine, point, "constraint.paired.even.extent_admitted")
    assert isinstance(refused, Decided) and refused.value is False


def test_providers_do_not_create_another_semantic_choice() -> None:
    even = PAIRED_SELECTION.kernel("even")
    assert tuple(provider.id for provider in even.providers) == ("even.rtl", "even.hls")
    assert all(provider.kernel_id == "even" for provider in even.providers)
    spec = build_paired_kernel_op_spec()
    decision_paths = {str(item.path) for item in spec.decisions}
    assert not any("provider" in path or "binding" in path for path in decision_paths)


def test_the_selection_declares_exactly_one_identity_decision() -> None:
    spec = build_paired_kernel_op_spec()
    identity_decisions = [item for item in spec.decisions if item.path == _PATHS.kernel]
    assert len(identity_decisions) == 1
    assert set(PAIRED_SELECTION.candidate_ids) == {"even", "any"}


def test_demands_follow_the_selected_kernel() -> None:
    engine, point = _point("even", 1)
    demand = engine.query_property(point, _PATHS.demand("parameter"))
    assert isinstance(demand, Decided)
    assert cast(Port, demand.value).operand.id == "w"
    engine, point = _point("any", 1)
    absent = engine.query_property(point, _PATHS.demand("parameter"))
    assert isinstance(absent, Absent)


def test_region_is_unresolved_before_the_pool_is_committed() -> None:
    engine = Engine()
    point = engine.start(
        engine.validate(build_paired_kernel_op_spec()),
        {
            PairedPaths.SOURCE_ID: "scope",
            PairedPaths.EXTENT: 8,
            PairedPaths.CLOCK: 5.0,
        },
    )
    assert isinstance(engine.query_property(point, _PATHS.region), Unresolved)


def test_repeated_placement_of_one_kernel_does_not_collide() -> None:
    kernel = PAIRED_SELECTION.kernel("even")
    left = kernel.place("left")
    right = kernel.place("right")
    selection = KernelSelection("placed", (left,))
    other = KernelSelection("placed_other", (right,))
    left_paths = {str(path) for path in spec_declaration_paths(selection.build_spec())}
    right_paths = {str(path) for path in spec_declaration_paths(other.build_spec())}
    assert not left_paths & right_paths
    assert left.region_path != right.region_path
    assert left.id == right.id == "even"


def test_an_optional_pool_may_select_no_kernel() -> None:
    selection = KernelSelection(
        "optional.supply", (PAIRED_SELECTION.kernel("even").place("supply"),), optional=True
    )
    assert selection.candidate_ids == ("even", NO_KERNEL)


def test_a_kernel_must_declare_its_region_property() -> None:
    even = PAIRED_SELECTION.kernel("even")
    with pytest.raises(SpecAuthoringError) as caught:
        KernelDeclaration("broken", "1", even.spec, QualifiedPath("semantic.paired.missing"))
    assert "kernel-region-path-missing" in {issue.code for issue in caught.value.issues}


def test_a_demand_must_name_a_declared_port_property() -> None:
    even = PAIRED_SELECTION.kernel("even")
    with pytest.raises(SpecAuthoringError) as caught:
        KernelDeclaration(
            "broken",
            "1",
            even.spec,
            even.region_path,
            demands=(KernelDemand("parameter", QualifiedPath("semantic.paired.missing")),),
        )
    assert "kernel-demand-path-missing" in {issue.code for issue in caught.value.issues}


def test_a_pool_refuses_duplicate_kernel_identities() -> None:
    even = PAIRED_SELECTION.kernel("even")
    with pytest.raises(SpecAuthoringError) as caught:
        KernelSelection("duplicate", (even, even))
    assert "kernel-id-duplicate" in {issue.code for issue in caught.value.issues}


def _paired_model(extent: int = 8) -> ModelWrapper:
    node = helper.make_node(
        "PairedKernelDataflowOp",
        ["x"],
        ["y"],
        name="paired0",
        domain="dataflow.synthetic_kernel_op",
        dataflow_scope_id="paired-scope",
    )
    graph = helper.make_graph(
        [node],
        "paired",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [extent])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [extent])],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-pool-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("dataflow.synthetic_kernel_op", 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["INT8"])
    model.set_tensor_datatype("y", DataType["INT8"])
    return model


@dataclass
class _BuildConfig:
    synth_clk_period_ns: float


@pytest.mark.parametrize("kernel_id,lanes", [("even", 2), ("any", 1)])
def test_a_pool_backed_operation_passes_the_conformance_harness(
    tmp_path: Path, kernel_id: str, lanes: int
) -> None:
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_paired_model(),
            node_name="paired0",
            operation_type=PairedKernelDataflowOp,
            config=_BuildConfig(5.0),
            complete_assignments={
                _PATHS.kernel: kernel_id,
                QualifiedPath(f"paired.{kernel_id}.lanes"): lanes,
            },
            rejected_assignments={_PATHS.kernel: "absent-kernel"},
            reload_path=tmp_path / f"paired-{kernel_id}.onnx",
            stale_config=_BuildConfig(3.0),
        )
    )
    assert isinstance(result.original.result, RegionRef)
    assert result.original.result.region_id == f"paired.{kernel_id}"
    assert result.restored.result == result.original.result


def test_a_selected_kernel_binds_to_its_region_demands_and_providers() -> None:
    engine, point = _point("even", 2)
    instance = bind_kernel(engine, PAIRED_SELECTION, point)
    assert isinstance(instance, Decided)
    bound = instance.value
    assert bound.id == "even"
    assert bound.selection == PAIRED_SELECTION.name
    assert bound.identity == SelectedKernel("paired.compute", "even", "1")
    assert bound.assignments == {QualifiedPath("paired.even.lanes"): 2}
    region = engine.query_property(point, _PATHS.region)
    assert isinstance(region, Decided)
    assert bound.region == region.value
    assert set(bound.demands) == {"parameter"}
    assert tuple(item.id for item in bound.providers) == ("even.rtl", "even.hls")


def test_the_bound_instance_carries_only_its_own_local_choices() -> None:
    engine, point = _point("any", 3)
    instance = bind_kernel(engine, PAIRED_SELECTION, point)
    assert isinstance(instance, Decided)
    assert instance.value.assignments == {QualifiedPath("paired.any.lanes"): 3}
    assert instance.value.demands == {}
