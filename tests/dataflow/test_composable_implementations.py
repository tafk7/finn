# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.kernels.test_module_build_spec import UnavailableModule
from dataflow.ops.mvau.test_dot_product_design import DESIGN_INPUTS, Problem_, _occurrence
from dataflow.physical_fixture import configure, source_model
from finn.dataflow._engine import Absent, Answer, Decided, Unresolved
from finn.dataflow.artifacts.build import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.designs.design import DataflowDesign
from finn.dataflow.designs.physical import DesignPhysicalRelation
from finn.dataflow.model.composition import NetworkResult, RegionResult
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign, WeightSupply
from finn.dataflow.ops.mvau.computation import (
    AccumulationMode,
    ActivationMode,
    MvauComputationProfile,
)
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.ops.physical import (
    associate_physical_use,
    authorize_component_use,
    capture_local_physical,
    capture_local_relation,
    install_compiler_physical_component,
    materialize_local_physical,
    prepare_local_physical,
    validate_compiler_physical_use,
)
from finn.dataflow.space.capabilities import implementation_identity
from finn.dataflow.space.occurrence import ProjectionAssessment
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    constraint,
    derived,
    reject,
)


def test_real_replay_dotp_and_composite_publish_typed_logical_capabilities() -> None:
    design = _occurrence(
        repetitions=1,
        matrix_width=4,
        matrix_height=4,
        activation="INT3",
        weight="INT3",
        accumulator="INT16",
        pe=2,
        simd=2,
    )
    logical = design.assess_view("logical").accepted_answer
    assert isinstance(logical, Decided)
    assert isinstance(logical.value, NetworkResult)
    assert tuple(child.use_path.value for child in logical.value.children) == (
        "replay",
        "compute",
    )
    replay = design.kernel("replay")
    assert isinstance(replay, Decided)
    leaf = replay.value.assess_view("logical").accepted_answer
    assert isinstance(leaf, Decided) and isinstance(leaf.value, RegionResult)
    assert implementation_identity(replay.value).family == "replay_buffer"
    assert {"dataflow", "logical", "physical"} <= set(replay.value.capability_names())


class _ViewLeaf(Space):
    id = "view_leaf"
    version = "1"
    value = Input(int)
    estimator = Input(str, allow_absent=True)
    ready = Readiness()
    logical = Projection(value, readiness=ready)
    physical = Projection(value, readiness=ready)
    cost_ready = Readiness()
    cost = Projection(estimator, readiness=cost_ready)
    exports = (value, estimator)


class _ViewRoot(Space):
    value = Problem(int)
    estimator = Decision(str, values=("area", "timing"))
    leaf = Subspace(_ViewLeaf, value=value, estimator=estimator)


def test_third_view_and_lazy_child_input_do_not_block_unrelated_views() -> None:
    root = _ViewRoot.start({_ViewRoot.value: 7}, namespace="views")
    assert root.leaf.logical.accepted_answer == Decided(7)
    assert root.leaf.physical.accepted_answer == Decided(7)
    assert isinstance(root.leaf.cost.accepted_answer, Unresolved)
    assert root.assign(_ViewRoot.estimator, "area").leaf.cost.accepted_answer == Decided("area")


class _PendingView(Space):
    value = Decision(int, values=(1, 2))
    extra = Readiness()
    result = Projection(value, readiness=extra)


class _PendingConstraintView(Space):
    value = Problem(int)
    limit = Decision(int, values=(1, 2))

    @constraint(value=value, limit=limit)
    def within(*, value: int, limit: int) -> bool:
        return value <= limit

    accepts = ConstraintGroup(within)
    extra = Readiness()
    result = Projection(value, readiness=extra, constraints=accepts)


def test_view_readiness_includes_output_and_acceptance_prerequisites() -> None:
    pending = _PendingView.start({}, namespace="pending").result
    assert pending.readiness.ready is None
    assert isinstance(pending.output, Unresolved)

    constrained = _PendingConstraintView.start(
        {_PendingConstraintView.value: 1}, namespace="constrained"
    ).result
    assert constrained.readiness.ready is None
    assert isinstance(constrained.accepted_answer, Unresolved)


class _ConditionalLeaf(Space):
    required = Input(int)
    ready = Readiness()
    logical = Projection(required, readiness=ready)
    exports = (required,)


class _ConditionalRoot(Space):
    enabled = Decision(bool, values=(False, True))
    missing = Decision(int, values=(1, 2))
    child = Subspace(_ConditionalLeaf, when=enabled, required=missing)


def test_inactive_child_does_not_resolve_bound_inputs() -> None:
    root = _ConditionalRoot.start({}, namespace="conditional")
    inactive = root.assign(_ConditionalRoot.enabled, False)
    assert isinstance(inactive.child.logical.accepted_answer, Absent)
    active = root.assign(_ConditionalRoot.enabled, True)
    assert isinstance(active.child.logical.accepted_answer, Unresolved)


class _IndependentLogicalDesign(DotProductDesign):
    id = "independent_logical_dot_product"
    logical_mode = Decision(str, values=("accept", "reject"))

    @constraint(mode=logical_mode)
    def logical_mode_supported(*, mode: str) -> object:
        return True if mode == "accept" else reject("logical-mode-rejected", "logical-only")

    dataflow_support = ConstraintGroup(
        *DotProductDesign.dataflow_support.constraints,
        logical_mode_supported,
        name="logical_support",
    )


class _IndependentRoot(Problem_):
    design = Subspace(
        _IndependentLogicalDesign,
        name="dot_product",
        **{name: cast("object", getattr(Problem_, name)) for name in DESIGN_INPUTS},
    )


def _independent_design() -> _IndependentLogicalDesign:
    root = _IndependentRoot.start(
        {
            _IndependentRoot.repetitions: 1,
            _IndependentRoot.matrix_width: 4,
            _IndependentRoot.matrix_height: 4,
            _IndependentRoot.activation_type: DataType["INT3"],
            _IndependentRoot.weight_type: DataType["INT3"],
            _IndependentRoot.accumulator_type: DataType["INT16"],
            _IndependentRoot.output_type: DataType["INT16"],
            _IndependentRoot.narrow_weights: True,
            _IndependentRoot.computation_profile: MvauComputationProfile(
                AccumulationMode.INTEGER, ActivationMode.NONE
            ),
            _IndependentRoot.target_dsp: DspBlock.DSP58,
            _IndependentRoot.clock_period_ns: 4.0,
            _IndependentRoot.initializer_present: True,
        },
        namespace="independent",
    )
    design = cast(_IndependentLogicalDesign, root.design)
    design = cast(
        _IndependentLogicalDesign,
        design.assign(WeightedDotProductDesign.pe, 2)
        .assign(WeightedDotProductDesign.simd, 2)
        .assign(DotProductDesign.weight_supply, WeightSupply.EXTERNAL),
    )
    design = cast(_IndependentLogicalDesign, design.compute.select("dotp_axi").root.design)
    compute = design.kernel("compute")
    assert isinstance(compute, Decided)
    design = cast(
        _IndependentLogicalDesign,
        compute.value.assign(DotpAxiKernel.compute_pumping, False).root.design,
    )
    return design


def test_local_physical_capture_ignores_unresolved_and_rejected_logical_only_choice(
    tmp_path,
) -> None:
    design = _independent_design()
    unresolved_capture = capture_local_physical(design)
    assert isinstance(design.dataflow.accepted_answer, Unresolved)
    with pytest.raises(DataflowOpError, match="relation"):
        capture_local_relation(design, unresolved_capture)

    rejected = cast(
        _IndependentLogicalDesign,
        design.assign(_IndependentLogicalDesign.logical_mode, "reject"),
    )
    rejected_capture = capture_local_physical(rejected)
    assert rejected_capture.point_fingerprint == unresolved_capture.point_fingerprint
    assert isinstance(rejected.dataflow.accepted_answer, Absent)
    with pytest.raises(DataflowOpError, match="relation"):
        capture_local_relation(rejected, rejected_capture)

    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_local_physical(
        unresolved_capture,
        roots={"finnlib": __import__("pathlib").Path("deps/finnlib").resolve()},
        template_roots=(
            __import__("pathlib").Path("src/finn/dataflow/designs/templates").resolve(),
        ),
        blobs=store,
    )
    built = materialize_local_physical(prepared, store=store)
    assert built.capture_fingerprint == unresolved_capture.point_fingerprint

    accepted = cast(
        _IndependentLogicalDesign,
        design.assign(_IndependentLogicalDesign.logical_mode, "accept"),
    )
    relation = capture_local_relation(accepted, capture_local_physical(accepted))
    assert relation.logical_fingerprint


class _CaptureSpace(Space):
    id = "capture_dependency"
    version = "1"
    supported_mode = Decision(int, values=(1, 2))
    private_cost = Decision(int, values=(10, 20))

    @derived(ModuleBuildRequirements)
    def module() -> ModuleBuildRequirements:
        return ModuleBuildRequirements(
            "capture_dependency",
            "1",
            (),
            ModuleABIRequirements(FixedModuleName("capture_dependency"), (), ()),
            (),
            (),
        )

    @constraint(mode=supported_mode)
    def supported(*, mode: int) -> bool:
        return mode in (1, 2)

    accepts = ConstraintGroup(supported)
    ready = Readiness(properties=(module,), constraints=accepts)
    physical = Projection(module, readiness=ready, constraints=accepts)


def test_local_capture_tracks_consumed_closure_but_excludes_unrelated_view_choice() -> None:
    root = _CaptureSpace.start({}, namespace="capture")
    first = root.assign(_CaptureSpace.supported_mode, 1)
    second = root.assign(_CaptureSpace.supported_mode, 2)
    first_capture = capture_local_physical(first)
    second_capture = capture_local_physical(second)
    assert first_capture.requirements == second_capture.requirements
    assert first_capture.point_fingerprint != second_capture.point_fingerprint
    assert first_capture.dependencies != second_capture.dependencies

    unrelated = first.assign(_CaptureSpace.private_cost, 10)
    unrelated_capture = capture_local_physical(unrelated)
    assert unrelated_capture.point_fingerprint == first_capture.point_fingerprint
    assert unrelated_capture.dependencies == first_capture.dependencies


def test_compiler_association_is_per_use_and_revalidates_current_source() -> None:
    model, build, context = source_model()
    left = configure(MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context))
    right = configure(MvauDataflowOp(model.graph.node[1]).bind(model, build, graph_context=context))
    left_design = cast(DataflowDesign, left.selected_implementation())
    right_design = cast(DataflowDesign, right.selected_implementation())
    left_local = capture_local_physical(left_design)
    right_local = capture_local_physical(right_design)
    assert left_local != right_local
    assert left_local.physical_fingerprint == right_local.physical_fingerprint
    left_relation = capture_local_relation(left_design, left_local)
    use = associate_physical_use(
        left,
        left_design,
        left_local,
        left_relation,
        model=model,
        build=build,
        graph_context=context,
    )
    assert (
        validate_compiler_physical_use(
            left,
            left_design,
            use,
            model=model,
            build=build,
            graph_context=context,
        )
        == ()
    )
    model.set_initializer("pair_left_W", np.zeros((4, 4), dtype=np.float32))
    assert validate_compiler_physical_use(
        left,
        left_design,
        use,
        model=model,
        build=build,
        graph_context=context,
    )


def test_plain_space_completes_source_selection_build_association_and_installation(
    tmp_path,
) -> None:
    model, build, context = source_model()
    production = configure(
        MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context)
    )
    production_design = cast(DataflowDesign, production.selected_implementation())
    production_local = capture_local_physical(production_design)
    production_relation = capture_local_relation(production_design, production_local)
    logical = production_design.assess_view("logical").accepted_answer
    assert isinstance(logical, Decided)

    class PlainComposite(Space):
        id = "plain_composite"
        version = "1"

        @derived(ModuleBuildRequirements)
        def requirements() -> ModuleBuildRequirements:
            return production_local.requirements

        @derived(NetworkResult)
        def logical_value() -> NetworkResult:
            return logical.value

        @derived(DesignPhysicalRelation)
        def relation_value() -> DesignPhysicalRelation:
            return cast(DesignPhysicalRelation, production_relation.relation)

        physical_ready = Readiness()
        logical_ready = Readiness()
        relation_ready = Readiness()
        physical = Projection(requirements, readiness=physical_ready)
        logical = Projection(logical_value, readiness=logical_ready)
        physical_relation = Projection(relation_value, readiness=relation_ready)

    class PlainSelectedMvau(MvauDataflowOp):
        implementation = Subspace(PlainComposite)

        @staticmethod
        def _network_answer(answer: Answer[NetworkResult]) -> Answer[object]:
            return Decided(answer.value.network) if isinstance(answer, Decided) else answer

        def selected_dataflow(self):
            assessment = self.implementation.logical
            return ProjectionAssessment(
                assessment.projection,
                assessment.readiness,
                assessment.constraints,
                self._network_answer(assessment.output),
                self._network_answer(assessment.accepted_answer),
            )

        def selected_implementation(self) -> object:
            return self.implementation

    operation = PlainSelectedMvau(model.graph.node[0]).bind(model, build, graph_context=context)
    implementation = operation.selected_implementation()
    assert isinstance(implementation, PlainComposite)
    local = capture_local_physical(implementation)
    relation = capture_local_relation(implementation, local)
    use = associate_physical_use(
        operation,
        implementation,
        local,
        relation,
        model=model,
        build=build,
        graph_context=context,
    )
    store = ArtifactStore(tmp_path / "plain-store")
    prepared = prepare_local_physical(
        local,
        roots={"finnlib": __import__("pathlib").Path("deps/finnlib").resolve()},
        template_roots=(
            __import__("pathlib").Path("src/finn/dataflow/designs/templates").resolve(),
        ),
        blobs=store,
    )
    built = materialize_local_physical(prepared, store=store)
    authorized = authorize_component_use(
        operation,
        implementation,
        use,
        built,
        model=model,
        build=build,
        graph_context=context,
        store=store,
    )
    first = install_compiler_physical_component(
        operation,
        implementation,
        authorized,
        outer_instance_id="plain_0",
        model=model,
        build=build,
        graph_context=context,
        store=store,
    )
    second = install_compiler_physical_component(
        operation,
        implementation,
        authorized,
        outer_instance_id="plain_1",
        model=model,
        build=build,
        graph_context=context,
        store=store,
    )
    assert first.outer_instance_id != second.outer_instance_id
    assert first.component == second.component


def test_parent_physical_hook_is_not_forced_to_build_unused_child_views() -> None:
    class Parent(Space):
        id = "fused_parent"
        version = "1"
        width = Problem(int)
        child = Subspace(UnavailableModule, width=width)

        @derived(ModuleBuildRequirements, width=width)
        def fused(*, width: int) -> ModuleBuildRequirements:
            return ModuleBuildRequirements(
                "fused_parent",
                "1",
                (("WIDTH", width),),
                ModuleABIRequirements(
                    FixedModuleName("fused_parent"), (), (("WIDTH", str(width)),), ()
                ),
                (),
                (),
            )

        physical_ready = Readiness(properties=(fused,))
        physical = Projection(fused, readiness=physical_ready)

    parent = Parent.start({Parent.width: 2}, namespace="parent_only")
    assert isinstance(parent.child.physical.accepted_answer, Absent)
    assert isinstance(parent.physical.accepted_answer, Decided)
