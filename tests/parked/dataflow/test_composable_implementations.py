# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from parked.dataflow.kernels.test_module_build_spec import UnavailableModule
from parked.dataflow.ops.mvau.test_dot_product_kernel import KERNEL_INPUTS, Problem_, _occurrence
from parked.dataflow.physical_fixture import configure, source_model
from finn.kernels._engine import Absent, Decided, Unresolved
from finn.kernels.artifacts.build import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.artifacts.store import ArtifactStore
from finn.parked.dataflow.model import Kernel
from finn.parked.dataflow.model.physical.interface import PhysicalResult
from finn.parked.dataflow.ops.binding import ImplementationBinding
from finn.parked.dataflow.kernels.matmul.base import MatmulInterface
from finn.dataflow.model.logical.composition import NetworkResult, RegionResult
from finn.parked.dataflow.ops.base import DataflowOpError
from finn.parked.dataflow.kernels.matmul.base import WeightedDotProductKernel
from finn.parked.dataflow.kernels.matmul.dot_product import DotProductKernel, WeightSupply
from finn.kernels.resources import template_root
from finn.parked.dataflow.kernels.matmul.base import (
    AccumulationMode,
    ActivationMode,
    MvauComputationProfile,
)
from finn.parked.dataflow.ops.mvau.op import MvauSpace
from finn.parked.dataflow.ops.reconstruction import build_space
from finn.parked.dataflow.kernels.dotp_axi import DotpAxiKernel
from finn.parked.dataflow.kernels.matmul.base import DspBlock
from finn.parked.dataflow.model import PhysicalView
from finn.parked.dataflow.ops.physical import (
    associate_physical_use,
    authorize_component_use,
    capture_local_physical,
    install_compiler_physical_component,
    materialize_local_physical,
    prepare_local_physical,
    validate_compiler_physical_use,
)
from finn.parked.dataflow.model.identity import implementation_identity
from finn.kernels.space.declarations import (
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
    replay = design.child("replay")
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
    physical = PhysicalView(value, readiness=ready)
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


class _IndependentLogicalKernel(DotProductKernel):
    id = "independent_logical_dot_product"
    logical_mode = Decision(str, values=("accept", "reject"))

    @constraint(mode=logical_mode)
    def logical_mode_supported(*, mode: str) -> object:
        return True if mode == "accept" else reject("logical-mode-rejected", "logical-only")

    logical_support = ConstraintGroup(
        *DotProductKernel.logical_support.constraints,
        logical_mode_supported,
        name="logical_support",
    )


class _IndependentRoot(Problem_):
    kernel = Subspace(
        _IndependentLogicalKernel,
        name="dot_product",
        **{name: cast("object", getattr(Problem_, name)) for name in KERNEL_INPUTS},
    )


def _independent_design() -> _IndependentLogicalKernel:
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
    design = cast(_IndependentLogicalKernel, root.kernel)
    design = cast(
        _IndependentLogicalKernel,
        design.assign(WeightedDotProductKernel.pe, 2)
        .assign(WeightedDotProductKernel.simd, 2)
        .assign(DotProductKernel.weight_supply, WeightSupply.EXTERNAL),
    )
    design = cast(_IndependentLogicalKernel, design.compute.select("dotp_axi").root.kernel)
    compute = design.child("compute")
    assert isinstance(compute, Decided)
    design = cast(
        _IndependentLogicalKernel,
        compute.value.assign(DotpAxiKernel.compute_pumping, False).root.kernel,
    )
    return design


def test_local_physical_capture_ignores_unresolved_and_rejected_logical_only_choice(
    tmp_path,
) -> None:
    design = _independent_design()
    unresolved_capture = capture_local_physical(design)
    assert isinstance(design.dataflow.accepted_answer, Unresolved)
    assert "physical_relation" not in type(design).capability_names()

    rejected = cast(
        _IndependentLogicalKernel,
        design.assign(_IndependentLogicalKernel.logical_mode, "reject"),
    )
    rejected_capture = capture_local_physical(rejected)
    assert rejected_capture.point_fingerprint == unresolved_capture.point_fingerprint
    assert isinstance(rejected.dataflow.accepted_answer, Absent)
    assert "physical_relation" not in type(design).capability_names()

    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_local_physical(
        unresolved_capture,
        roots={"finnlib": __import__("pathlib").Path("deps/finnlib").resolve()},
        template_roots=(template_root(),),
        blobs=store,
    )
    built = materialize_local_physical(prepared, store=store)
    assert built.requirements_fingerprint == unresolved_capture.physical_fingerprint

    accepted = cast(
        _IndependentLogicalKernel,
        design.assign(_IndependentLogicalKernel.logical_mode, "accept"),
    )
    assert capture_local_physical(accepted).requirements == unresolved_capture.requirements


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
    left = configure(
        model.get_customop_wrapper(model.graph.node[0])
        .set_context(build, graph_context=context)
        .space
    )
    right = configure(
        model.get_customop_wrapper(model.graph.node[1])
        .set_context(build, graph_context=context)
        .space
    )
    left_design = cast(Kernel, left.selected_kernel())
    right_design = cast(Kernel, right.selected_kernel())
    left_local = capture_local_physical(left_design)
    right_local = capture_local_physical(right_design)
    assert left_local != right_local
    assert left_local.physical_fingerprint == right_local.physical_fingerprint
    use = associate_physical_use(
        left,
        left_local,
        model=model,
        build=build,
        graph_context=context,
    )
    assert (
        validate_compiler_physical_use(
            left,
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
        use,
        model=model,
        build=build,
        graph_context=context,
    )


def test_plain_space_completes_source_selection_build_association_and_installation(
    tmp_path,
) -> None:
    model, build, context = source_model()
    production = configure(model.get_customop_wrapper(model.graph.node[0]).set_context(build).space)
    production_local = capture_local_physical(production.require_implementation())

    class PlainComposite(MatmulInterface):
        id = "plain_composite"
        version = "1"

        @derived(PhysicalResult)
        def requirements() -> PhysicalResult:
            return cast(PhysicalResult, production_local.physical)

        physical_ready = Readiness()
        physical = Projection(requirements, readiness=physical_ready)

    class PlainPhysicalSpace(MvauSpace):
        implementation_binding = ImplementationBinding(("implementation",))
        implementation = Subspace(
            PlainComposite,
            repetitions=MvauSpace.repetitions,
            matrix_width=MvauSpace.matrix_width,
            matrix_height=MvauSpace.matrix_height,
            activation_type=MvauSpace.activation.datatype,
            weight_type=MvauSpace.weight.datatype,
            accumulator_type=MvauSpace.accumulator_type,
            output_type=MvauSpace.output_type,
            computation_profile=MvauSpace.profile,
            integer_bounds=MvauSpace.integer_bounds,
        )

    operation = build_space(PlainPhysicalSpace, model, model.graph.node[0], build=build)
    node_use = operation
    implementation = node_use.require_implementation()
    assert isinstance(implementation, PlainComposite)
    local = capture_local_physical(implementation)
    use = associate_physical_use(node_use, local, model=model, build=build)
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        associate_physical_use(node_use, production_local, model=model, build=build)
    store = ArtifactStore(tmp_path / "plain-store")
    prepared = prepare_local_physical(
        local,
        roots={"finnlib": __import__("pathlib").Path("deps/finnlib").resolve()},
        template_roots=(template_root(),),
        blobs=store,
    )
    built = materialize_local_physical(prepared, store=store)
    authorized = authorize_component_use(
        node_use, use, built, model=model, build=build, store=store
    )
    first = install_compiler_physical_component(
        node_use,
        authorized,
        outer_instance_id="plain_0",
        model=model,
        build=build,
        store=store,
    )
    second = install_compiler_physical_component(
        node_use,
        authorized,
        outer_instance_id="plain_1",
        model=model,
        build=build,
        store=store,
    )
    assert first.outer_instance_id != second.outer_instance_id
    assert first.component == second.component


def test_built_component_cannot_cross_physical_choice_points(tmp_path) -> None:
    model, build, context = source_model()
    bound = (
        model.get_customop_wrapper(model.graph.node[0])
        .set_context(build, graph_context=context)
        .space
    )
    pumped = configure(bound, pumping=True)
    unpumped = configure(bound.reconstruct(), pumping=False)
    pumped_design = cast(Kernel, pumped.selected_kernel())
    unpumped_design = cast(Kernel, unpumped.selected_kernel())
    pumped_local = capture_local_physical(pumped_design)
    unpumped_local = capture_local_physical(unpumped_design)
    assert pumped_local.physical_fingerprint != unpumped_local.physical_fingerprint

    store = ArtifactStore(tmp_path / "cross-store")
    prepared = prepare_local_physical(
        pumped_local,
        roots={"finnlib": __import__("pathlib").Path("deps/finnlib").resolve()},
        template_roots=(template_root(),),
        blobs=store,
    )
    built = materialize_local_physical(prepared, store=store)
    use = associate_physical_use(
        unpumped,
        unpumped_local,
        model=model,
        build=build,
        graph_context=context,
    )
    with pytest.raises(DataflowOpError, match="different local capture|requirements differ"):
        authorize_component_use(
            unpumped,
            use,
            built,
            model=model,
            build=build,
            graph_context=context,
            store=store,
        )


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
