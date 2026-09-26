# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Regressions for assessed child capabilities and direct Kernel views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest

from parked.dataflow.kernels.test_composition_boundary import PlainChildComposite, PlainLogicalLeaf
from parked.dataflow.kernels.test_module_build_spec import ModuleKernel, _region
from parked.dataflow.ops.mvau.test_dot_product_kernel import _unconfigured
from parked.dataflow.ops.test_dataflow_op import Build, _replay_model
from finn.kernels._engine import Absent, Decided, Unresolved
from finn.kernels.artifacts.abi import ComponentABI
from finn.kernels.artifacts.build import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.parked.dataflow.model import (
    Kernel,
    KernelChoice,
    LogicalView,
    ModuleParameter,
    NetworkBoundary,
    PhysicalView,
    RegionDeclaration,
)
from finn.parked.dataflow.kernels.dotp_axi import DotpAxiKernel
from finn.dataflow.model.logical import (
    BoundaryContract,
    DataflowNetwork,
    NetworkNode,
    NetworkResult,
    RegionEndpoint,
    RegionResult,
)
from finn.parked.dataflow.ops.base import DataflowOpError
from finn.parked.dataflow.kernels.matmul.dot_product import DotProductKernel, WeightSupply
from finn.parked.dataflow.ops.physical import capture_local_physical
from finn.parked.dataflow.kernels.replay import ActivationReplayKernel
from finn.parked.dataflow.ops.replay.op import ReplaySpace
from parked.dataflow.ops.factory import make_space
from finn.parked.dataflow.ops.binding import ChoiceBinding
from finn.kernels.space import (
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
from finn.kernels.space.occurrence import layer_runtime
from finn.dataflow.model.logical.semantics import DATAFLOW_LOGICAL_RESULT_SEMANTICS


def _configured_dotp(activation: str) -> DotProductKernel:
    kernel = _unconfigured(
        matrix_width=4,
        matrix_height=4,
        activation=activation,
        weight="INT3",
        accumulator="INT16",
    )
    kernel = (
        kernel.assign(DotProductKernel.pe, 2)
        .assign(DotProductKernel.simd, 2)
        .assign(DotProductKernel.weight_supply, WeightSupply.EXTERNAL)
    )
    kernel = cast(DotProductKernel, kernel.compute.select("dotp_axi").root.kernel)
    child = kernel.child("compute")
    assert isinstance(child, Decided)
    return cast(
        DotProductKernel,
        cast(DotpAxiKernel, child.value).assign(DotpAxiKernel.compute_pumping, True).root.kernel,
    )


def test_composite_physical_preserves_real_child_acceptance() -> None:
    supported = _configured_dotp("INT3")
    assert isinstance(supported.physical.accepted_answer, Decided)
    assert capture_local_physical(supported).requirements.implementation_id == (
        "finn.mvau.decomposed.external"
    )

    rejected = _configured_dotp("INT1")
    child = rejected.child("compute")
    assert isinstance(child, Decided)
    child_answer = child.value.assess_view("physical").accepted_answer
    assert isinstance(child_answer, Absent)
    assert {finding.code for finding in child_answer.findings} == {"dotp-axi-operands-too-narrow"}
    parent_answer = rejected.physical.accepted_answer
    assert isinstance(parent_answer, Absent)
    assert {finding.code for finding in parent_answer.findings} == {"dotp-axi-operands-too-narrow"}
    with pytest.raises(DataflowOpError, match="not accepted"):
        capture_local_physical(rejected)


class GatedLogicalChild(PlainLogicalLeaf):
    id = "gated_logical_child"
    enabled = Input(bool)
    supported = Input(bool)

    @constraint(supported=supported)
    def supported_here(*, supported: bool) -> object:
        return True if supported else reject("logical-child-rejected", "unsupported")

    accepts = ConstraintGroup(supported_here)
    ready = Readiness(properties=(PlainLogicalLeaf.logical_result,))
    logical = Projection(
        PlainLogicalLeaf.logical_result,
        applicable_if=enabled,
        readiness=ready,
        constraints=accepts,
    )


class GatedLogicalParent(Kernel):
    id = "gated_logical_parent"
    width = Input(int)
    enabled = Input(bool)
    supported = Input(bool)
    child_node = KernelChoice(
        Subspace(
            GatedLogicalChild,
            width=width,
            enabled=enabled,
            supported=supported,
        )
    )
    source = NetworkBoundary(child_node.input("in"))
    result = NetworkBoundary(child_node.output("out"))


class GatedLogicalRoot(Space):
    width = Problem(int)
    enabled = Problem(bool)
    supported = Problem(bool)
    kernel = Subspace(
        GatedLogicalParent,
        width=width,
        enabled=enabled,
        supported=supported,
    )


class PendingLogicalChild(PlainLogicalLeaf):
    id = "pending_logical_child"
    required = Decision(int, values=(1, 2))
    ready = Readiness(
        decisions=(required,),
        properties=(PlainLogicalLeaf.logical_result,),
    )
    logical = Projection(PlainLogicalLeaf.logical_result, readiness=ready)


class PendingLogicalParent(Kernel):
    id = "pending_logical_parent"
    width = Input(int)
    child_node = KernelChoice(Subspace(PendingLogicalChild, width=width))
    source = NetworkBoundary(child_node.input("in"))
    result = NetworkBoundary(child_node.output("out"))


class PendingLogicalRoot(Space):
    width = Problem(int)
    kernel = Subspace(PendingLogicalParent, width=width)


def test_composite_logical_preserves_child_applicability_and_readiness() -> None:
    active = GatedLogicalRoot.start(
        {
            GatedLogicalRoot.width: 2,
            GatedLogicalRoot.enabled: True,
            GatedLogicalRoot.supported: True,
        }
    )
    assert isinstance(active.kernel.logical.accepted_answer, Decided)
    inactive = GatedLogicalRoot.start(
        {
            GatedLogicalRoot.width: 2,
            GatedLogicalRoot.enabled: False,
            GatedLogicalRoot.supported: True,
        }
    )
    child = inactive.kernel.child("child_node")
    assert isinstance(child, Decided)
    assert isinstance(child.value.assess_view("logical").accepted_answer, Absent)
    assert isinstance(inactive.kernel.logical.accepted_answer, Absent)

    rejected = GatedLogicalRoot.start(
        {
            GatedLogicalRoot.width: 2,
            GatedLogicalRoot.enabled: True,
            GatedLogicalRoot.supported: False,
        }
    )
    answer = rejected.kernel.logical.accepted_answer
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"logical-child-rejected"}

    pending = PendingLogicalRoot.start({PendingLogicalRoot.width: 2})
    child = pending.kernel.child("child_node")
    assert isinstance(child, Decided)
    assert isinstance(child.value.assess_view("logical").accepted_answer, Unresolved)
    assert isinstance(pending.kernel.logical.accepted_answer, Unresolved)
    resolved = child.value.assign(PendingLogicalChild.required, 1).root
    assert isinstance(resolved.kernel.logical.accepted_answer, Decided)  # type: ignore[attr-defined]


class ContractPhysicalLeaf(ModuleKernel):
    id = "contract_physical_leaf"
    enabled = Input(bool)
    supported = Input(bool)
    required = Decision(int, values=(1, 2))

    @constraint(supported=supported)
    def supported_here(*, supported: bool) -> object:
        return True if supported else reject("physical-contract-rejected", "unsupported")

    accepts = ConstraintGroup(supported_here)
    ready = Readiness(decisions=(required,), properties=(ModuleKernel.physical_result,))
    physical = PhysicalView(
        ModuleKernel.physical_result,
        applicable_if=enabled,
        readiness=ready,
        constraints=accepts,
    )


class ContractPhysicalParent(Kernel):
    id = "contract_physical_parent"
    width = Input(int)
    enabled = Input(bool)
    supported = Input(bool)
    child_node = KernelChoice(
        Subspace(
            ContractPhysicalLeaf,
            width=width,
            enabled=enabled,
            supported=supported,
        ),
        outputs=("logical_result", "physical_result", "physical_streams"),
    )
    result = NetworkBoundary(child_node.output("out"))

    @derived(ModuleBuildRequirements, child=child_node.physical_result)
    def physical_result(*, child: ModuleBuildRequirements) -> ModuleBuildRequirements:
        return child


class ContractPhysicalRoot(Space):
    width = Problem(int)
    enabled = Problem(bool)
    supported = Problem(bool)
    kernel = Subspace(
        ContractPhysicalParent,
        width=width,
        enabled=enabled,
        supported=supported,
    )


def _physical_contract_root(*, enabled: bool, supported: bool) -> ContractPhysicalRoot:
    return ContractPhysicalRoot.start(
        {
            ContractPhysicalRoot.width: 2,
            ContractPhysicalRoot.enabled: enabled,
            ContractPhysicalRoot.supported: supported,
        }
    )


def test_composite_physical_preserves_child_applicability_readiness_and_acceptance() -> None:
    inactive = _physical_contract_root(enabled=False, supported=True)
    assert isinstance(inactive.kernel.physical.accepted_answer, Absent)

    pending = _physical_contract_root(enabled=True, supported=True)
    assert isinstance(pending.kernel.physical.accepted_answer, Unresolved)
    child = pending.kernel.child("child_node")
    assert isinstance(child, Decided)
    resolved = child.value.assign(ContractPhysicalLeaf.required, 1).root
    assert isinstance(resolved.kernel.physical.accepted_answer, Decided)  # type: ignore[attr-defined]

    rejected = _physical_contract_root(enabled=True, supported=False)
    child = rejected.kernel.child("child_node")
    assert isinstance(child, Decided)
    rejected = cast(
        ContractPhysicalRoot,
        child.value.assign(ContractPhysicalLeaf.required, 1).root,
    )
    answer = rejected.kernel.physical.accepted_answer
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"physical-contract-rejected"}


def test_consumed_stream_capability_keeps_its_pre_kp_dependency_path() -> None:
    root = _physical_contract_root(enabled=True, supported=True)
    paths = {item.path.value for item in layer_runtime(root.kernel).compiled.spec.properties}
    assert (
        "semantic.root.kernel.child_node.contract_physical_leaf.accepted-physical-streams" in paths
    )
    assert not any(path.endswith("accepted-physical_streams") for path in paths)


class GuardedLeaf(ModuleKernel):
    id = "guarded_leaf"
    enabled = Input(bool)
    guarded_ready = Readiness(properties=(ModuleKernel.physical_result,))
    physical = PhysicalView(
        ModuleKernel.physical_result,
        applicable_if=enabled,
        readiness=guarded_ready,
    )


class InheritedGuardedLeaf(GuardedLeaf):
    id = "inherited_guarded_leaf"


class GuardedComposite(Kernel):
    id = "guarded_composite"
    width = Input(int)
    enabled = Input(bool)
    child_node = KernelChoice(
        Subspace(ModuleKernel, width=width),
        outputs=("logical_result", "physical_result", "physical_streams"),
    )
    result = NetworkBoundary(child_node.output("out"))

    @derived(ModuleBuildRequirements, child=child_node.physical_result)
    def guarded_module(*, child: ModuleBuildRequirements) -> ModuleBuildRequirements:
        return child

    guarded_ready = Readiness(properties=(guarded_module,))
    physical = PhysicalView(guarded_module, applicable_if=enabled, readiness=guarded_ready)


class InheritedGuardedComposite(GuardedComposite):
    id = "inherited_guarded_composite"


class GuardedLogicalLeaf(ModuleKernel):
    id = "guarded_logical_leaf"
    enabled = Input(bool)
    supported = Input(bool)

    @constraint(supported=supported)
    def logical_supported(*, supported: bool) -> object:
        return True if supported else reject("logical-contract-rejected", "unsupported")

    logical_acceptance = ConstraintGroup(logical_supported)
    guarded_logical_ready = Readiness(properties=(ModuleKernel.logical_result,))
    logical = LogicalView(
        ModuleKernel.logical_result,
        applicable_if=enabled,
        readiness=guarded_logical_ready,
        constraints=logical_acceptance,
    )


class InheritedGuardedLogicalLeaf(GuardedLogicalLeaf):
    id = "inherited_guarded_logical_leaf"


class GuardedLogicalComposite(PlainChildComposite):
    id = "guarded_logical_composite"
    enabled = Input(bool)
    supported = Input(bool)

    @constraint(supported=supported)
    def logical_supported(*, supported: bool) -> object:
        return True if supported else reject("logical-contract-rejected", "unsupported")

    logical_acceptance = ConstraintGroup(logical_supported)
    guarded_logical_ready = Readiness(properties=(PlainChildComposite.logical_result,))
    logical = LogicalView(
        PlainChildComposite.logical_result,
        applicable_if=enabled,
        readiness=guarded_logical_ready,
        constraints=logical_acceptance,
    )


class InheritedGuardedLogicalComposite(GuardedLogicalComposite):
    id = "inherited_guarded_logical_composite"


@pytest.mark.parametrize("kernel_type", (GuardedLeaf, InheritedGuardedLeaf))
def test_leaf_synthesis_preserves_direct_and_inherited_physical_views(
    kernel_type: type[GuardedLeaf],
) -> None:
    class Root(Space):
        width = Problem(int)
        enabled = Problem(bool)
        kernel = Subspace(kernel_type, width=width, enabled=enabled)

    assert kernel_type.physical.applicable_if is GuardedLeaf.enabled
    assert isinstance(
        Root.start({Root.width: 2, Root.enabled: False}).kernel.physical.accepted_answer,
        Absent,
    )


@pytest.mark.parametrize("kernel_type", (GuardedComposite, InheritedGuardedComposite))
def test_composite_synthesis_preserves_direct_and_inherited_physical_views(
    kernel_type: type[GuardedComposite],
) -> None:
    class Root(Space):
        width = Problem(int)
        enabled = Problem(bool)
        kernel = Subspace(kernel_type, width=width, enabled=enabled)

    assert kernel_type.physical.applicable_if is GuardedComposite.enabled
    assert isinstance(
        Root.start({Root.width: 2, Root.enabled: False}).kernel.physical.accepted_answer,
        Absent,
    )


@pytest.mark.parametrize("kernel_type", (GuardedLogicalLeaf, InheritedGuardedLogicalLeaf))
def test_leaf_synthesis_preserves_complete_direct_and_inherited_logical_views(
    kernel_type: type[GuardedLogicalLeaf],
) -> None:
    class Root(Space):
        width = Problem(int)
        enabled = Decision(bool, values=(False, True))
        supported = Problem(bool)
        kernel = Subspace(
            kernel_type,
            width=width,
            enabled=enabled,
            supported=supported,
        )

    root = Root.start({Root.width: 2, Root.supported: True})
    assert isinstance(root.kernel.logical.accepted_answer, Unresolved)
    assert isinstance(root.assign(Root.enabled, False).kernel.logical.accepted_answer, Absent)
    rejected = Root.start({Root.width: 2, Root.supported: False}).assign(Root.enabled, True)
    answer = rejected.kernel.logical.accepted_answer
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"logical-contract-rejected"}


@pytest.mark.parametrize("kernel_type", (GuardedLogicalComposite, InheritedGuardedLogicalComposite))
def test_composite_synthesis_preserves_complete_direct_and_inherited_logical_views(
    kernel_type: type[GuardedLogicalComposite],
) -> None:
    class Root(Space):
        width = Problem(int)
        enabled = Decision(bool, values=(False, True))
        supported = Problem(bool)
        kernel = Subspace(
            kernel_type,
            width=width,
            enabled=enabled,
            supported=supported,
        )

    root = Root.start({Root.width: 2, Root.supported: True})
    assert isinstance(root.kernel.logical.accepted_answer, Unresolved)
    assert isinstance(root.assign(Root.enabled, False).kernel.logical.accepted_answer, Absent)
    rejected = Root.start({Root.width: 2, Root.supported: False}).assign(Root.enabled, True)
    answer = rejected.kernel.logical.accepted_answer
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"logical-contract-rejected"}


class AuthoredLogical(Kernel):
    id = "authored_logical"
    valid = Input(bool)

    @derived(DATAFLOW_LOGICAL_RESULT_SEMANTICS, valid=valid)
    def logical_result(*, valid: bool) -> RegionResult:
        region = _region(2)
        return RegionResult(region if valid else replace(region, outputs=region.outputs * 2))

    ready = Readiness(properties=(logical_result,))
    logical = LogicalView(logical_result, readiness=ready)


class AuthoredNetwork(Kernel):
    id = "authored_network"
    valid = Input(bool)

    @derived(DATAFLOW_LOGICAL_RESULT_SEMANTICS, valid=valid)
    def logical_result(*, valid: bool) -> NetworkResult:
        region = _region(2)
        node = NetworkNode("node", region)
        boundary = BoundaryContract(
            "out",
            RegionEndpoint("node", "out"),
            region.outputs[0].port.beat_sequence,
        )
        return NetworkResult(DataflowNetwork((node,) if valid else (node, node), (), (boundary,)))

    ready = Readiness(properties=(logical_result,))
    logical = LogicalView(logical_result, readiness=ready)


@pytest.mark.parametrize("kernel_type", (AuthoredLogical, AuthoredNetwork))
def test_direct_logical_view_applies_canonical_domain_validation(
    kernel_type: type[Kernel],
) -> None:
    class Root(Space):
        valid = Problem(bool)
        kernel = Subspace(kernel_type, valid=valid)

    assert isinstance(Root.start({Root.valid: True}).kernel.logical.accepted_answer, Decided)
    answer = Root.start({Root.valid: False}).kernel.logical.accepted_answer
    assert isinstance(answer, Absent)
    assert any("duplicate" in finding.code for finding in answer.findings)


class IndependentPhysical(Kernel):
    id = "independent_physical"
    semantic_extent = Input(int)
    region = RegionDeclaration(
        family="logical",
        version="1",
        construct=_region,
        width=semantic_extent,
    )
    WIDTH = ModuleParameter.constant(8, why="fixed physical interface")

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        return ComponentABI("fixed", (), (("WIDTH", str(parameters["WIDTH"])),))


class DependentPhysical(IndependentPhysical):
    id = "dependent_physical"
    WIDTH = ModuleParameter(IndependentPhysical.semantic_extent)


@pytest.mark.parametrize(
    ("kernel_type", "before_kind", "dependencies"),
    (
        (IndependentPhysical, Decided, ()),
        (DependentPhysical, Unresolved, ("parameter_WIDTH",)),
    ),
)
def test_leaf_physical_requirements_track_only_consumed_dependencies(
    kernel_type: type[Kernel],
    before_kind: type[object],
    dependencies: tuple[str, ...],
) -> None:
    class Root(Space):
        semantic_extent = Decision(int, values=(2, 4))
        kernel = Subspace(kernel_type, semantic_extent=semantic_extent)

    root = Root.start({})
    assert isinstance(root.kernel.physical.accepted_answer, before_kind)
    assert tuple(name for name, _source in kernel_type.physical_result.dependencies) == dependencies
    assert isinstance(
        root.assign(Root.semantic_extent, 2).kernel.physical.accepted_answer,
        Decided,
    )


class SharedValueChild(PlainLogicalLeaf):
    id = "shared_value_child"
    enabled = Input(bool)
    logical_ready = Readiness(properties=(PlainLogicalLeaf.logical_result,))
    logical = Projection(
        PlainLogicalLeaf.logical_result,
        applicable_if=enabled,
        readiness=logical_ready,
    )

    @derived(ModuleBuildRequirements, logical=PlainLogicalLeaf.logical_result)
    def physical_result(*, logical: RegionResult) -> ModuleBuildRequirements:
        width = logical.region.outputs[0].port.beat_sequence.elements_per_beat
        return ModuleBuildRequirements(
            "shared-fixed",
            "1",
            (("WIDTH", width),),
            ModuleABIRequirements(FixedModuleName("shared_fixed"), (), (("WIDTH", str(width)),)),
            (),
            (),
        )

    physical_ready = Readiness(properties=(physical_result,))
    physical = Projection(physical_result, readiness=physical_ready)
    exports = (PlainLogicalLeaf.logical_result, physical_result)


class SharedValueParent(Kernel):
    id = "shared_value_parent"
    width = Input(int)
    enabled = Input(bool)
    child_node = KernelChoice(
        Subspace(SharedValueChild, width=width, enabled=enabled),
        outputs=("logical_result", "physical_result"),
    )
    source = NetworkBoundary(child_node.input("in"))
    result = NetworkBoundary(child_node.output("out"))

    @derived(ModuleBuildRequirements, child=child_node.physical_result)
    def physical_result(*, child: ModuleBuildRequirements) -> ModuleBuildRequirements:
        return child


def test_nesting_does_not_attach_logical_applicability_to_raw_physical_values() -> None:
    class DirectRoot(Space):
        width = Problem(int)
        enabled = Problem(bool)
        child = Subspace(SharedValueChild, width=width, enabled=enabled)

    class NestedRoot(Space):
        width = Problem(int)
        enabled = Problem(bool)
        kernel = Subspace(SharedValueParent, width=width, enabled=enabled)

    for enabled in (True, False):
        direct = DirectRoot.start({DirectRoot.width: 2, DirectRoot.enabled: enabled})
        nested = NestedRoot.start({NestedRoot.width: 2, NestedRoot.enabled: enabled})
        assert isinstance(direct.child.physical.accepted_answer, Decided)
        child = nested.kernel.child("child_node")
        assert isinstance(child, Decided)
        assert isinstance(child.value.assess_view("physical").accepted_answer, Decided)
        assert isinstance(nested.kernel.physical.accepted_answer, Decided)


class OutputConditionalChild(PlainLogicalLeaf):
    id = "output_conditional_child"

    @derived(bool, value=PlainLogicalLeaf.logical_result)
    def enabled(*, value: RegionResult) -> bool:
        return bool(value.region.outputs)

    ready = Readiness(properties=(PlainLogicalLeaf.logical_result,))
    logical = Projection(
        PlainLogicalLeaf.logical_result,
        applicable_if=enabled,
        readiness=ready,
    )


class OutputConditionalParent(Kernel):
    id = "output_conditional_parent"
    width = Input(int)
    child_node = KernelChoice(Subspace(OutputConditionalChild, width=width))
    source = NetworkBoundary(child_node.input("in"))
    result = NetworkBoundary(child_node.output("out"))


def test_output_dependent_applicability_remains_acyclic_when_nested() -> None:
    class Root(Space):
        width = Problem(int)
        kernel = Subspace(OutputConditionalParent, width=width)

    root = Root.start({Root.width: 2})
    assert isinstance(root.kernel.logical.accepted_answer, Decided)


class AlternatePhysicalOutput(ModuleKernel):
    id = "alternate_physical_output"
    enabled = Input(bool)

    @derived(ModuleBuildRequirements, original=ModuleKernel.physical_result)
    def actual_output(*, original: ModuleBuildRequirements) -> ModuleBuildRequirements:
        return replace(original, implementation_id="actual-view-value")

    ready = Readiness(properties=(actual_output,))
    physical = PhysicalView(actual_output, applicable_if=enabled, readiness=ready)


class AlternatePhysicalParent(Kernel):
    id = "alternate_physical_parent"
    width = Input(int)
    enabled = Input(bool)
    child_node = KernelChoice(
        Subspace(AlternatePhysicalOutput, width=width, enabled=enabled),
        outputs=("logical_result", "physical_result"),
    )
    result = NetworkBoundary(child_node.output("out"))

    @derived(ModuleBuildRequirements, child=child_node.physical_result)
    def physical_result(*, child: ModuleBuildRequirements) -> ModuleBuildRequirements:
        return child


def test_kernel_choice_forwards_the_authored_physical_view_output() -> None:
    class Root(Space):
        width = Problem(int)
        enabled = Problem(bool)
        kernel = Subspace(AlternatePhysicalParent, width=width, enabled=enabled)

    enabled = Root.start({Root.width: 2, Root.enabled: True})
    answer = enabled.kernel.physical.accepted_answer
    assert isinstance(answer, Decided)
    assert answer.value.implementation_id == "actual-view-value"

    disabled = Root.start({Root.width: 2, Root.enabled: False})
    assert isinstance(disabled.kernel.physical.accepted_answer, Absent)


class GuardedReplay(ActivationReplayKernel):
    @derived(bool)
    def available() -> bool:
        return False

    logical = LogicalView(
        ActivationReplayKernel.logical_result,
        applicable_if=available,
        readiness=ActivationReplayKernel.logical_ready,
        constraints=ActivationReplayKernel.logical_accepts,
    )


class PendingReplay(ActivationReplayKernel):
    permission = Decision(bool, values=(False, True))
    pending_ready = Readiness(
        decisions=(permission,),
        properties=(ActivationReplayKernel.logical_result,),
    )
    logical = LogicalView(
        ActivationReplayKernel.logical_result,
        readiness=pending_ready,
        constraints=ActivationReplayKernel.logical_accepts,
    )


class RejectedReplay(ActivationReplayKernel):
    @constraint()
    def rejected() -> object:
        return reject("guarded-replay-rejected", "logical view rejected")

    rejected_accepts = ConstraintGroup(rejected)
    logical = LogicalView(
        ActivationReplayKernel.logical_result,
        readiness=ActivationReplayKernel.logical_ready,
        constraints=(ActivationReplayKernel.logical_accepts, rejected_accepts),
    )


class GuardedReplayOp(ReplaySpace):
    kernel = Subspace(
        GuardedReplay,
        repetitions=ReplaySpace.repetitions,
        matrix_width=ReplaySpace.matrix_width,
        matrix_height=ReplaySpace.matrix_height,
        activation_type=ReplaySpace.activation.datatype,
    )


class PendingReplayOp(ReplaySpace):
    choice_bindings = (
        *ReplaySpace.choice_bindings,
        ChoiceBinding("kernel_permission", ("kernel",), "permission"),
    )
    kernel = Subspace(
        PendingReplay,
        repetitions=ReplaySpace.repetitions,
        matrix_width=ReplaySpace.matrix_width,
        matrix_height=ReplaySpace.matrix_height,
        activation_type=ReplaySpace.activation.datatype,
    )


class RejectedReplayOp(ReplaySpace):
    kernel = Subspace(
        RejectedReplay,
        repetitions=ReplaySpace.repetitions,
        matrix_width=ReplaySpace.matrix_width,
        matrix_height=ReplaySpace.matrix_height,
        activation_type=ReplaySpace.activation.datatype,
    )


def _configured_replay_type(operation_type: type[ReplaySpace]) -> ReplaySpace:
    model = _replay_model()
    operation = make_space(model, space_type=operation_type, build=Build())
    return cast(
        ReplaySpace,
        operation.kernel.assign(ActivationReplayKernel.simd, 2).root,
    )


@pytest.mark.parametrize(
    ("operation_type", "answer_type"),
    (
        (GuardedReplayOp, Absent),
        (PendingReplayOp, Unresolved),
        (RejectedReplayOp, Absent),
    ),
)
def test_node_logical_mapping_requires_authored_logical_acceptance(
    operation_type: type[ReplaySpace],
    answer_type: type[object],
) -> None:
    operation = _configured_replay_type(operation_type)
    kernel = operation.require_implementation()
    assert isinstance(kernel, Kernel)
    assert isinstance(kernel.logical.accepted_answer, answer_type)
    assert isinstance(kernel.dataflow.accepted_answer, answer_type)
    assert isinstance(operation.dataflow.accepted_answer, answer_type)
    assert isinstance(operation.operand_mapping, answer_type)
