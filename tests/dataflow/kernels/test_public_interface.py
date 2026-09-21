# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public facets share facts with checked body exports, without choosing a Kernel."""

from dataclasses import replace

import pytest
from qonnx.core.datatype import DataType

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.analysis.integer_dot import DotProductBounds, IntegerRange, IntegerSupportReport
from finn.dataflow.kernels.matmul.base import (
    AccumulationMode,
    ActivationMode,
    DspBlock,
    MatmulInterface,
    MvauComputationProfile,
)
from finn.dataflow.kernels.matmul.dot_product import DotProductKernel, KERNEL_INPUTS, WeightSupply
from finn.dataflow.kernels.replay import ActivationReplayKernel
from finn.dataflow.model.children import KernelChoice
from finn.dataflow.model.kernel import Kernel
from finn.dataflow.model.logical.authoring import NetworkBoundary
from finn.dataflow.model.logical.composition import RegionResult, logical_network
from finn.dataflow.model.logical.interface import (
    InterfaceError,
    OperandExport,
    OperandTarget,
    PublicOperand,
    validate_operand_export,
)
from finn.dataflow.model.logical.interface_authoring import (
    PublicOperandDeclaration,
    operand_domain,
    operand_export,
    operand_type,
)
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.dataflow.model.logical.network import PositionMap, RegionEndpoint
from finn.dataflow.model.logical.refs import RegionInputRef, RegionOutputRef
from finn.dataflow.model.logical.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
    QONNX_DATATYPE_CODEC,
)
from finn.dataflow.space.declarations import (
    Input,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    declared_members,
    derived,
)
from dataflow.kernels.test_kernel import ToyKernel, _region
from dataflow.kernels.test_composition_boundary import NestedCompositeKernel, PortedModule, _design
from dataflow.kernels.test_module_build_spec import UnavailableModule, _region as required_region


class MatrixFacts(Space):
    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    weight_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    accumulator_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    output_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    computation_profile = Problem(MvauComputationProfile)
    integer_bounds = Problem(DotProductBounds, required=False)
    narrow_weights = Problem(bool, required=False)
    target_dsp = Problem(DspBlock, required=False)
    clock_period_ns = Problem(float, required=False)
    numerical_support = Problem(IntegerSupportReport, required=False)
    initializer_present = Problem(bool)


class MatrixHarness(MatrixFacts):
    family = Subspace(
        MatmulInterface,
        **{
            name: getattr(MatrixFacts, name)
            for name, value in declared_members(MatmulInterface)
            if isinstance(value, Input)
        },
    )
    kernel = Subspace(
        DotProductKernel, **{name: getattr(MatrixFacts, name) for name in KERNEL_INPUTS}
    )


def matrix_point(accumulator="INT32", output=None, bounds=None):
    facts = {
        MatrixHarness.repetitions: 2,
        MatrixHarness.matrix_width: 8,
        MatrixHarness.matrix_height: 4,
        MatrixHarness.activation_type: DataType["INT8"],
        MatrixHarness.weight_type: DataType["INT8"],
        MatrixHarness.accumulator_type: DataType[accumulator],
        MatrixHarness.output_type: DataType[output or accumulator],
        MatrixHarness.computation_profile: MvauComputationProfile(
            AccumulationMode.INTEGER, ActivationMode.NONE
        ),
        MatrixHarness.initializer_present: True,
    }
    if bounds is not None:
        facts[MatrixHarness.integer_bounds] = bounds
    return MatrixHarness.start(facts)


def test_common_family_type_and_domains_without_folding_target_or_candidate():
    root = matrix_point()
    for occurrence in (root.family, root.kernel):
        assert operand_type(occurrence, "result") == Decided(DataType["INT32"])
        assert operand_domain(occurrence, "weights") == Decided(RectangularDomain((8, 4)))
    assert isinstance(root.kernel.assess_view("logical").accepted_answer, Unresolved)
    assert not isinstance(root.kernel.assess_view("physical").accepted_answer, Decided)


def test_known_result_type_does_not_bypass_output_requirement_or_precision():
    root = matrix_point(output="INT16")
    assert root.family.answer(MatmulInterface.result_type) == Decided(DataType["INT32"])
    assert isinstance(operand_type(root.family, "result"), Absent)
    assert isinstance(operand_type(matrix_point(accumulator="INT8").family, "result"), Absent)


def test_authenticated_bounds_refine_precision_and_changed_values_invalidate_it():
    def bounds(high):
        interval = IntegerRange(-high, high)
        return DotProductBounds(interval, interval, interval, (interval,), 8)

    assert operand_type(
        matrix_point(accumulator="INT8", bounds=bounds(100)).family, "result"
    ) == Decided(DataType["INT8"])
    assert isinstance(
        operand_type(matrix_point(accumulator="INT8", bounds=bounds(200)).family, "result"), Absent
    )


@pytest.mark.parametrize(
    "supply", [WeightSupply.EXTERNAL, WeightSupply.EMBEDDED, WeightSupply.DECOUPLED]
)
def test_matrix_export_owns_transpose_and_unported_weights(supply):
    kernel = matrix_point().kernel.assign(DotProductKernel.pe, 2).assign(DotProductKernel.simd, 2)
    kernel = kernel.assign(DotProductKernel.weight_supply, supply)
    candidate = "dotp_axi_embedded" if supply is WeightSupply.EMBEDDED else "dotp_axi"
    kernel = kernel.compute.select(candidate).root.kernel
    assert isinstance(kernel.logical.accepted_answer, Decided)
    if supply is WeightSupply.EMBEDDED:
        assert isinstance(kernel.physical.accepted_answer, Absent)
    else:
        assert isinstance(kernel.physical.accepted_answer, Unresolved)
    answer = operand_export(kernel, "weights")
    assert isinstance(answer, Decided), answer
    export = answer.value
    assert export.operand.domain == RectangularDomain((8, 4))
    assert all(target.position_map.mapped((3, 2)) == (2, 3) for target in export.targets)
    if supply is WeightSupply.EXTERNAL:
        network = logical_network(kernel.assess_view("logical").accepted_answer.value)
        assert export.public_beat(network, 0)
    else:
        assert not any(target.presentations for target in export.targets)
        if supply is WeightSupply.DECOUPLED:
            assert tuple(target.ref for target in export.targets) == (
                RegionInputRef("memory", "W"),
            )
        with pytest.raises(InterfaceError, match="presentation"):
            export.presentation()


def _copy_export(public, logical):
    return OperandExport(
        public,
        (
            OperandTarget(
                RegionOutputRef("root", "value"),
                PositionMap.row_major_reshape(
                    public.domain, logical.region.outputs[0].port.operand.position_domain
                ),
                (RegionEndpoint("root", "output"),),
            ),
        ),
    )


class PublicCopy(ToyKernel):
    @derived(QONNX_DATATYPE_VALUE_SEMANTICS)
    def element_type():
        return DataType["INT8"]

    @derived(RectangularDomain, extent=ToyKernel.extent)
    def result_domain(*, extent):
        return RectangularDomain((extent,))

    type_ready = Readiness(properties=(element_type,))
    domain_ready = Readiness(properties=(result_domain,))
    result_type = Projection(element_type, readiness=type_ready)
    domain = Projection(result_domain, readiness=domain_ready)
    public_operands = (
        PublicOperandDeclaration("result", "output", result_type, domain, _copy_export),
    )


def copy_point(kernel_type=PublicCopy):
    class Root(Space):
        extent = Problem(int)
        lanes = Problem(int)
        copy = Subspace(kernel_type, extent=extent, lanes=lanes)

    return Root.start({Root.extent: 8, Root.lanes: 2}).copy


def test_region_normalization_preserves_actual_endpoint_and_order():
    kernel = copy_point()
    export = operand_export(kernel, "result")
    assert isinstance(export, Decided), export
    network = logical_network(kernel.logical.accepted_answer.value)
    assert network.boundaries[1].endpoint == RegionEndpoint("root", "output")
    assert export.value.public_beat(network, 1) == ((2,), (3,))


def test_full_logical_and_dataflow_acceptance_require_export_body_agreement():
    class WrongCopy(PublicCopy):
        @derived(QONNX_DATATYPE_VALUE_SEMANTICS)
        def wrong_type():
            return DataType["INT4"]

        wrong_ready = Readiness(properties=(wrong_type,))
        wrong = Projection(wrong_type, readiness=wrong_ready)
        public_operands = (
            PublicOperandDeclaration("result", "output", wrong, PublicCopy.domain, _copy_export),
        )

    kernel = copy_point(WrongCopy)
    assert operand_type(kernel, "result") == Decided(DataType["INT4"])
    assert isinstance(kernel.logical.accepted_answer, Absent)
    assert isinstance(kernel.dataflow.accepted_answer, Absent)
    assert isinstance(operand_export(kernel, "result"), Absent)


def _boundary_copy_export(public, logical):
    network = logical_network(logical)
    endpoint = next(boundary.endpoint for boundary in network.boundaries if boundary.id == "result")
    operand = network.node(endpoint.node_id).region.output_interface(endpoint.port_id).port.operand
    return OperandExport(
        public,
        (
            OperandTarget(
                RegionOutputRef(endpoint.node_id, operand.id),
                PositionMap.row_major_reshape(public.domain, operand.position_domain),
                (endpoint,),
            ),
        ),
    )


def _wrapped_copy(body_name):
    class Wrapped(Kernel):
        id = "wrapped_copy"
        extent = PublicCopy.extent
        lanes = PublicCopy.lanes
        element_type = PublicCopy.element_type
        result_domain = PublicCopy.result_domain
        type_ready = PublicCopy.type_ready
        domain_ready = PublicCopy.domain_ready
        result_type = PublicCopy.result_type
        domain = PublicCopy.domain
        implementation = KernelChoice(
            Subspace(PublicCopy, extent=extent, lanes=lanes), node_id=body_name
        )
        activation = NetworkBoundary(implementation.input("input"))
        result = NetworkBoundary(implementation.output("output"))
        public_operands = (
            PublicOperandDeclaration(
                "result", "output", result_type, domain, _boundary_copy_export
            ),
        )

    return Wrapped


def test_private_child_rename_and_extra_space_preserve_public_binding_and_values():
    old = copy_point(_wrapped_copy("old_body"))
    renamed_type = _wrapped_copy("renamed_body")

    class Enclosure(Space):
        extent = Input(int)
        lanes = Input(int)
        inner = Subspace(renamed_type, extent=extent, lanes=lanes)

    class Root(Space):
        extent = Problem(int)
        lanes = Problem(int)
        additional = Subspace(Enclosure, extent=extent, lanes=lanes)

    renamed = Root.start({Root.extent: 8, Root.lanes: 2}).additional.inner
    # A caller binds only this semantic key; neither private body id appears in it.
    public_key = "result"
    first = operand_export(old, public_key)
    second = operand_export(renamed, public_key)
    assert isinstance(first, Decided), first
    assert isinstance(second, Decided), second
    assert first.value.operand == second.value.operand
    assert first.value.targets[0].ref.node_id == "old_body"
    assert second.value.targets[0].ref.node_id == "renamed_body"
    assert first.value.public_beat(
        logical_network(old.logical.accepted_answer.value), 2
    ) == second.value.public_beat(logical_network(renamed.logical.accepted_answer.value), 2)


def test_direct_region_required_value_has_no_invented_presentation():
    network = logical_network(RegionResult(required_region(4)))
    domain = RectangularDomain((4,))
    export = OperandExport(
        PublicOperand("weights", "input", DataType["INT8"], domain),
        (
            OperandTarget(
                RegionInputRef("root", "value"), PositionMap.row_major_reshape(domain, domain)
            ),
        ),
    )
    validate_operand_export(network, export)
    assert export.targets[0].presentations == ()
    with pytest.raises(InterfaceError, match="presentation"):
        export.presentation()


def test_explicit_region_with_implementation_children_and_wrong_kind_child():
    class RegionWithChildren(PortedModule):
        helper = KernelChoice(Subspace(UnavailableModule, width=PortedModule.width))

    logical = _design(RegionWithChildren).logical.accepted_answer
    assert isinstance(logical, Decided)
    assert isinstance(logical.value, RegionResult)
    assert isinstance(_design(NestedCompositeKernel).child_region("inner"), Absent)


def test_multiple_presentations_select_a_port_and_unported_input_stays_unported():
    region = _region(4, 2)
    outputs = (
        *region.outputs,
        replace(region.outputs[0], port=replace(region.outputs[0].port, id="other")),
    )
    network = logical_network(RegionResult(replace(region, outputs=outputs)))
    domain = RectangularDomain((4,))
    export = OperandExport(
        PublicOperand("result", "output", DataType["INT8"], domain),
        (
            OperandTarget(
                RegionOutputRef("root", "value"),
                PositionMap.row_major_reshape(domain, domain),
                (RegionEndpoint("root", "output"), RegionEndpoint("root", "other")),
            ),
        ),
    )
    validate_operand_export(network, export)
    with pytest.raises(InterfaceError, match="presentation"):
        export.presentation()
    assert export.public_beat(network, 1, RegionEndpoint("root", "other")) == ((2,), (3,))


@pytest.mark.parametrize("bad", ["type", "direction", "domain", "coverage", "presentation"])
def test_bad_exports_refuse(bad):
    region = _region(4, 2)
    network = logical_network(RegionResult(region))
    domain = RectangularDomain((4,))
    public = PublicOperand("activation", "input", DataType["INT8"], domain)
    target = OperandTarget(
        RegionInputRef("root", "value"),
        PositionMap.row_major_reshape(domain, domain),
        (RegionEndpoint("root", "input"),),
    )
    if bad == "type":
        public = replace(public, element_type=DataType["INT4"])
    elif bad == "direction":
        public = replace(public, direction="output")
    elif bad == "domain":
        public = replace(public, domain=RectangularDomain((2, 2)))
    elif bad == "coverage":
        target = replace(
            target, position_map=PositionMap({(0,): (0,)}).bind_domains(domain, domain)
        )
    else:
        target = replace(target, presentations=(RegionEndpoint("root", "output"),))
    with pytest.raises(ValueError):
        validate_operand_export(network, OperandExport(public, (target,)))


def test_replay_type_and_result_domain_do_not_require_simd():
    class Root(Space):
        rows = Problem(int)
        width = Problem(int)
        count = Problem(int)
        datatype = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
        replay = Subspace(
            ActivationReplayKernel,
            repetitions=rows,
            matrix_width=width,
            matrix_height=count,
            activation_type=datatype,
        )

    replay = Root.start(
        {Root.rows: 2, Root.width: 8, Root.count: 3, Root.datatype: DataType["INT4"]}
    ).replay
    assert operand_type(replay, "result") == Decided(DataType["INT4"])
    assert operand_domain(replay, "result") == Decided(RectangularDomain((6, 8)))
    assert isinstance(replay.logical.accepted_answer, Unresolved)
