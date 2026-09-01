# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU-common input supply for external or FINN RTL memstream delivery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from finn.dataflow.authoring.input_supply import (
    InputSupplyAlternative,
    InputSupplyContext,
    InputSupplyDeclaration,
    SupplierAttachment,
    declare_input_supply,
)
from finn.dataflow.authoring.scope import Ref, Scope, finite, predicate
from finn.dataflow.design import Answer, DesignSpaceSpec, EvaluatorSpec
from finn.dataflow.ops.mvau.associations import (
    BindingLocalStateDestination,
    CoordinateMappingKind,
    MVAUParameterTopology,
    MVAUSourceAssociation,
)
from finn.dataflow.kernels.finn_rtl_memstream import (
    FinnRtlMemstreamInputs,
    FinnRtlMemstreamKernel,
)
from finn.dataflow.ops.mvau.problem import MVAUProblem
from finn.dataflow.parameters.cyclic.computation import CYCLIC_PARAMETER_DELIVERY
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import InputInterface, Port

EXTERNAL_SUPPLY = "external"
FINN_RTL_MEMSTREAM_SUPPLY = "finn_rtl_memstream"
WEIGHT_SOURCE_OPERAND = "weight"
DELIVERY_NODE = "delivery"
DELIVERY_EDGE = "weight"


@dataclass(frozen=True)
class MVAUWeightSupplySettings:
    """Operation-owned facts and choices shared by every mapped MVAU design."""

    problem: MVAUProblem
    ram_style: Ref[CyclicRamStyle]
    pumped_memory: Ref[bool]


@dataclass(frozen=True)
class MVAUInputSupply:
    """The closed MVAU supply declaration and its shared Kernel choices."""

    declaration: InputSupplyDeclaration
    settings: MVAUWeightSupplySettings


def _memstream_selected(choice: Ref[str]) -> EvaluatorSpec[Answer[bool]]:
    return predicate(
        choice.path,
        {"supply": choice},
        lambda supply: supply == FINN_RTL_MEMSTREAM_SUPPLY,
    )


def _output_port(consumer: InputInterface) -> Port:
    return Port("weight", consumer.port.operand, consumer.port.beat_sequence)


def _declare_finn_rtl_memstream(context: InputSupplyContext) -> SupplierAttachment:
    settings = cast(MVAUWeightSupplySettings, context.inputs)
    output_port = context.design.derived(
        context.name("output_port"),
        Port,
        dependencies={"consumer": context.mapping.consumer},
        evaluate=_output_port,
        applies_if=context.applies_if,
    )
    delivery = context.design.region(
        context.name("region"),
        node_id=DELIVERY_NODE,
        dependencies={"output_port": output_port},
        evaluate=construct_cyclic_parameter_region,
        computation=CYCLIC_PARAMETER_DELIVERY,
        applies_if=context.applies_if,
    )
    sets = context.design.derived(
        context.name("sets"),
        int,
        dependencies={},
        evaluate=lambda: 1,
        applies_if=context.applies_if,
    )
    context.design.kernels(
        "delivery",
        covers=(delivery,),
        candidates=(FinnRtlMemstreamKernel,),
        inputs=FinnRtlMemstreamInputs(
            role=delivery.role,
            region=delivery.region,
            computation=delivery.computation,
            output_port=output_port,
            initializer_available=settings.problem.weight_initializer_available,
            runtime_writable=settings.problem.runtime_writable,
            target_memory_capabilities=settings.problem.target_memory_capabilities,
            ram_style=settings.ram_style,
            pumped_memory=settings.pumped_memory,
            sets=sets,
        ),
        applies_if=context.applies_if,
    )
    return SupplierAttachment(delivery, "weight", DELIVERY_EDGE)


def declare_mvau_input_supply(problem: MVAUProblem) -> MVAUInputSupply:
    """Declare ``external | finn_rtl_memstream`` once at MVAU Operation scope."""

    configured: MVAUWeightSupplySettings | None = None

    def configure(scope: Scope, choice: Ref[str]) -> MVAUWeightSupplySettings:
        nonlocal configured
        applies = _memstream_selected(choice)
        configured = MVAUWeightSupplySettings(
            problem,
            scope.decision(
                "finn_rtl_memstream.ram_style",
                CyclicRamStyle,
                domain=finite(tuple(CyclicRamStyle)),
                applies_if=applies,
            ),
            scope.decision(
                "finn_rtl_memstream.pumped_memory",
                bool,
                domain=finite((False, True)),
                applies_if=applies,
            ),
        )
        return configured

    declaration = declare_input_supply(
        "mvau_weight_supply",
        "1",
        namespace="mvau.input.weight",
        source_operand=WEIGHT_SOURCE_OPERAND,
        alternatives=(
            InputSupplyAlternative(FINN_RTL_MEMSTREAM_SUPPLY, _declare_finn_rtl_memstream),
        ),
        configure=configure,
    )
    if configured is None:
        raise AssertionError("the MVAU input-supply configuration did not run")
    return MVAUInputSupply(declaration, configured)


def _supplied_source_association(
    core_source_association: MVAUSourceAssociation,
    supply: str,
) -> MVAUSourceAssociation:
    if supply == EXTERNAL_SUPPLY:
        return core_source_association
    operands = tuple(
        replace(
            operand,
            destination=BindingLocalStateDestination(DELIVERY_NODE, "weights"),
            mapping=CoordinateMappingKind.TRANSPOSE_2D,
        )
        if operand.role == WEIGHT_SOURCE_OPERAND
        else operand
        for operand in core_source_association.operands
    )
    return replace(
        core_source_association,
        parameter_topology=MVAUParameterTopology.CYCLIC,
        operands=operands,
        supply_kernel_id=FINN_RTL_MEMSTREAM_SUPPLY,
    )


def declare_supplied_source_association(
    namespace: str,
    core_source_association: Ref[MVAUSourceAssociation],
    supply: InputSupplyDeclaration,
) -> tuple[Ref[MVAUSourceAssociation], DesignSpaceSpec]:
    """Derive the selected source/local-state association for one MVAU design."""

    scope = Scope(namespace)
    association = scope.derived(
        "source_association",
        MVAUSourceAssociation,
        dependencies={
            "core_source_association": core_source_association,
            "supply": supply.choice,
        },
        evaluate=_supplied_source_association,
    )
    return association, scope.spec()


__all__ = [
    "DELIVERY_EDGE",
    "DELIVERY_NODE",
    "EXTERNAL_SUPPLY",
    "FINN_RTL_MEMSTREAM_SUPPLY",
    "MVAUInputSupply",
    "MVAUWeightSupplySettings",
    "WEIGHT_SOURCE_OPERAND",
    "declare_mvau_input_supply",
    "declare_supplied_source_association",
]
