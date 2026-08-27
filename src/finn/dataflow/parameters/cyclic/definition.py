# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic parameter-delivery Kernel definition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

from finn.dataflow.design import (
    DATAFLOW_REGION_SEMANTICS,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernel import (
    BindingDefinition,
    KernelDefinition,
    RegionDeclaration,
    build_kernel_semantic_declarations,
)
from finn.dataflow.parameters.cyclic.region import (
    construct_chunked_cyclic_parameter_region,
    construct_full_tile_cyclic_parameter_region,
)
from finn.dataflow.region import Port

E = TypeVar("E", bound=Enum)


class CyclicParameterRegionDeclaration(str, Enum):
    """Stable identities for the two evidenced cyclic output contracts."""

    FULL_TILE = "full_tile"
    CHUNKED = "chunked"


class CyclicParameterBinding(str, Enum):
    """Concrete cyclic-memory implementation identities."""

    FINN_RTL_MEMSTREAM = "finn_rtl_memstream"
    FINNLIB_HLS_MEMSTREAM = "finnlib_hls_memstream"


class CyclicRamStyle(str, Enum):
    """Binding-local on-chip memory implementation choice."""

    AUTO = "auto"
    BRAM = "block"
    LUTRAM = "distributed"
    URAM = "ultra"


@dataclass(frozen=True)
class CyclicParameterBindingWitness:
    """Binding-owned account of how the cyclic source realizes its region."""

    binding_id: str
    region_declaration_id: str
    ram_style: CyclicRamStyle
    pumped_memory: bool
    mechanisms: tuple[str, ...] = ("local_parameter_state", "cyclic_sequence_replay")


class CyclicParameterKernelPaths:
    """Stable paths for the cyclic parameter-delivery Kernel."""

    OUTPUT_PORT = QualifiedPath("problem.cyclic_parameter.output_port")
    INITIALIZER_AVAILABLE = QualifiedPath("problem.cyclic_parameter.initializer_available")
    RUNTIME_WRITABLE = QualifiedPath("problem.cyclic_parameter.runtime_writable")

    REGION_DECLARATION = QualifiedPath("cyclic_parameter.region_declaration")
    BINDING = QualifiedPath("cyclic_parameter.binding")
    RAM_STYLE = QualifiedPath("cyclic_parameter.binding.ram_style")
    PUMPED_MEMORY = QualifiedPath("cyclic_parameter.binding.pumped_memory")

    FULL_TILE_REGION = QualifiedPath("semantic.cyclic_parameter.region_declarations.full_tile")
    CHUNKED_REGION = QualifiedPath("semantic.cyclic_parameter.region_declarations.chunked")
    REGION = QualifiedPath("semantic.cyclic_parameter.region")
    REGION_VALIDATION = QualifiedPath("semantic.cyclic_parameter.region_validation")
    BINDING_WITNESS = QualifiedPath("binding.cyclic_parameter.witness")

    REGION_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.cyclic_parameter.region_structurally_well_formed"
    )
    LOCAL_STATE_AVAILABLE = QualifiedPath("constraint.cyclic_parameter.local_state_available")
    PUMPING_SUPPORTED = QualifiedPath("constraint.cyclic_parameter.pumping_supported")


_BOOL_SEMANTICS = ValueSemantics.immutable_nominal(bool, name="boolean")
_PORT_SEMANTICS = ValueSemantics.immutable_nominal(Port, name="Port")
_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_BOOL_OBJECT_SEMANTICS = as_object_semantics(_BOOL_SEMANTICS)
_PORT_OBJECT_SEMANTICS = as_object_semantics(_PORT_SEMANTICS)


def _enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


_REGION_DECLARATION_SEMANTICS = _enum_semantics(CyclicParameterRegionDeclaration)
_BINDING_SEMANTICS = _enum_semantics(CyclicParameterBinding)
_RAM_STYLE_SEMANTICS = _enum_semantics(CyclicRamStyle)
_WITNESS_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(
        CyclicParameterBindingWitness, name="CyclicParameterBindingWitness"
    )
)


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


_REGION_DECLARATION_REF = DependencyRef.decision(
    "region_declaration",
    CyclicParameterKernelPaths.REGION_DECLARATION,
    _REGION_DECLARATION_SEMANTICS,
)
_BINDING_REF = DependencyRef.decision(
    "binding", CyclicParameterKernelPaths.BINDING, _BINDING_SEMANTICS
)
_RAM_STYLE_REF = DependencyRef.decision(
    "ram_style", CyclicParameterKernelPaths.RAM_STYLE, _RAM_STYLE_SEMANTICS
)
_PUMPED_MEMORY_REF = DependencyRef.decision(
    "pumped_memory", CyclicParameterKernelPaths.PUMPED_MEMORY, _BOOL_OBJECT_SEMANTICS
)


def _region_applies(
    declaration: CyclicParameterRegionDeclaration,
) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["region_declaration"] is declaration)

    return EvaluatorSpec((_REGION_DECLARATION_REF,), evaluate)


def _derive_full_tile_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        construct_full_tile_cyclic_parameter_region(cast(Port, dependencies["output_port"]))
    )


def _derive_chunked_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        construct_chunked_cyclic_parameter_region(cast(Port, dependencies["output_port"]))
    )


_CYCLIC_REGION_DECLARATIONS = (
    RegionDeclaration(
        CyclicParameterRegionDeclaration.FULL_TILE.value,
        CyclicParameterKernelPaths.FULL_TILE_REGION,
    ),
    RegionDeclaration(
        CyclicParameterRegionDeclaration.CHUNKED.value,
        CyclicParameterKernelPaths.CHUNKED_REGION,
    ),
)


def _local_state_available(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        cast(bool, dependencies["initializer_available"])
        or cast(bool, dependencies["runtime_writable"])
    )


def _pumping_supported(dependencies: DependencyView) -> Answer[bool]:
    if not cast(bool, dependencies["pumped_memory"]):
        return Decided(True)
    return Decided(
        dependencies["binding"] is CyclicParameterBinding.FINN_RTL_MEMSTREAM
        and dependencies["region_declaration"] is CyclicParameterRegionDeclaration.FULL_TILE
    )


def _derive_binding_witness(dependencies: DependencyView) -> Answer[object]:
    binding = cast(CyclicParameterBinding, dependencies["binding"])
    declaration = cast(CyclicParameterRegionDeclaration, dependencies["region_declaration"])
    return Decided(
        CyclicParameterBindingWitness(
            binding.value,
            declaration.value,
            cast(CyclicRamStyle, dependencies["ram_style"]),
            cast(bool, dependencies["pumped_memory"]),
        )
    )


def build_cyclic_parameter_kernel_spec(
    output_port_dependency: DependencyRef | None = None,
    *,
    problem_fields_required: bool = True,
) -> DesignSpaceSpec:
    """Build a flat cyclic-delivery specification.

    A standalone Kernel reads its exact output port from problem data.  A larger
    scope may instead supply a property dependency, preserving the same output
    region declarations without evaluating a nested design space.
    """
    semantic = build_kernel_semantic_declarations(
        _CYCLIC_REGION_DECLARATIONS,
        selected_region_path=CyclicParameterKernelPaths.REGION,
        validation_report_path=CyclicParameterKernelPaths.REGION_VALIDATION,
        structural_constraint_path=CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
    )
    output_port_fields: tuple[ProblemField, ...]
    if output_port_dependency is None:
        output_port_dependency = DependencyRef.problem(
            "output_port",
            CyclicParameterKernelPaths.OUTPUT_PORT,
            _PORT_OBJECT_SEMANTICS,
        )
        output_port_fields = (
            ProblemField(CyclicParameterKernelPaths.OUTPUT_PORT, _PORT_OBJECT_SEMANTICS),
        )
    else:
        output_port_fields = ()
        if output_port_dependency.name != "output_port":
            raise ValueError("output_port_dependency must use the name 'output_port'")
        if not output_port_dependency.value_semantics.is_compatible_with(_PORT_OBJECT_SEMANTICS):
            raise TypeError("output_port_dependency must carry Port values")

    return DesignSpaceSpec(
        problem_schema=ProblemSchema(
            (
                *output_port_fields,
                ProblemField(
                    CyclicParameterKernelPaths.INITIALIZER_AVAILABLE,
                    _BOOL_OBJECT_SEMANTICS,
                    required=problem_fields_required,
                ),
                ProblemField(
                    CyclicParameterKernelPaths.RUNTIME_WRITABLE,
                    _BOOL_OBJECT_SEMANTICS,
                    required=problem_fields_required,
                ),
            )
        ),
        decisions=(
            Decision(
                CyclicParameterKernelPaths.REGION_DECLARATION,
                _REGION_DECLARATION_SEMANTICS,
                _finite_domain(tuple(CyclicParameterRegionDeclaration)),
            ),
            Decision(
                CyclicParameterKernelPaths.BINDING,
                _BINDING_SEMANTICS,
                _finite_domain(tuple(CyclicParameterBinding)),
            ),
            Decision(
                CyclicParameterKernelPaths.RAM_STYLE,
                _RAM_STYLE_SEMANTICS,
                _finite_domain(tuple(CyclicRamStyle)),
            ),
            Decision(
                CyclicParameterKernelPaths.PUMPED_MEMORY,
                _BOOL_OBJECT_SEMANTICS,
                _finite_domain((False, True)),
            ),
        ),
        properties=(
            DerivedProperty(
                CyclicParameterKernelPaths.FULL_TILE_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec((output_port_dependency,), _derive_full_tile_region),
                applies_if=_region_applies(CyclicParameterRegionDeclaration.FULL_TILE),
            ),
            DerivedProperty(
                CyclicParameterKernelPaths.CHUNKED_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec((output_port_dependency,), _derive_chunked_region),
                applies_if=_region_applies(CyclicParameterRegionDeclaration.CHUNKED),
            ),
            semantic.selected_region,
            semantic.validation_report,
            DerivedProperty(
                CyclicParameterKernelPaths.BINDING_WITNESS,
                _WITNESS_SEMANTICS,
                EvaluatorSpec(
                    (
                        _BINDING_REF,
                        _REGION_DECLARATION_REF,
                        _RAM_STYLE_REF,
                        _PUMPED_MEMORY_REF,
                    ),
                    _derive_binding_witness,
                ),
            ),
        ),
        constraints=(
            semantic.structural_constraint,
            Constraint(
                CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "initializer_available",
                            CyclicParameterKernelPaths.INITIALIZER_AVAILABLE,
                            _BOOL_OBJECT_SEMANTICS,
                        ),
                        DependencyRef.problem(
                            "runtime_writable",
                            CyclicParameterKernelPaths.RUNTIME_WRITABLE,
                            _BOOL_OBJECT_SEMANTICS,
                        ),
                    ),
                    _local_state_available,
                ),
            ),
            Constraint(
                CyclicParameterKernelPaths.PUMPING_SUPPORTED,
                EvaluatorSpec(
                    (_PUMPED_MEMORY_REF, _BINDING_REF, _REGION_DECLARATION_REF),
                    _pumping_supported,
                ),
            ),
        ),
        constraint_sets=(
            ConstraintSet(
                "cyclic_model_structural",
                (CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
            ConstraintSet(
                "cyclic_binding_feasibility",
                (
                    CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE,
                    CyclicParameterKernelPaths.PUMPING_SUPPORTED,
                ),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "cyclic_model_structural",
                decisions=(CyclicParameterKernelPaths.REGION_DECLARATION,),
                properties=(
                    CyclicParameterKernelPaths.REGION,
                    CyclicParameterKernelPaths.REGION_VALIDATION,
                ),
                constraints=(CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
            ReadinessProfile(
                "cyclic_binding_feasibility",
                decisions=(
                    CyclicParameterKernelPaths.REGION_DECLARATION,
                    CyclicParameterKernelPaths.BINDING,
                    CyclicParameterKernelPaths.RAM_STYLE,
                    CyclicParameterKernelPaths.PUMPED_MEMORY,
                ),
                properties=(
                    CyclicParameterKernelPaths.REGION,
                    CyclicParameterKernelPaths.REGION_VALIDATION,
                    CyclicParameterKernelPaths.BINDING_WITNESS,
                ),
                constraints=(
                    CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE,
                    CyclicParameterKernelPaths.PUMPING_SUPPORTED,
                ),
            ),
        ),
    )


CYCLIC_PARAMETER_KERNEL = KernelDefinition(
    id="parameter.cyclic",
    spec=build_cyclic_parameter_kernel_spec(),
    region_declarations=_CYCLIC_REGION_DECLARATIONS,
    binding_definitions=tuple(
        BindingDefinition(binding.value) for binding in CyclicParameterBinding
    ),
    selected_region_path=CyclicParameterKernelPaths.REGION,
    binding_decision_path=CyclicParameterKernelPaths.BINDING,
    binding_witness_path=CyclicParameterKernelPaths.BINDING_WITNESS,
    structural_readiness_profile="cyclic_model_structural",
    binding_readiness_profile="cyclic_binding_feasibility",
)
CYCLIC_PARAMETER_KERNEL_SPEC = CYCLIC_PARAMETER_KERNEL.spec

__all__ = [
    "CYCLIC_PARAMETER_KERNEL",
    "CYCLIC_PARAMETER_KERNEL_SPEC",
    "CyclicParameterBinding",
    "CyclicParameterBindingWitness",
    "CyclicParameterKernelPaths",
    "CyclicParameterRegionDeclaration",
    "CyclicRamStyle",
    "build_cyclic_parameter_kernel_spec",
]
