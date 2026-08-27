# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic parameter-delivery Kernel definition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    AbsenceMode,
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
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernel import (
    BindingDefinition,
    KernelDefinition,
    RegionDeclaration,
    build_kernel_semantic_declarations,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import Port

E = TypeVar("E", bound=Enum)


class CyclicParameterBinding(str, Enum):
    """Concrete cyclic-memory implementation identities."""

    FINN_RTL_MEMSTREAM = "finn_rtl_memstream"
    FINNLIB_HLS_MEMSTREAM = "finnlib_hls_memstream"


class CyclicRamStyle(str, Enum):
    """RAM implementations exposed by the FINN RTL memstream binding."""

    AUTO = "auto"
    BRAM = "block"
    LUTRAM = "distributed"
    URAM = "ultra"


@dataclass(frozen=True)
class CyclicTargetMemoryCapabilities:
    """Target facts used by the current on-chip delivery constraints."""

    supports_initialized_uram: bool


@dataclass(frozen=True)
class CyclicParameterBindingSelection:
    """Resolved cyclic-memory association, independent of feasibility."""

    binding_id: str
    ram_style: CyclicRamStyle | None
    pumped_memory: bool | None


class CyclicParameterKernelPaths:
    """Stable paths for the cyclic parameter-delivery Kernel."""

    OUTPUT_PORT = QualifiedPath("problem.cyclic_parameter.output_port")
    INITIALIZER_AVAILABLE = QualifiedPath("problem.cyclic_parameter.initializer_available")
    RUNTIME_WRITABLE = QualifiedPath("problem.cyclic_parameter.runtime_writable")
    TARGET_MEMORY_CAPABILITIES = QualifiedPath(
        "problem.target.cyclic_parameter_memory_capabilities"
    )

    BINDING = QualifiedPath("cyclic_parameter.binding")
    RAM_STYLE = QualifiedPath("cyclic_parameter.binding.ram_style")
    PUMPED_MEMORY = QualifiedPath("cyclic_parameter.binding.pumped_memory")

    LOCAL_STATE_SOURCE_REGION = QualifiedPath(
        "semantic.cyclic_parameter.region_declarations.local_state_source"
    )
    REGION = QualifiedPath("semantic.cyclic_parameter.region")
    REGION_VALIDATION = QualifiedPath("semantic.cyclic_parameter.region_validation")
    BINDING_SELECTION = QualifiedPath("binding.cyclic_parameter.selection")

    REGION_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.cyclic_parameter.region_structurally_well_formed"
    )
    LOCAL_STATE_AVAILABLE = QualifiedPath("constraint.cyclic_parameter.local_state_available")
    URAM_INITIALIZATION_SUPPORTED = QualifiedPath(
        "constraint.cyclic_parameter.uram_initialization_supported"
    )
    PUMPING_SUPPORTED = QualifiedPath("constraint.cyclic_parameter.pumping_supported")


_BOOL_SEMANTICS = ValueSemantics.immutable_nominal(bool, name="boolean")
_PORT_SEMANTICS = ValueSemantics.immutable_nominal(Port, name="Port")
_TARGET_MEMORY_SEMANTICS = ValueSemantics.immutable_nominal(
    CyclicTargetMemoryCapabilities, name="CyclicTargetMemoryCapabilities"
)
_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_BOOL_OBJECT_SEMANTICS = as_object_semantics(_BOOL_SEMANTICS)
_PORT_OBJECT_SEMANTICS = as_object_semantics(_PORT_SEMANTICS)
_TARGET_MEMORY_OBJECT_SEMANTICS = as_object_semantics(_TARGET_MEMORY_SEMANTICS)


def _enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


_BINDING_SEMANTICS = _enum_semantics(CyclicParameterBinding)
_RAM_STYLE_SEMANTICS = _enum_semantics(CyclicRamStyle)
_BINDING_SELECTION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(
        CyclicParameterBindingSelection, name="CyclicParameterBindingSelection"
    )
)


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


_BINDING_REF = DependencyRef.decision(
    "binding", CyclicParameterKernelPaths.BINDING, _BINDING_SEMANTICS
)
_RAM_STYLE_REF = DependencyRef.decision(
    "ram_style",
    CyclicParameterKernelPaths.RAM_STYLE,
    _RAM_STYLE_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_PUMPED_MEMORY_REF = DependencyRef.decision(
    "pumped_memory",
    CyclicParameterKernelPaths.PUMPED_MEMORY,
    _BOOL_OBJECT_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_OUTPUT_PORT_REF = DependencyRef.problem(
    "output_port", CyclicParameterKernelPaths.OUTPUT_PORT, _PORT_OBJECT_SEMANTICS
)


def _binding_applies(binding: CyclicParameterBinding) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["binding"] is binding)

    return EvaluatorSpec((_BINDING_REF,), evaluate)


def _finn_rtl_uram_applies(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        dependencies["binding"] is CyclicParameterBinding.FINN_RTL_MEMSTREAM
        and dependencies["ram_style"] is CyclicRamStyle.URAM
    )


def _derive_local_state_source_region(dependencies: DependencyView) -> Answer[object]:
    return Decided(construct_cyclic_parameter_region(cast(Port, dependencies["output_port"])))


_CYCLIC_REGION_DECLARATIONS = (
    RegionDeclaration(
        "local_state_source",
        CyclicParameterKernelPaths.LOCAL_STATE_SOURCE_REGION,
    ),
)


def _local_state_available(dependencies: DependencyView) -> Answer[bool]:
    binding = cast(CyclicParameterBinding, dependencies["binding"])
    initialized = cast(bool, dependencies["initializer_available"])
    runtime_writable = cast(bool, dependencies["runtime_writable"])
    if binding is CyclicParameterBinding.FINNLIB_HLS_MEMSTREAM:
        return Decided(initialized)
    return Decided(initialized or runtime_writable)


def _uram_initialization_supported(dependencies: DependencyView) -> Answer[bool]:
    if cast(bool, dependencies["runtime_writable"]):
        return Decided(True)
    target = dependencies["target_memory_capabilities"]
    if target is ABSENT:
        return Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "cyclic-target-memory-capabilities-missing",
                    CyclicParameterKernelPaths.URAM_INITIALIZATION_SUPPORTED,
                    "initialized URAM requires target memory capabilities",
                    trace=(CyclicParameterKernelPaths.TARGET_MEMORY_CAPABILITIES,),
                ),
            )
        )
    return Decided(cast(CyclicTargetMemoryCapabilities, target).supports_initialized_uram)


def _pumping_supported(dependencies: DependencyView) -> Answer[bool]:
    pumped = dependencies["pumped_memory"]
    if pumped is ABSENT or not cast(bool, pumped):
        return Decided(True)
    output_port = cast(Port, dependencies["output_port"])
    return Decided(output_port.beat_sequence.elements_per_beat > 1)


def _derive_binding_selection(dependencies: DependencyView) -> Answer[object]:
    ram_style = dependencies["ram_style"]
    pumped_memory = dependencies["pumped_memory"]
    return Decided(
        CyclicParameterBindingSelection(
            cast(CyclicParameterBinding, dependencies["binding"]).value,
            None if ram_style is ABSENT else cast(CyclicRamStyle, ram_style),
            None if pumped_memory is ABSENT else cast(bool, pumped_memory),
        )
    )


def build_cyclic_parameter_kernel_spec(
    output_port_dependency: DependencyRef | None = None,
    *,
    problem_fields_required: bool = True,
) -> DesignSpaceSpec:
    """Build a flat cyclic local-state-source specification."""
    semantic = build_kernel_semantic_declarations(
        _CYCLIC_REGION_DECLARATIONS,
        selected_region_path=CyclicParameterKernelPaths.REGION,
        validation_report_path=CyclicParameterKernelPaths.REGION_VALIDATION,
        structural_constraint_path=CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
    )
    output_port_fields: tuple[ProblemField, ...]
    if output_port_dependency is None:
        output_port_dependency = _OUTPUT_PORT_REF
        output_port_fields = (
            ProblemField(CyclicParameterKernelPaths.OUTPUT_PORT, _PORT_OBJECT_SEMANTICS),
        )
    else:
        output_port_fields = ()
        if output_port_dependency.name != "output_port":
            raise ValueError("output_port_dependency must use the name 'output_port'")
        if not output_port_dependency.value_semantics.is_compatible_with(_PORT_OBJECT_SEMANTICS):
            raise TypeError("output_port_dependency must carry Port values")

    ram_applies = _binding_applies(CyclicParameterBinding.FINN_RTL_MEMSTREAM)
    pumped_applies = _binding_applies(CyclicParameterBinding.FINN_RTL_MEMSTREAM)
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
                ProblemField(
                    CyclicParameterKernelPaths.TARGET_MEMORY_CAPABILITIES,
                    _TARGET_MEMORY_OBJECT_SEMANTICS,
                    required=False,
                ),
            )
        ),
        decisions=(
            Decision(
                CyclicParameterKernelPaths.BINDING,
                _BINDING_SEMANTICS,
                _finite_domain(tuple(CyclicParameterBinding)),
            ),
            Decision(
                CyclicParameterKernelPaths.RAM_STYLE,
                _RAM_STYLE_SEMANTICS,
                _finite_domain(tuple(CyclicRamStyle)),
                applies_if=ram_applies,
            ),
            Decision(
                CyclicParameterKernelPaths.PUMPED_MEMORY,
                _BOOL_OBJECT_SEMANTICS,
                _finite_domain((False, True)),
                applies_if=pumped_applies,
            ),
        ),
        properties=(
            DerivedProperty(
                CyclicParameterKernelPaths.LOCAL_STATE_SOURCE_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec((output_port_dependency,), _derive_local_state_source_region),
            ),
            semantic.selected_region,
            semantic.validation_report,
            DerivedProperty(
                CyclicParameterKernelPaths.BINDING_SELECTION,
                _BINDING_SELECTION_SEMANTICS,
                EvaluatorSpec(
                    (_BINDING_REF, _RAM_STYLE_REF, _PUMPED_MEMORY_REF),
                    _derive_binding_selection,
                ),
            ),
        ),
        constraints=(
            semantic.structural_constraint,
            Constraint(
                CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE,
                EvaluatorSpec(
                    (
                        _BINDING_REF,
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
                CyclicParameterKernelPaths.URAM_INITIALIZATION_SUPPORTED,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "runtime_writable",
                            CyclicParameterKernelPaths.RUNTIME_WRITABLE,
                            _BOOL_OBJECT_SEMANTICS,
                        ),
                        DependencyRef.problem(
                            "target_memory_capabilities",
                            CyclicParameterKernelPaths.TARGET_MEMORY_CAPABILITIES,
                            _TARGET_MEMORY_OBJECT_SEMANTICS,
                            absence=AbsenceMode.ALLOWS_ABSENT,
                        ),
                    ),
                    _uram_initialization_supported,
                ),
                applies_if=EvaluatorSpec((_BINDING_REF, _RAM_STYLE_REF), _finn_rtl_uram_applies),
            ),
            Constraint(
                CyclicParameterKernelPaths.PUMPING_SUPPORTED,
                EvaluatorSpec((_PUMPED_MEMORY_REF, output_port_dependency), _pumping_supported),
                applies_if=pumped_applies,
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
                    CyclicParameterKernelPaths.URAM_INITIALIZATION_SUPPORTED,
                    CyclicParameterKernelPaths.PUMPING_SUPPORTED,
                ),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "cyclic_model_structural",
                properties=(
                    CyclicParameterKernelPaths.REGION,
                    CyclicParameterKernelPaths.REGION_VALIDATION,
                ),
                constraints=(CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
            ReadinessProfile(
                "cyclic_binding_feasibility",
                decisions=(
                    CyclicParameterKernelPaths.BINDING,
                    CyclicParameterKernelPaths.RAM_STYLE,
                    CyclicParameterKernelPaths.PUMPED_MEMORY,
                ),
                properties=(
                    CyclicParameterKernelPaths.REGION,
                    CyclicParameterKernelPaths.REGION_VALIDATION,
                    CyclicParameterKernelPaths.BINDING_SELECTION,
                ),
                constraints=(
                    CyclicParameterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE,
                    CyclicParameterKernelPaths.URAM_INITIALIZATION_SUPPORTED,
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
    binding_selection_path=CyclicParameterKernelPaths.BINDING_SELECTION,
    structural_readiness_profile="cyclic_model_structural",
    binding_readiness_profile="cyclic_binding_feasibility",
)
CYCLIC_PARAMETER_KERNEL_SPEC = CYCLIC_PARAMETER_KERNEL.spec

__all__ = [
    "CYCLIC_PARAMETER_KERNEL",
    "CYCLIC_PARAMETER_KERNEL_SPEC",
    "CyclicParameterBinding",
    "CyclicParameterBindingSelection",
    "CyclicParameterKernelPaths",
    "CyclicRamStyle",
    "CyclicTargetMemoryCapabilities",
    "build_cyclic_parameter_kernel_spec",
]
