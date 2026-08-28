# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic parameter-supply Kernels driven by compute demand.

Each supplier is a Kernel with an independently meaningful production schedule:
it replays a stored parameter array as an ordered boundary sequence.  Its Region
is derived from the demand the selected compute Kernel published, never from a
duplicated delivery tile choice.

One supplier-local choice remains because it is a real alternative: a memstream
may either emit exactly the demanded sequence, or emit its natural full-tile
sequence and leave the reordering to a separately selected adapter Kernel.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar, cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    AbsenceMode,
    Answer,
    Constraint,
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
    QualifiedPath,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import Kernel, KernelExport, KernelProvider, KernelSelection
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import Port

E = TypeVar("E", bound=Enum)


class MVAUWeightSupplyKernelId(str, Enum):
    """Stable identities of the initial weight-supply Kernel pool."""

    FINN_RTL_MEMSTREAM = "finn_rtl_memstream"
    FINNLIB_HLS_MEMSTREAM = "finnlib_hls_memstream"


class CyclicRamStyle(str, Enum):
    """RAM implementations exposed by the FINN RTL memstream Kernel."""

    AUTO = "auto"
    BRAM = "block"
    LUTRAM = "distributed"
    URAM = "ultra"


class WeightOrganization(str, Enum):
    """Whether a supplier serves the demand exactly or its own full tile."""

    AS_DEMANDED = "as_demanded"
    STANDARD_FULL_TILE = "standard_full_tile"


#: Selection name and path root of the weight-supply pool.
MVAU_WEIGHT_SUPPLY_SELECTION_NAME = "mvau.weight_supply"

#: The name under which every supplier exports its produced boundary Port.
OUTPUT_PORT_EXPORT = "output_port"


class MVAUWeightSupplyProblemPaths:
    """Problem facts the supply pool reads but does not own."""

    INITIALIZER_AVAILABLE = QualifiedPath("problem.mvau.weight_initializer_available")
    RUNTIME_WRITABLE = QualifiedPath("problem.cyclic_parameter.runtime_writable")
    TARGET_MEMORY_CAPABILITIES = QualifiedPath(
        "problem.target.cyclic_parameter_memory_capabilities"
    )


_BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))
_PORT = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
_REGION = as_object_semantics(DATAFLOW_REGION_SEMANTICS)


def _enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


_RAM_STYLE = _enum_semantics(CyclicRamStyle)
_ORGANIZATION = _enum_semantics(WeightOrganization)

_INITIALIZER = DependencyRef.problem(
    "initializer_available", MVAUWeightSupplyProblemPaths.INITIALIZER_AVAILABLE, _BOOL
)
_RUNTIME_WRITABLE = DependencyRef.problem(
    "runtime_writable", MVAUWeightSupplyProblemPaths.RUNTIME_WRITABLE, _BOOL
)


class SupportsInitializedUram:
    """Target capability fact consumed by the FINN RTL memstream Kernel."""

    __slots__ = ("supports_initialized_uram",)

    def __init__(self, supports_initialized_uram: bool) -> None:
        self.supports_initialized_uram = supports_initialized_uram

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SupportsInitializedUram)
            and other.supports_initialized_uram == self.supports_initialized_uram
        )

    def __hash__(self) -> int:
        return hash(self.supports_initialized_uram)

    def __repr__(self) -> str:
        return f"SupportsInitializedUram({self.supports_initialized_uram!r})"


_TARGET_MEMORY = as_object_semantics(
    ValueSemantics.immutable_nominal(SupportsInitializedUram, name="SupportsInitializedUram")
)
_TARGET_MEMORY_REF = DependencyRef.problem(
    "target_memory_capabilities",
    MVAUWeightSupplyProblemPaths.TARGET_MEMORY_CAPABILITIES,
    _TARGET_MEMORY,
    absence=AbsenceMode.ALLOWS_ABSENT,
)


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


class MVAUWeightSupplyKernelPathSet:
    """Every path one supply Kernel owns locally."""

    def __init__(self, kernel_id: str) -> None:
        self.kernel_id = kernel_id

    def _local(self, suffix: str) -> QualifiedPath:
        return QualifiedPath(f"{MVAU_WEIGHT_SUPPLY_SELECTION_NAME}.{self.kernel_id}.{suffix}")

    def _semantic(self, suffix: str) -> QualifiedPath:
        return QualifiedPath(
            f"semantic.{MVAU_WEIGHT_SUPPLY_SELECTION_NAME}.{self.kernel_id}.{suffix}"
        )

    def constraint(self, suffix: str) -> QualifiedPath:
        return QualifiedPath(
            f"constraint.{MVAU_WEIGHT_SUPPLY_SELECTION_NAME}.{self.kernel_id}.{suffix}"
        )

    @property
    def organization(self) -> QualifiedPath:
        return self._local("organization")

    @property
    def ram_style(self) -> QualifiedPath:
        return self._local("ram_style")

    @property
    def pumped_memory(self) -> QualifiedPath:
        return self._local("pumped_memory")

    @property
    def output_port(self) -> QualifiedPath:
        return self._semantic("output_port")

    @property
    def region(self) -> QualifiedPath:
        return self._semantic("region")


def _output_port_property(
    paths: MVAUWeightSupplyKernelPathSet,
    demand_ref: DependencyRef,
    full_tile_ref: DependencyRef,
) -> DerivedProperty:
    organization_ref = DependencyRef.decision("organization", paths.organization, _ORGANIZATION)

    def derive(dependencies: DependencyView) -> Answer[object]:
        organization = cast(WeightOrganization, dependencies["organization"])
        if organization is WeightOrganization.AS_DEMANDED:
            return Decided(cast(Port, dependencies["demand"]))
        return Decided(cast(Port, dependencies["full_tile"]))

    return DerivedProperty(
        paths.output_port,
        _PORT,
        EvaluatorSpec((organization_ref, demand_ref, full_tile_ref), derive),
    )


def _region_property(paths: MVAUWeightSupplyKernelPathSet) -> DerivedProperty:
    output_ref = DependencyRef.property("output_port", paths.output_port, _PORT)

    def derive(dependencies: DependencyView) -> Answer[object]:
        return Decided(construct_cyclic_parameter_region(cast(Port, dependencies["output_port"])))

    return DerivedProperty(paths.region, _REGION, EvaluatorSpec((output_ref,), derive))


def _local_state_available(*, requires_initializer: bool) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        initialized = cast(bool, dependencies["initializer_available"])
        if requires_initializer:
            return Decided(initialized)
        return Decided(initialized or cast(bool, dependencies["runtime_writable"]))

    return EvaluatorSpec((_INITIALIZER, _RUNTIME_WRITABLE), evaluate)


def _pumping_supported(paths: MVAUWeightSupplyKernelPathSet) -> EvaluatorSpec[Answer[bool]]:
    pumped_ref = DependencyRef.decision("pumped_memory", paths.pumped_memory, _BOOL)
    output_ref = DependencyRef.property("output_port", paths.output_port, _PORT)

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        if not cast(bool, dependencies["pumped_memory"]):
            return Decided(True)
        port = cast(Port, dependencies["output_port"])
        return Decided(port.beat_sequence.elements_per_beat > 1)

    return EvaluatorSpec((pumped_ref, output_ref), evaluate)


def _uram_initialization_supported(
    paths: MVAUWeightSupplyKernelPathSet,
) -> EvaluatorSpec[Answer[bool]]:
    ram_ref = DependencyRef.decision("ram_style", paths.ram_style, _RAM_STYLE)

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        if dependencies["ram_style"] is not CyclicRamStyle.URAM:
            return Decided(True)
        if cast(bool, dependencies["runtime_writable"]):
            return Decided(True)
        target = dependencies["target_memory_capabilities"]
        if target is ABSENT:
            return Unresolved(
                (
                    Finding(
                        FindingKind.LIMITATION,
                        "cyclic-target-memory-capabilities-missing",
                        paths.constraint("uram_initialization_supported"),
                        "initialized URAM requires target memory capabilities",
                        trace=(MVAUWeightSupplyProblemPaths.TARGET_MEMORY_CAPABILITIES,),
                    ),
                )
            )
        return Decided(cast(SupportsInitializedUram, target).supports_initialized_uram)

    return EvaluatorSpec((ram_ref, _RUNTIME_WRITABLE, _TARGET_MEMORY_REF), evaluate)


def build_finn_rtl_memstream_kernel(
    demand_ref: DependencyRef, full_tile_ref: DependencyRef
) -> Kernel:
    """The existing FINN RTL cyclic memory streamer."""

    paths = MVAUWeightSupplyKernelPathSet(MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value)
    spec = DesignSpaceSpec(
        decisions=(
            Decision(paths.organization, _ORGANIZATION, _finite_domain(tuple(WeightOrganization))),
            Decision(paths.ram_style, _RAM_STYLE, _finite_domain(tuple(CyclicRamStyle))),
            Decision(paths.pumped_memory, _BOOL, _finite_domain((False, True))),
        ),
        properties=(
            _output_port_property(paths, demand_ref, full_tile_ref),
            _region_property(paths),
        ),
        constraints=(
            Constraint(
                paths.constraint("local_state_available"),
                _local_state_available(requires_initializer=False),
            ),
            Constraint(
                paths.constraint("uram_initialization_supported"),
                _uram_initialization_supported(paths),
            ),
            Constraint(paths.constraint("pumping_supported"), _pumping_supported(paths)),
        ),
    )
    return Kernel(
        paths.kernel_id,
        "1",
        spec,
        paths.region,
        feasibility_constraints=(
            paths.constraint("local_state_available"),
            paths.constraint("uram_initialization_supported"),
            paths.constraint("pumping_supported"),
        ),
        source_admission_constraints=(paths.constraint("local_state_available"),),
        exports=(KernelExport(OUTPUT_PORT_EXPORT, paths.output_port, _PORT),),
        providers=(KernelProvider("finn.rtl.memstream", paths.kernel_id),),
    )


def build_finnlib_memstream_kernel(
    demand_ref: DependencyRef, full_tile_ref: DependencyRef
) -> Kernel:
    """The FinnLib HLS memory-to-stream mechanism.

    It is a different microarchitecture from the FINN RTL streamer even where
    the two derive an equal Region: it has no RAM-style or memory-pumping
    choice and cannot serve a runtime-written array.
    """

    paths = MVAUWeightSupplyKernelPathSet(MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM.value)
    spec = DesignSpaceSpec(
        decisions=(
            Decision(paths.organization, _ORGANIZATION, _finite_domain(tuple(WeightOrganization))),
        ),
        properties=(
            _output_port_property(paths, demand_ref, full_tile_ref),
            _region_property(paths),
        ),
        constraints=(
            Constraint(
                paths.constraint("local_state_available"),
                _local_state_available(requires_initializer=True),
            ),
        ),
    )
    return Kernel(
        paths.kernel_id,
        "1",
        spec,
        paths.region,
        feasibility_constraints=(paths.constraint("local_state_available"),),
        source_admission_constraints=(paths.constraint("local_state_available"),),
        exports=(KernelExport(OUTPUT_PORT_EXPORT, paths.output_port, _PORT),),
        providers=(KernelProvider("finnlib.hls.memstream", paths.kernel_id),),
    )


def build_mvau_weight_supply_selection(
    demand_ref: DependencyRef,
    full_tile_ref: DependencyRef,
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> KernelSelection:
    """Build the optional weight-supply pool for one compute demand."""

    return KernelSelection(
        MVAU_WEIGHT_SUPPLY_SELECTION_NAME,
        (
            build_finn_rtl_memstream_kernel(demand_ref, full_tile_ref),
            build_finnlib_memstream_kernel(demand_ref, full_tile_ref),
        ),
        optional=True,
        applies_if=applies_if,
    )


FINN_RTL_MEMSTREAM_PATHS = MVAUWeightSupplyKernelPathSet(
    MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
)
FINNLIB_MEMSTREAM_PATHS = MVAUWeightSupplyKernelPathSet(
    MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM.value
)

__all__ = [
    "FINNLIB_MEMSTREAM_PATHS",
    "FINN_RTL_MEMSTREAM_PATHS",
    "MVAU_WEIGHT_SUPPLY_SELECTION_NAME",
    "MVAUWeightSupplyKernelId",
    "MVAUWeightSupplyKernelPathSet",
    "MVAUWeightSupplyProblemPaths",
    "OUTPUT_PORT_EXPORT",
    "CyclicRamStyle",
    "SupportsInitializedUram",
    "WeightOrganization",
    "build_finn_rtl_memstream_kernel",
    "build_finnlib_memstream_kernel",
    "build_mvau_weight_supply_selection",
]
