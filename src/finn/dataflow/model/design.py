# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Design declarations: the Space that owns semantics, segments, and topology.

A ``DataflowDesign`` owns every decision that changes its selected logical
Regions or the Network they form.  It structurally contains named segments of
candidate ``Kernel`` subspaces and explicit topology; each Kernel consumes
semantic facts as typed Inputs and derives exactly one canonical Region, and the
Design places the exact selected Regions into one validated Network.

Declarations and their lowering share this module.  Splitting them would put
the compiler above the vocabulary and the ``_finalize_compilation`` hook below
it, which is a cycle; ``kernel.py`` is arranged the same way for the same
reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Generic, TypeVar, cast

from finn.dataflow._engine import DependencyKind, QualifiedPath
from finn.dataflow.computation import ComputationContract
from finn.dataflow.model.compiler import _CompiledBranch, _CompiledSpace, _Ref
from finn.dataflow.model.declarations import (
    AuthoringError,
    Case,
    OneOf,
    Problem,
    Space,
    ValueSource,
    declared_members,
)
from finn.dataflow.model.kernel import Kernel, _KernelCompilation
from finn.dataflow.network import PositionMap
from finn.dataflow.region import DataflowRegion

D = TypeVar("D", bound="DataflowDesign")


class DataflowDesign(Space):
    """One authored composition of Kernel segments and explicit topology."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        return _finalize_design(cls, compiled)


class Kernels(OneOf):
    """One stable Design segment holding exactly one selected Kernel.

    A thin ``OneOf`` specialization.  Selector creation, case gating, stable
    namespaces, case inspection, and selected-output forwarding all stay in the
    generic branch machinery; what a Kernel segment adds is the required
    computation, the default case id, the implicit selected Region, and the
    stable Design role and Network node identity.
    """

    computation: ComputationContract
    node_id: str | None

    #: A Kernel segment's selector chooses a Kernel, so it says so.
    selector_name: ClassVar[str] = "kernel"

    __slots__ = ("computation", "node_id")

    def __init__(
        self,
        *cases: Case,
        computation: ComputationContract,
        role: str | None = None,
        node_id: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(computation, ComputationContract):
            raise AuthoringError("a Kernel segment declares one ComputationContract")
        if role is not None and not role:
            raise AuthoringError("a Kernel segment role must be non-empty")
        if node_id is not None and not node_id:
            raise AuthoringError("a Kernel segment node id must be non-empty")
        # The role *is* the namespace segment, so it travels as the branch's
        # stable name rather than as a second parallel identity.
        self._initialize(cases, ("region",), when, role)
        object.__setattr__(self, "computation", computation)
        object.__setattr__(self, "node_id", node_id)

    def case_id(self, case: Case) -> str | None:
        """A Kernel case is named by its Kernel id unless the author overrides it."""

        if case.stable_name is not None:
            return case.stable_name
        return cast(str, getattr(case.space_type, "id", "")) or None

    def check_case(self, owner_name: str, member_name: str, case: Case) -> None:
        if not issubclass(case.space_type, Kernel):
            raise AuthoringError(
                f"{owner_name}.{member_name} is a Kernel segment, but "
                f"{case.space_type.__name__} is not a Kernel"
            )
        offered = getattr(case.space_type, "computation", None)
        if offered != self.computation:
            raise AuthoringError(
                f"{owner_name}.{member_name} requires computation "
                f"{self.computation.id}:{self.computation.version}, but "
                f"{case.space_type.__name__} declares "
                f"{getattr(offered, 'id', offered)!r}"
            )

    @property
    def region(self) -> ValueSource[object]:
        """The candidate-independent selected Region of this segment."""

        return self.__getattr__("region")

    def input(self, port_id: str) -> SegmentEndpoint:
        """An immutable claim that the selected Region has this input port."""

        return SegmentEndpoint(self, port_id, output=False)

    def output(self, port_id: str) -> SegmentEndpoint:
        """An immutable claim that the selected Region has this output port."""

        return SegmentEndpoint(self, port_id, output=True)


@dataclass(frozen=True, slots=True, eq=False)
class SegmentEndpoint:
    """A segment, a stable port id, and an expected direction.

    Deliberately not a resolved model endpoint and not an engine ref.  The
    Region owns the real port list, so this claim is checked against the
    selected Region by Network construction and canonical validation rather than
    maintained as a second port schema that can drift.
    """

    segment: Kernels
    port_id: str
    output: bool

    def __post_init__(self) -> None:
        if not self.port_id:
            raise AuthoringError("an endpoint needs a non-empty port id")


@dataclass(frozen=True, slots=True, eq=False)
class Sink:
    """One consuming endpoint of a Connection and its source-to-sink map.

    Per-sink rather than per-edge because a canonical fan-out is one Edge with
    several sink contracts, and each contract owns its own position map.  An
    omitted map means the identity over the selected source port's image.
    """

    endpoint: SegmentEndpoint
    position_map: ValueSource[PositionMap] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, SegmentEndpoint):
            raise AuthoringError("a Sink consumes a segment endpoint")
        if self.endpoint.output:
            raise AuthoringError(
                f"a Sink must name an input port; {self.endpoint.port_id!r} is declared as output"
            )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Connection:
    """One canonical Edge: one source endpoint replicated to ordered sinks."""

    source: SegmentEndpoint
    sinks: tuple[Sink, ...]
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        source: SegmentEndpoint,
        *sinks: Sink,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(source, SegmentEndpoint):
            raise AuthoringError("a Connection starts at a segment endpoint")
        if not source.output:
            raise AuthoringError(
                f"a Connection source must name an output port; "
                f"{source.port_id!r} is declared as input"
            )
        if not sinks:
            raise AuthoringError("a Connection needs at least one Sink")
        if any(not isinstance(sink, Sink) for sink in sinks):
            raise AuthoringError("a Connection takes Sink declarations after its source")
        if name is not None and not name:
            raise AuthoringError("a Connection name must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sinks", tuple(sinks))
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(self, "when", when)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Boundary:
    """One externally visible endpoint of the selected Network."""

    endpoint: SegmentEndpoint
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        endpoint: SegmentEndpoint,
        *,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(endpoint, SegmentEndpoint):
            raise AuthoringError("a Boundary exposes a segment endpoint")
        if name is not None and not name:
            raise AuthoringError("a Boundary name must be non-empty")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(self, "when", when)


TopologyDeclaration = Connection | Boundary

#: Class-body values the Design compiler recognizes beyond the generic set.
TOPOLOGY_TYPES: tuple[type, ...] = (Connection, Boundary)


def topology_members(
    design_type: type[DataflowDesign],
) -> tuple[tuple[str, TopologyDeclaration], ...]:
    """Collect Connection and Boundary members in deterministic inherited order."""

    ordered: dict[str, TopologyDeclaration] = {}
    for base in reversed(design_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, TOPOLOGY_TYPES):
                ordered[name] = cast(TopologyDeclaration, value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces a topology declaration with "
                    f"{type(value).__name__}"
                )
    return tuple(ordered.items())


# -- lowering -----------------------------------------------------------------


@dataclass(frozen=True)
class _CompiledKernelCase:
    """One candidate Kernel compiled beneath a segment namespace."""

    kernel_id: str
    compiled: _CompiledSpace[Kernel]

    @property
    def metadata(self) -> _KernelCompilation[Kernel]:
        return cast("_KernelCompilation[Kernel]", self.compiled.extension)


@dataclass(frozen=True)
class _CompiledKernelSegment:
    """One stable Design role and the Kernel branch that fills it."""

    member_name: str
    role: str
    node_id: str
    required_computation: ComputationContract
    active: _Ref[object] | None
    selector: _Ref[object] | None
    cases: tuple[_CompiledKernelCase, ...]
    selected_region: _Ref[DataflowRegion]

    def case(self, kernel_id: str) -> _CompiledKernelCase:
        for candidate in self.cases:
            if candidate.kernel_id == kernel_id:
                return candidate
        raise AuthoringError(f"segment {self.role!r} has no candidate {kernel_id!r}")


@dataclass(frozen=True)
class _DesignCompilation(Generic[D]):
    """Design-only metadata attached to a generic compiled Space."""

    owner: type[D]
    design_id: str
    design_version: str
    segments: tuple[_CompiledKernelSegment, ...]

    def segment(self, role: str) -> _CompiledKernelSegment:
        for candidate in self.segments:
            if candidate.role == role:
                return candidate
        raise AuthoringError(f"this Design has no segment {role!r}")


def _segment(
    design_type: type[DataflowDesign],
    member_name: str,
    declaration: Kernels,
    branch: _CompiledBranch,
) -> _CompiledKernelSegment:
    role = branch.namespace.rsplit(".", 1)[-1]
    cases = []
    for case in branch.cases:
        compiled = cast("_CompiledSpace[Kernel]", case.compiled)
        if not isinstance(compiled.extension, _KernelCompilation):
            raise AuthoringError(
                f"{design_type.__name__}.{member_name} candidate "
                f"{compiled.owner.__name__} did not produce Kernel metadata"
            )
        cases.append(_CompiledKernelCase(case.case_id, compiled))
    return _CompiledKernelSegment(
        member_name,
        role,
        declaration.node_id or role,
        declaration.computation,
        branch.active,
        branch.selector,
        tuple(cases),
        cast("_Ref[DataflowRegion]", branch.output("region")),
    )


def _finalize_design(design_type: type[D], compiled: object) -> object:
    """Validate one Design and attach its private compilation record."""

    if not isinstance(compiled, _CompiledSpace):
        raise AuthoringError(f"{design_type.__name__} received an invalid Space compilation")
    design = cast("_CompiledSpace[D]", compiled)

    if not design_type.id:
        raise AuthoringError(f"{design_type.__name__} must declare a non-empty id")
    if not design_type.version:
        raise AuthoringError(f"{design_type.__name__} must declare a non-empty version")

    declarations = dict(declared_members(design_type))
    problem_members = tuple(
        name for name, declaration in declarations.items() if isinstance(declaration, Problem)
    )
    if problem_members:
        raise AuthoringError(
            f"{design_type.__name__} must consume external facts through Input; "
            f"Design-owned Problem members are {problem_members}"
        )

    segments = tuple(
        _segment(design_type, name, declaration, design.branch(name))
        for name, declaration in declarations.items()
        if isinstance(declaration, Kernels)
    )
    if not segments:
        raise AuthoringError(f"{design_type.__name__} must declare at least one Kernel segment")
    _check_unique(design_type, segments)

    metadata: _DesignCompilation[D] = _DesignCompilation(
        design_type,
        design_type.id,
        design_type.version,
        segments,
    )
    return replace(design, extension=metadata)


def _check_unique(
    design_type: type[DataflowDesign], segments: tuple[_CompiledKernelSegment, ...]
) -> None:
    for label, values in (
        ("role", tuple(item.role for item in segments)),
        ("node id", tuple(item.node_id for item in segments)),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise AuthoringError(
                f"{design_type.__name__} declares segment {label} {duplicates[0]!r} twice"
            )


def _region_path(segment: _CompiledKernelSegment) -> QualifiedPath:
    reference = segment.selected_region
    if reference.kind is not DependencyKind.PROPERTY:
        raise AuthoringError(f"segment {segment.role!r} did not select a Region property")
    return reference.path


__all__ = [
    "TOPOLOGY_TYPES",
    "Boundary",
    "Connection",
    "DataflowDesign",
    "Kernels",
    "SegmentEndpoint",
    "Sink",
    "TopologyDeclaration",
    "topology_members",
]
