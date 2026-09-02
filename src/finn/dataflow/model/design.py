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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar, Generic, TypeVar, cast

from finn.dataflow._engine import (
    ABSENT,
    AbsenceMode,
    Absent,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignPoint,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
)
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design.region import DATAFLOW_NETWORK_SEMANTICS
from finn.dataflow.model.compiler import (
    _CompiledBranch,
    _CompiledSpace,
    _Ref,
    answer_for,
    imported_decisions,
)
from finn.dataflow.model.declarations import (
    AuthoringError,
    Case,
    ChildValue,
    Decision,
    OneOf,
    Problem,
    Space,
    Use,
    ValueSource,
    declared_members,
    semantics_for,
)
from finn.dataflow.model.kernel import Kernel, _KernelCompilation, configure_kernel
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import validate_network
from finn.dataflow.region import BeatSequence, DataflowRegion, Port

D = TypeVar("D", bound="DataflowDesign")


class DataflowDesign(Space):
    """One authored composition of Kernel segments and explicit topology."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        return _finalize_design(cls, compiled)

    def _initialize(
        self,
        compilation: _DesignCompilation[DataflowDesign],
        network: DataflowNetwork,
        kernels: Mapping[str, Kernel],
        selected: Mapping[str, str],
        values: Mapping[int, object],
        assignments: Mapping[QualifiedPath, object],
        imported: Sequence[QualifiedPath],
    ) -> None:
        self._compilation = compilation
        self._values = MappingProxyType(dict(values))
        self.resolved_network = network
        self.kernels = MappingProxyType(dict(kernels))
        self.selected_candidates = MappingProxyType(dict(selected))
        self.assignments = MappingProxyType(dict(assignments))
        self.imported_decisions = tuple(imported)

    def _space_value(self, declaration: ValueSource[object]) -> object:
        try:
            return self._values[id(declaration)]
        except KeyError:
            raise AttributeError(
                "this declaration is not retained on the configured Design; "
                "only Design-owned decisions and selectors are"
            ) from None

    def _design_kernel(self, declaration: Kernels) -> Kernel:
        for segment in self._compilation.segments:
            if segment.declaration is declaration:
                try:
                    return self.kernels[segment.role]
                except KeyError:
                    raise AttributeError(
                        f"segment {segment.role!r} is not active in this configuration"
                    ) from None
        raise AttributeError("this segment does not belong to the configured Design")

    def node_id(self, role: str) -> str:
        """The stable Network node this Design role occupies."""

        return self._compilation.segment(role).node_id

    def region_family(self, role: str) -> tuple[str, str]:
        """The selected Region's semantic family and version at one role."""

        segment = self._compilation.segment(role)
        case = segment.case(self.selected_candidates[role])
        return case.metadata.region_family, case.metadata.region_version


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

    def __get__(self, instance: object | None, owner: type[object]) -> object:
        if instance is None:
            return self
        resolver = getattr(instance, "_design_kernel", None)
        if resolver is None:
            raise AttributeError("a selected Kernel exists only on a configured Design")
        return resolver(self)

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
    declaration: Kernels
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
class _CompiledEndpoint:
    """One resolved segment endpoint claim: node, port, expected direction."""

    role: str
    node_id: str
    port_id: str
    output: bool

    @property
    def endpoint(self) -> RegionEndpoint:
        return RegionEndpoint(self.node_id, self.port_id)


@dataclass(frozen=True)
class _CompiledSink:
    endpoint: _CompiledEndpoint
    position_map: _Ref[object] | None


@dataclass(frozen=True)
class _CompiledConnection:
    edge_id: str
    source: _CompiledEndpoint
    sinks: tuple[_CompiledSink, ...]
    active: _Ref[object] | None


@dataclass(frozen=True)
class _CompiledBoundary:
    boundary_id: str
    endpoint: _CompiledEndpoint
    active: _Ref[object] | None


@dataclass(frozen=True)
class _DesignCompilation(Generic[D]):
    """Design-only metadata attached to a generic compiled Space."""

    owner: type[D]
    design_id: str
    design_version: str
    segments: tuple[_CompiledKernelSegment, ...]
    connections: tuple[_CompiledConnection, ...]
    boundaries: tuple[_CompiledBoundary, ...]
    network: _Ref[DataflowNetwork]
    feasibility_set: str
    readiness_profile: str
    local_decisions: tuple[tuple[str, _Ref[object]], ...]

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
        declaration,
        role,
        declaration.node_id or role,
        declaration.computation,
        branch.active,
        branch.selector,
        tuple(cases),
        cast("_Ref[DataflowRegion]", branch.output("region")),
    )


def _value_ref(
    design_type: type[DataflowDesign],
    compiled: _CompiledSpace[D],
    source: ValueSource[object],
    what: str,
) -> _Ref[object]:
    """Resolve one Design-visible value declaration to its compiled handle."""

    if isinstance(source, ChildValue):
        names = {
            id(value): name
            for base in reversed(design_type.__mro__)
            if issubclass(base, Space) and base is not Space
            for name, value in base.__dict__.items()
            if isinstance(value, Use)
        }
        child = names.get(id(source.use))
        if child is None:
            raise AuthoringError(f"{design_type.__name__} {what} names a child outside the class")
        return compiled.child(child).exported(source.member_name)
    names = {
        id(value): name
        for base in reversed(design_type.__mro__)
        if issubclass(base, Space) and base is not Space
        for name, value in base.__dict__.items()
        if isinstance(value, ValueSource)
    }
    name = names.get(id(source))
    if name is None:
        raise AuthoringError(f"{design_type.__name__} {what} names a value outside the class")
    return compiled.member(name)


def _endpoint(
    design_type: type[DataflowDesign],
    by_declaration: Mapping[int, _CompiledKernelSegment],
    endpoint: SegmentEndpoint,
    what: str,
) -> _CompiledEndpoint:
    segment = by_declaration.get(id(endpoint.segment))
    if segment is None:
        raise AuthoringError(
            f"{design_type.__name__} {what} names a Kernel segment outside the class"
        )
    return _CompiledEndpoint(segment.role, segment.node_id, endpoint.port_id, endpoint.output)


def _boolean_ref(
    design_type: type[DataflowDesign],
    compiled: _CompiledSpace[D],
    when: ValueSource[bool] | None,
    what: str,
) -> _Ref[object] | None:
    if when is None:
        return None
    reference = _value_ref(design_type, compiled, cast("ValueSource[object]", when), what)
    if reference.semantics.type_token is not bool:
        raise AuthoringError(f"{design_type.__name__} {what} when= is not Boolean")
    return reference


def _topology(
    design_type: type[D],
    compiled: _CompiledSpace[D],
    segments: tuple[_CompiledKernelSegment, ...],
) -> tuple[tuple[_CompiledConnection, ...], tuple[_CompiledBoundary, ...]]:
    by_declaration = {
        id(declaration): segment
        for segment, declaration in zip(
            segments,
            (value for _name, value in declared_members(design_type) if isinstance(value, Kernels)),
        )
    }
    connections: list[_CompiledConnection] = []
    boundaries: list[_CompiledBoundary] = []
    for member_name, declaration in topology_members(design_type):
        identity = declaration.stable_name or member_name
        what = f"topology {identity!r}"
        active = _boolean_ref(design_type, compiled, declaration.when, what)
        if isinstance(declaration, Connection):
            connections.append(
                _CompiledConnection(
                    identity,
                    _endpoint(design_type, by_declaration, declaration.source, what),
                    tuple(
                        _CompiledSink(
                            _endpoint(design_type, by_declaration, sink.endpoint, what),
                            None
                            if sink.position_map is None
                            else _value_ref(
                                design_type,
                                compiled,
                                cast("ValueSource[object]", sink.position_map),
                                what,
                            ),
                        )
                        for sink in declaration.sinks
                    ),
                    active,
                )
            )
        else:
            boundaries.append(
                _CompiledBoundary(
                    identity,
                    _endpoint(design_type, by_declaration, declaration.endpoint, what),
                    active,
                )
            )
    for label, values in (
        ("edge id", tuple(item.edge_id for item in connections)),
        ("boundary id", tuple(item.boundary_id for item in boundaries)),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise AuthoringError(f"{design_type.__name__} declares {label} {duplicates[0]!r} twice")
    return tuple(connections), tuple(boundaries)


def _active(value: object) -> bool:
    """An absent condition is not a true one: the branch simply is not there."""

    return value is not ABSENT and bool(value)


_EMPTY_SEQUENCE = BeatSequence(0, ())


def _source_port(regions: Mapping[str, DataflowRegion], endpoint: _CompiledEndpoint) -> Port | None:
    region = regions.get(endpoint.role)
    if region is None:
        return None
    try:
        return (
            region.output_interface(endpoint.port_id).port
            if endpoint.output
            else region.input_interface(endpoint.port_id).port
        )
    except KeyError:
        return None


def _network_property(
    path: QualifiedPath,
    segments: tuple[_CompiledKernelSegment, ...],
    connections: tuple[_CompiledConnection, ...],
    boundaries: tuple[_CompiledBoundary, ...],
) -> DerivedProperty:
    """One ordinary property holding the Network the selected Regions form.

    The evaluator receives values only: segment activity, the exact selected
    Regions, topology conditions, and explicit position maps.  It never sees a
    Kernel object, an Engine, a point, or a compiler record.

    Where a claim cannot be honoured -- a port that the selected Region does not
    have, an edge into an inactive segment -- it builds the node-qualified
    endpoint anyway and lets canonical ``validate_network`` name the failure, so
    the diagnostics stay the canon's rather than a Design-specific paraphrase.
    """

    dependencies: list[DependencyRef] = []
    for segment in segments:
        dependencies.append(
            replace(segment.selected_region, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                f"region@{segment.role}"
            )
        )
        if segment.active is not None:
            dependencies.append(
                replace(segment.active, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                    f"segment_active@{segment.role}"
                )
            )
    for connection in connections:
        if connection.active is not None:
            dependencies.append(
                replace(connection.active, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                    f"edge_active@{connection.edge_id}"
                )
            )
        for index, sink in enumerate(connection.sinks):
            if sink.position_map is not None:
                dependencies.append(
                    replace(sink.position_map, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                        f"map@{connection.edge_id}#{index}"
                    )
                )
    for boundary in boundaries:
        if boundary.active is not None:
            dependencies.append(
                replace(boundary.active, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                    f"boundary_active@{boundary.boundary_id}"
                )
            )

    def build(values: DependencyView) -> Answer[object]:
        regions: dict[str, DataflowRegion] = {}
        nodes: list[NetworkNode] = []
        for segment in segments:
            if segment.active is not None and not _active(values[f"segment_active@{segment.role}"]):
                continue
            region = values[f"region@{segment.role}"]
            if region is ABSENT:
                return Absent(
                    (
                        Finding(
                            FindingKind.REJECTION,
                            "design-active-segment-without-region",
                            path,
                            "an active segment did not resolve a Region",
                            (("role", segment.role),),
                        ),
                    )
                )
            regions[segment.role] = cast(DataflowRegion, region)
            nodes.append(NetworkNode(segment.node_id, cast(DataflowRegion, region)))

        edges: list[Edge] = []
        for connection in connections:
            if connection.active is not None and not _active(
                values[f"edge_active@{connection.edge_id}"]
            ):
                continue
            source_port = _source_port(regions, connection.source)
            identity = (
                PositionMap(())
                if source_port is None
                else PositionMap.identity(source_port.beat_sequence.image)
            )
            sinks: list[SinkContract] = []
            for index, sink in enumerate(connection.sinks):
                declared = (
                    None
                    if sink.position_map is None
                    else values[f"map@{connection.edge_id}#{index}"]
                )
                if declared is ABSENT:
                    return Absent(
                        (
                            Finding(
                                FindingKind.REJECTION,
                                "design-position-map-absent",
                                path,
                                "an active sink did not resolve its position map",
                                (("edge", connection.edge_id),),
                            ),
                        )
                    )
                sinks.append(
                    SinkContract(
                        sink.endpoint.endpoint,
                        identity if declared is None else cast(PositionMap, declared),
                    )
                )
            edges.append(Edge(connection.edge_id, connection.source.endpoint, tuple(sinks)))

        contracts: list[BoundaryContract] = []
        for boundary in boundaries:
            if boundary.active is not None and not _active(
                values[f"boundary_active@{boundary.boundary_id}"]
            ):
                continue
            port = _source_port(regions, boundary.endpoint)
            contracts.append(
                BoundaryContract(
                    boundary.boundary_id,
                    boundary.endpoint.endpoint,
                    _EMPTY_SEQUENCE if port is None else port.beat_sequence,
                )
            )
        return Decided(DataflowNetwork(tuple(nodes), tuple(edges), tuple(contracts)))

    return DerivedProperty(
        path,
        semantics_for(DATAFLOW_NETWORK_SEMANTICS),
        EvaluatorSpec(tuple(dependencies), build),
    )


def _network_valid_constraint(path: QualifiedPath, network: _Ref[DataflowNetwork]) -> Constraint:
    """Adapt every canonical network issue, preserving its code."""

    def evaluate(values: DependencyView) -> Answer[bool]:
        report = validate_network(cast(DataflowNetwork, values["network"]))
        if not report.issues:
            return Decided(True)
        return Absent(
            tuple(
                Finding(
                    FindingKind.REJECTION,
                    f"design-network-{issue.code}",
                    path,
                    issue.message,
                    (("network_path", issue.path),),
                    (network.path,),
                )
                for issue in report.issues
            )
        )

    return Constraint(path, EvaluatorSpec((network.dependency("network"),), evaluate))


def _correspondence_constraint(
    path: QualifiedPath,
    network: _Ref[DataflowNetwork],
    segments: tuple[_CompiledKernelSegment, ...],
) -> Constraint:
    """The one Design-specific supplement to canonical Network validation.

    Canonical validation already proves every semantic topology claim.  What it
    cannot know is that these nodes are exactly this Design's active segments and
    hold exactly their selected Regions -- which is the whole relationship a
    coverage object would otherwise have to carry.
    """

    dependencies = [network.dependency("network")]
    for segment in segments:
        dependencies.append(
            replace(segment.selected_region, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                f"region@{segment.role}"
            )
        )
        if segment.active is not None:
            dependencies.append(
                replace(segment.active, absence=AbsenceMode.ALLOWS_ABSENT).dependency(
                    f"segment_active@{segment.role}"
                )
            )

    def evaluate(values: DependencyView) -> Answer[bool]:
        resolved = cast(DataflowNetwork, values["network"])
        expected: dict[str, DataflowRegion] = {}
        for segment in segments:
            if segment.active is not None and not _active(values[f"segment_active@{segment.role}"]):
                continue
            region = values[f"region@{segment.role}"]
            if region is not ABSENT:
                expected[segment.node_id] = cast(DataflowRegion, region)
        findings: list[Finding] = []
        if not expected:
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "design-no-active-segment",
                    path,
                    "a Design point must leave at least one segment active",
                )
            )
        actual = {node.id: node.region for node in resolved.nodes}
        for node_id in sorted(set(expected) ^ set(actual)):
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "design-segment-node-mismatch",
                    path,
                    "active segments and Network nodes must correspond exactly",
                    (("node", node_id),),
                )
            )
        for node_id in sorted(set(expected) & set(actual)):
            if expected[node_id] != actual[node_id]:
                findings.append(
                    Finding(
                        FindingKind.REJECTION,
                        "design-node-region-mismatch",
                        path,
                        "a Network node does not hold its segment's selected Region",
                        (("node", node_id),),
                    )
                )
        return Decided(True) if not findings else Absent(tuple(findings))

    return Constraint(path, EvaluatorSpec(tuple(dependencies), evaluate))


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

    connections, boundaries = _topology(design_type, design, segments)
    network_path = QualifiedPath(f"semantic.{design.namespace}.network")
    network_property = _network_property(network_path, segments, connections, boundaries)
    network: _Ref[DataflowNetwork] = _Ref(
        network_path, DependencyKind.PROPERTY, network_property.value_semantics
    )
    generated = (
        _network_valid_constraint(
            QualifiedPath(f"constraint.{design.namespace}.network_structurally_valid"),
            network,
        ),
        _correspondence_constraint(
            QualifiedPath(f"constraint.{design.namespace}.segments_match_network"),
            network,
            segments,
        ),
    )
    constraints = (*design.spec.constraints, *generated)
    constraint_paths = tuple(item.path for item in constraints)
    feasibility_name = f"{design.namespace}.feasibility"
    readiness_name = f"{design.namespace}.configured"
    local_decisions = tuple(
        (name, design.member(name))
        for name, declaration in declarations.items()
        if isinstance(declaration, Decision)
    ) + tuple(
        (segment.role, segment.selector) for segment in segments if segment.selector is not None
    )
    specification = replace(
        design.spec,
        properties=(*design.spec.properties, network_property),
        constraints=constraints,
        constraint_sets=(
            *design.spec.constraint_sets,
            ConstraintSet(feasibility_name, constraint_paths),
        ),
        readiness_profiles=(
            *design.spec.readiness_profiles,
            ReadinessProfile(
                readiness_name,
                tuple(item.path for item in design.spec.decisions),
                _readiness_properties(segments, network),
                constraint_paths,
            ),
        ),
    )

    metadata: _DesignCompilation[D] = _DesignCompilation(
        design_type,
        design_type.id,
        design_type.version,
        segments,
        connections,
        boundaries,
        network,
        feasibility_name,
        readiness_name,
        local_decisions,
    )
    return replace(design, spec=specification, extension=metadata)


def _readiness_properties(
    segments: tuple[_CompiledKernelSegment, ...],
    network: _Ref[DataflowNetwork],
) -> tuple[QualifiedPath, ...]:
    """Selected Regions, the Network, and each candidate's own physical values.

    Taking the candidates' ``configured`` profiles rather than every property in
    the flat spec keeps readiness to the values a configured Design will actually
    retain; an inactive candidate's obligations reduce to final absence anyway.
    """

    paths = [network.path, *(segment.selected_region.path for segment in segments)]
    for segment in segments:
        for case in segment.cases:
            wanted = case.metadata.readiness_profile
            for profile in case.compiled.spec.readiness_profiles:
                if profile.name == wanted:
                    paths.extend(profile.properties)
    return tuple(dict.fromkeys(paths))


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


# -- configuration ------------------------------------------------------------


def _blocked(namespace: str, code: str, message: str) -> Unresolved:
    return Unresolved((Finding(FindingKind.BLOCKER, code, QualifiedPath(namespace), message),))


def _selected_case(
    engine: Engine,
    point: DesignPoint,
    segment: _CompiledKernelSegment,
) -> Answer[str]:
    if segment.selector is None:
        return Decided(segment.cases[0].kernel_id)
    if segment.selector.path not in point.assignments:
        return Unresolved(
            (
                Finding(
                    FindingKind.BLOCKER,
                    "design-selector-uncommitted",
                    segment.selector.path,
                    "a Design segment selector is not committed",
                ),
            )
        )
    return Decided(cast(str, point.assignments[segment.selector.path]))


def _segment_is_active(
    engine: Engine,
    point: DesignPoint,
    segment: _CompiledKernelSegment,
) -> Answer[bool]:
    if segment.active is None:
        return Decided(True)
    answer = answer_for(engine, point, segment.active)
    if isinstance(answer, Decided):
        return Decided(bool(answer.value))
    if isinstance(answer, Absent):
        # An absent condition is not a true one: the segment simply is not there.
        return Decided(False)
    return Unresolved(answer.findings)


def configure_design(
    engine: Engine,
    compiled: _CompiledSpace[D],
    point: DesignPoint,
) -> Answer[D]:
    """Resolve one selected Design, then detach it from the engine and point."""

    metadata = compiled.extension
    if not isinstance(metadata, _DesignCompilation):
        raise AuthoringError(f"{compiled.owner.__name__} is not a compiled DataflowDesign")
    design = cast("_DesignCompilation[D]", metadata)

    readiness = engine.check_readiness(point, design.readiness_profile)
    if readiness.ready is not True:
        findings = tuple(
            finding
            for answer in readiness.answers.values()
            if isinstance(answer, Unresolved)
            for finding in answer.findings
        )
        return cast(
            "Answer[D]",
            Unresolved(findings)
            if findings
            else _blocked(
                compiled.namespace,
                "design-not-ready",
                f"{design.design_id} is not ready to configure",
            ),
        )

    assessment = engine.evaluate_constraint_set(point, design.feasibility_set)
    if assessment.verdict is not True:
        findings = tuple(
            finding
            for answer in assessment.answers.values()
            if isinstance(answer, (Absent, Unresolved))
            for finding in answer.findings
        )
        return cast(
            "Answer[D]",
            Unresolved(findings)
            if findings
            else _blocked(
                compiled.namespace,
                "design-infeasible",
                f"{design.design_id} does not cover this configuration",
            ),
        )

    kernels: dict[str, Kernel] = {}
    selected: dict[str, str] = {}
    for segment in design.segments:
        active = _segment_is_active(engine, point, segment)
        if not isinstance(active, Decided):
            return cast("Answer[D]", active)
        if not active.value:
            continue
        chosen = _selected_case(engine, point, segment)
        if not isinstance(chosen, Decided):
            return cast("Answer[D]", chosen)
        case = segment.case(chosen.value)
        configured = configure_kernel(engine, case.compiled, point)
        if not isinstance(configured, Decided):
            return cast("Answer[D]", configured)
        kernels[segment.role] = configured.value
        selected[segment.role] = chosen.value

    resolved = engine.query_property(point, design.network.path)
    if not isinstance(resolved, Decided):
        return cast("Answer[D]", Unresolved(resolved.findings))
    network = cast(DataflowNetwork, resolved.value)

    mismatch = _correspondence_findings(compiled.namespace, design, network, kernels, selected)
    if mismatch:
        return cast("Answer[D]", Unresolved(mismatch))

    assignments = {
        reference.path: point.assignments[reference.path]
        for _name, reference in design.local_decisions
        if reference.path in point.assignments
    }
    retained = {
        id(declaration): point.assignments[compiled.member(name).path]
        for name, declaration in declared_members(design.owner)
        if isinstance(declaration, Decision) and compiled.member(name).path in point.assignments
    }
    instance = object.__new__(design.owner)
    DataflowDesign._initialize(
        instance,
        cast("_DesignCompilation[DataflowDesign]", design),
        network,
        kernels,
        selected,
        retained,
        assignments,
        imported_decisions(
            point,
            compiled.spec,
            compiled.inputs,
            {reference.path for _name, reference in design.local_decisions},
        ),
    )
    return Decided(instance)


def _correspondence_findings(
    namespace: str,
    design: _DesignCompilation[D],
    network: DataflowNetwork,
    kernels: Mapping[str, Kernel],
    selected: Mapping[str, str],
) -> tuple[Finding, ...]:
    """Exact role, node, Region, and configured-Kernel correspondence.

    The generated constraint already proved node-to-Region correspondence over
    values.  This proves the extra thing configuration introduces: the object
    now sitting at each node promises exactly that node's Region.
    """

    findings: list[Finding] = []
    nodes = {node.id: node.region for node in network.nodes}
    expected = {design.segment(role).node_id for role in kernels}
    if expected != set(nodes):
        findings.append(
            Finding(
                FindingKind.BLOCKER,
                "design-configured-node-mismatch",
                QualifiedPath(namespace),
                "configured segments and Network nodes must correspond exactly",
            )
        )
    for role, kernel in kernels.items():
        segment = design.segment(role)
        node_region = nodes.get(segment.node_id)
        if node_region is None or kernel.resolved_region != node_region:
            findings.append(
                Finding(
                    FindingKind.BLOCKER,
                    "design-configured-region-mismatch",
                    QualifiedPath(namespace),
                    "a configured Kernel does not realize its Network node's Region",
                    (("role", role), ("node", segment.node_id)),
                )
            )
        if type(kernel).id != selected[role]:
            findings.append(
                Finding(
                    FindingKind.BLOCKER,
                    "design-configured-candidate-mismatch",
                    QualifiedPath(namespace),
                    "a configured Kernel is not the selected candidate",
                    (("role", role), ("selected", selected[role])),
                )
            )
    return tuple(findings)


__all__ = [
    "TOPOLOGY_TYPES",
    "Boundary",
    "Connection",
    "DataflowDesign",
    "Kernels",
    "SegmentEndpoint",
    "Sink",
    "TopologyDeclaration",
    "configure_design",
    "topology_members",
]
