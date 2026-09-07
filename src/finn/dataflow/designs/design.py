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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from inspect import Parameter as _SignatureParameter, Signature
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from finn.dataflow._engine import (
    ABSENT,
    AbsenceMode,
    Absent,
    Answer,
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
    Unresolved,
)
from finn.dataflow.computation import ComputationContract
from finn.dataflow.space.dataflow_value_semantics import DATAFLOW_NETWORK_SEMANTICS
from finn.dataflow.space.compiler import (
    _CompiledBranch,
    _CompiledSpace,
    _Ref,
    answer_for,
    imported_decisions,
    resolve_value_source,
)
from finn.dataflow.space.declarations import (
    AuthoringError,
    ConstraintGroup,
    Constraint as DeclaredConstraint,
    Derived,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    ValueSource,
    SubspaceChoice,
    _declaration_name,
    allow_absent,
    declared_members,
    reject,
    reject_all,
    semantics_for,
)
from finn.dataflow.kernels.kernel import (
    Kernel,
    _KernelCompilation,
)
from finn.dataflow.space.occurrence import (
    ProjectionAssessment,
    ChoiceView,
    evaluate_projection,
    layer_runtime,
    occurrence_choice,
)
from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.region import BeatSequence, DataflowRegion, Port

D = TypeVar("D", bound="DataflowDesign")


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class SelectedNetwork(Derived[DataflowNetwork]):
    """The Network the selected Regions form, as an ordinary declared member.

    A placeholder at declaration time and a real property after lowering.  The
    evaluator cannot be written in a class body: it needs the compiled position
    maps, whose gating is a lowering concern, and it needs each segment's
    resolved node identity.  What *can* be fixed in the class body is the thing
    that matters -- that a Design's Network is a declaration like any other,
    with a stable member name, ordinary value semantics, and an export a
    containing Space can name.

    That is the difference from the string-keyed generated projection it
    replaces.  A Network was previously reachable only by asking the occurrence
    layer for a projection called ``"dataflow"``, so an operation could not
    depend on a Design's Network the way a Design depends on a Kernel's Region.
    Now it can.
    """

    def __init__(self, *, name: str | None = None) -> None:
        object.__setattr__(self, "value_semantics", semantics_for(DATAFLOW_NETWORK_SEMANTICS))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a SelectedNetwork"))
        object.__setattr__(self, "dependencies", ())
        object.__setattr__(self, "evaluate", _unbuilt_network)


def _unbuilt_network() -> object:
    """Reached only if lowering failed to substitute the real evaluator.

    Not a point state, so not an ``unresolved()``: every compiled Design
    replaces this, and a Design that did not is a defect in this module rather
    than a design space missing information.
    """

    raise AuthoringError(
        "this Design's network member was never lowered; _finalize_design must substitute "
        "the evaluator built from its segments and topology"
    )


#: The member names ``DataflowDesign.__init_subclass__`` writes onto every
#: concrete subclass.  A declaration under one of these would be silently
#: replaced, so it is refused in the class body instead -- the same rule and the
#: same reason as ``RESERVED_KERNEL_NAMES``.  ``network`` is in the set because
#: the base declares it and its evaluator is lowering's to supply.
RESERVED_DESIGN_NAMES: frozenset[str] = frozenset(
    {
        "network",
        "network_structurally_valid",
        "segments_match_network",
        "dataflow_accepts",
        "dataflow_ready",
        "dataflow",
    }
)


class DataflowDesign(Space):
    """One authored composition of Kernel segments and explicit topology.

    The Design is asked for its Network the same way a Kernel is asked for its
    Region -- ``design.dataflow`` -- and there is no resolved-Design wrapper in
    between.  Everything a caller used to read off that wrapper is a question
    put to the occurrence itself, so a partially specialized Design answers what
    it can instead of refusing to exist.

    The Network resolves from semantic facts alone.  No candidate's physical
    projection is consulted, no ABI or source is required, and a Kernel whose
    build unit is explicitly unavailable still contributes its Region.  That is
    the U3 claim and ``dataflow`` is where it is checkable.

    Like ``Kernel``, the projection is synthesized per concrete subclass from
    the parts the author wrote -- the segments, the topology, and at most one
    ``dataflow_support`` group.  What the author does not write is the Network
    itself, its two structural constraints, or the readiness profile: each has
    exactly one correct form, and a Design that had to assemble them by hand
    would be a Design that could assemble them wrongly.
    """

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"

    #: The Network is the one automatic Design output to a containing Space.
    #: This is what lets an operation compose a Design exactly as a Design
    #: composes a Kernel.
    network = SelectedNetwork()
    _implicit_exports = ("network",)

    if TYPE_CHECKING:
        # Written onto every concrete subclass by ``__init_subclass__``, which a
        # type checker cannot see.  Annotated and never assigned, so it stays
        # out of ``__dict__`` -- ``declared_members`` and the reserved-name
        # check both read that, and an assignment here would make the base look
        # as though it had declared its own projection.
        dataflow: Projection[DataflowNetwork]
        network_structurally_valid: DeclaredConstraint
        segments_match_network: DeclaredConstraint
        dataflow_accepts: ConstraintGroup
        dataflow_ready: Readiness

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _synthesize_design_projection(cls)

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        return _finalize_design(cls, compiled)

    # -- what the Network is made of, role by role ----------------------------

    @property
    def roles(self) -> tuple[str, ...]:
        """Every Design role, in declaration order, active or not."""

        return tuple(segment.role for segment in _design_metadata(self).segments)

    def node_id(self, role: str) -> str:
        """The stable Network node this Design role occupies."""

        return _design_metadata(self).segment(role).node_id

    def computation(self, role: str) -> ComputationContract:
        """What the traffic at this role is required to mean."""

        return _design_metadata(self).segment(role).required_computation

    def is_active(self, role: str) -> Answer[bool]:
        """Whether this role is present at this point."""

        segment = _design_metadata(self).segment(role)
        if segment.active is None:
            return Decided(True)
        answer = _design_answer(self, segment.active)
        if isinstance(answer, Decided):
            return Decided(bool(answer.value))
        # An absent condition is not a true one: the segment simply is not there.
        return Decided(False) if isinstance(answer, Absent) else Unresolved(answer.findings)

    def selected(self, role: str) -> Answer[str]:
        """The candidate id filling this role."""

        return self._segment_view(role).selected()

    def kernel(self, role: str) -> Answer[Kernel]:
        """The child Kernel occurrence at this role's selected candidate."""

        chosen = self.selected(role)
        if not isinstance(chosen, Decided):
            return cast("Answer[Kernel]", chosen)
        return Decided(cast(Kernel, self._segment_view(role).alternative(chosen.value)))

    def region(self, role: str) -> Answer[DataflowRegion]:
        """The selected Region at this role, independent of which candidate won."""

        return cast(
            "Answer[DataflowRegion]",
            _design_answer(self, _design_metadata(self).segment(role).selected_region),
        )

    def region_family(self, role: str) -> Answer[tuple[str, str]]:
        """The selected Region's semantic family and version at one role."""

        chosen = self.selected(role)
        if not isinstance(chosen, Decided):
            return cast("Answer[tuple[str, str]]", chosen)
        metadata = _design_metadata(self).segment(role).case(chosen.value).metadata
        return Decided((metadata.region_family, metadata.region_version))

    # -- provenance -----------------------------------------------------------

    @property
    def assignments(self) -> Mapping[QualifiedPath, object]:
        """Every committed Decision this Design owns: its own, and its selectors."""

        runtime = layer_runtime(self)
        committed = runtime.point.assignments
        return MappingProxyType(
            {
                path: committed[path]
                for path in sorted(_design_metadata(self).design_decisions)
                if path in committed
            }
        )

    @property
    def imported_decisions(self) -> tuple[QualifiedPath, ...]:
        """Committed Decisions this Design reads but no part of it owns."""

        runtime = layer_runtime(self)
        return imported_decisions(
            runtime.point,
            runtime.compiled.spec,
            runtime.compiled.inputs,
            _design_metadata(self).internal_decisions,
        )

    def _segment_view(self, role: str) -> ChoiceView:
        """The bound SubspaceChoice view of one role, reached by its own declaration."""

        return occurrence_choice(self, _design_metadata(self).segment(role).declaration)


def _design_metadata(design: DataflowDesign) -> _DesignCompilation[DataflowDesign]:
    metadata = layer_runtime(design).compiled.extension
    if not isinstance(metadata, _DesignCompilation):
        raise AuthoringError(f"{type(design).__name__} is not a compiled DataflowDesign")
    return cast("_DesignCompilation[DataflowDesign]", metadata)


def _design_answer(design: DataflowDesign, reference: _Ref[object]) -> Answer[object]:
    runtime = layer_runtime(design)
    with runtime.lock:
        return answer_for(runtime.engine, runtime.point, reference)


def _atomic(what: str, value: str | None) -> None:
    """A role, node id, or case id is one path segment, not a qualified path.

    Namespaces are built by joining these with dots and read back by splitting
    on them, so a value containing a dot would compile to something other than
    what the author wrote and then be silently truncated on the way back.
    """

    if value is None:
        return
    if not value:
        raise AuthoringError(f"{what} must be non-empty")
    if "." in value:
        raise AuthoringError(f"{what} must be one path segment; {value!r} contains a dot")


class KernelChoice(SubspaceChoice):
    """One stable Design segment holding exactly one selected Kernel.

    A thin ``SubspaceChoice`` specialization written entirely through the
    generic seam.  Selector creation, alternative gating, stable namespaces,
    inspection, and selected-output forwarding all stay in the generic
    machinery; what a Kernel segment adds is the required computation, the
    implicit selected Region, and the stable Design role and Network node
    identity.

    Its alternatives are written positionally rather than as a mapping because a
    Kernel already carries its own stable ``id``, and that id *is* the
    alternative id -- which is what ``candidate_id`` exists to say.  Restating
    it as a mapping key would be the one duplication a SubspaceChoice's mapping
    exists to avoid, in the one place where the id is not the author's to
    choose.  ``Subspace(..., name=...)`` still aliases it, which is precisely
    what lets one Kernel class fill two candidate slots.
    """

    computation: ComputationContract
    node_id: str | None

    #: A Kernel segment's selector chooses a Kernel, so it says so.
    selector_name: ClassVar[str] = "kernel"

    __slots__ = ("computation", "node_id")

    def __init__(
        self,
        *alternatives: Subspace[Kernel],
        computation: ComputationContract,
        role: str | None = None,
        node_id: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(computation, ComputationContract):
            raise AuthoringError("a Kernel segment declares one ComputationContract")
        if any(not isinstance(item, Subspace) for item in alternatives):
            raise AuthoringError("a Kernel segment takes Subspace declarations")
        _atomic("a Kernel segment role", role)
        _atomic("a Kernel segment node id", node_id)
        object.__setattr__(self, "computation", computation)
        object.__setattr__(self, "node_id", node_id)
        # The role *is* the namespace segment, so it travels as the choice's
        # stable name rather than as a second parallel identity.
        self._from_candidates(
            tuple(cast("Subspace[Space]", item) for item in alternatives),
            when=when,
            name=role,
        )

    def candidate_id(self, subspace: Subspace[Space]) -> str:
        """A Kernel candidate is named by its Kernel id unless the author aliases it."""

        if subspace.stable_name is not None:
            _atomic("a Kernel candidate id", subspace.stable_name)
            return subspace.stable_name
        candidate = cast(str, getattr(subspace.space_type, "id", "")) or ""
        if not candidate:
            raise AuthoringError(
                f"a Kernel segment candidate needs a stable id; "
                f"{subspace.space_type.__name__} declares none, so pass name= on its Subspace"
            )
        _atomic("a Kernel candidate id", candidate)
        return candidate

    def default_outputs(self) -> tuple[str, ...]:
        """Every Kernel segment forwards the selected candidate's Region."""

        return ("region",)

    def validate_candidate(self, owner: type[Space], subspace: Subspace[Space]) -> None:
        del owner
        if not issubclass(subspace.space_type, Kernel):
            raise AuthoringError(
                f"a Kernel segment admits Kernels, and "
                f"{subspace.space_type.__name__} is not a Kernel"
            )
        offered = getattr(subspace.space_type, "computation", None)
        if offered != self.computation:
            raise AuthoringError(
                f"the segment requires computation "
                f"{self.computation.id}:{self.computation.version}, but "
                f"{subspace.space_type.__name__} declares "
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

    segment: KernelChoice
    port_id: str
    output: bool

    def __post_init__(self) -> None:
        if not self.port_id:
            raise AuthoringError("an endpoint needs a non-empty port id")


@dataclass(frozen=True, slots=True, eq=False)
class EdgeSink:
    """One consuming endpoint of a NetworkEdge and its source-to-sink map.

    Per-sink rather than per-edge because a canonical fan-out is one Edge with
    several sink contracts, and each contract owns its own position map.  An
    omitted map means the identity over the selected source port's image.
    """

    endpoint: SegmentEndpoint
    position_map: ValueSource[PositionMap] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, SegmentEndpoint):
            raise AuthoringError("an EdgeSink consumes a segment endpoint")
        if self.endpoint.output:
            raise AuthoringError(
                "an EdgeSink must name an input port; "
                f"{self.endpoint.port_id!r} is declared as output"
            )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NetworkEdge:
    """One canonical Edge: one source endpoint replicated to ordered sinks."""

    source: SegmentEndpoint
    sinks: tuple[EdgeSink, ...]
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        source: SegmentEndpoint,
        *sinks: EdgeSink,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(source, SegmentEndpoint):
            raise AuthoringError("a NetworkEdge starts at a segment endpoint")
        if not source.output:
            raise AuthoringError(
                f"a NetworkEdge source must name an output port; "
                f"{source.port_id!r} is declared as input"
            )
        if not sinks:
            raise AuthoringError("a NetworkEdge needs at least one EdgeSink")
        if any(not isinstance(sink, EdgeSink) for sink in sinks):
            raise AuthoringError("a NetworkEdge takes EdgeSink declarations after its source")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sinks", tuple(sinks))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a NetworkEdge"))
        object.__setattr__(self, "when", when)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NetworkBoundary:
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
            raise AuthoringError("a NetworkBoundary exposes a segment endpoint")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "stable_name", _declaration_name(name, "a NetworkBoundary"))
        object.__setattr__(self, "when", when)


TopologyDeclaration = NetworkEdge | NetworkBoundary

#: Class-body values the Design compiler recognizes beyond the generic set.
TOPOLOGY_TYPES: tuple[type, ...] = (NetworkEdge, NetworkBoundary)


def topology_members(
    design_type: type[DataflowDesign],
) -> tuple[tuple[str, TopologyDeclaration], ...]:
    """Collect NetworkEdge and NetworkBoundary members in deterministic inherited order."""

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

    #: The stable selector value.  Equals ``kernel_id`` unless the author aliased
    #: the case, which is legal precisely so one Kernel class can appear twice.
    case_id: str
    compiled: _CompiledSpace[Kernel]

    @property
    def kernel_id(self) -> str:
        return self.compiled.owner.id

    @property
    def metadata(self) -> _KernelCompilation[Kernel]:
        return cast("_KernelCompilation[Kernel]", self.compiled.extension)


@dataclass(frozen=True)
class _CompiledKernelSegment:
    """One stable Design role and the Kernel branch that fills it."""

    member_name: str
    declaration: KernelChoice
    role: str
    node_id: str
    required_computation: ComputationContract
    active: _Ref[object] | None
    selector: _Ref[object] | None
    cases: tuple[_CompiledKernelCase, ...]
    selected_region: _Ref[DataflowRegion]

    def case(self, case_id: str) -> _CompiledKernelCase:
        for candidate in self.cases:
            if candidate.case_id == case_id:
                return candidate
        raise AuthoringError(f"segment {self.role!r} has no candidate {case_id!r}")


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
    #: Decisions this Design owns: its own, its helpers', and its selectors.
    design_decisions: frozenset[QualifiedPath]
    #: Decisions a contained candidate Kernel owns.  Not the Design's to retain.
    kernel_decisions: frozenset[QualifiedPath]
    #: Every decision inside the complete Design fragment.  Nothing in here is
    #: imported provenance, however deeply nested it sits.
    internal_decisions: frozenset[QualifiedPath]

    def segment(self, role: str) -> _CompiledKernelSegment:
        for candidate in self.segments:
            if candidate.role == role:
                return candidate
        raise AuthoringError(f"this Design has no segment {role!r}")


def _segment(
    design_type: type[DataflowDesign],
    member_name: str,
    declaration: KernelChoice,
    branch: _CompiledBranch,
) -> _CompiledKernelSegment:
    # Taken from the declaration, not split back out of the namespace it was
    # joined into.  Reconstruction happened to work only because roles are
    # atomic, and would have gone quietly wrong the moment one was not.
    role = member_name if declaration.stable_name is None else declaration.stable_name
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
    reference = resolve_value_source(compiled, cast("ValueSource[object]", when), what)
    if reference.semantics.type_token is not bool:
        raise AuthoringError(f"{design_type.__name__} {what} when= is not Boolean")
    return reference


def _gate(active: _Ref[object] | None, scope: str) -> EvaluatorSpec[Answer[bool]] | None:
    """Lower one topology condition into an applicability evaluator."""

    if active is None:
        return None
    name = f"when@{scope}"

    def applies(values: DependencyView) -> Answer[bool]:
        return Decided(bool(values[name]))

    return EvaluatorSpec((active.dependency(name),), applies)


def _position_map_property(
    path: QualifiedPath,
    source: _Ref[object],
    gate: EvaluatorSpec[Answer[bool]] | None,
) -> DerivedProperty:
    """Forward one declared position map, under the NetworkEdge's own condition.

    The Network cannot depend on the author's map value directly.  A dependency
    is demanded before the evaluator runs, so an inactive NetworkEdge whose map
    reads an uncommitted Decision would leave the whole Network unresolved --
    an inactive topology declaration that is not actually inactive.  Routing the
    map through a property that carries the NetworkEdge's gate makes absence the
    answer, because applicability is decided before dependencies are prepared.
    """

    def forward(values: DependencyView) -> Answer[object]:
        return Decided(values["map"])

    return DerivedProperty(
        path,
        source.semantics,
        EvaluatorSpec((source.dependency("map"),), forward),
        gate,
    )


def _topology(
    design_type: type[D],
    compiled: _CompiledSpace[D],
    segments: tuple[_CompiledKernelSegment, ...],
) -> tuple[
    tuple[_CompiledConnection, ...],
    tuple[_CompiledBoundary, ...],
    tuple[DerivedProperty, ...],
]:
    by_declaration = {
        id(declaration): segment
        for segment, declaration in zip(
            segments,
            (
                value
                for _name, value in declared_members(design_type)
                if isinstance(value, KernelChoice)
            ),
        )
    }
    connections: list[_CompiledConnection] = []
    boundaries: list[_CompiledBoundary] = []
    maps: list[DerivedProperty] = []
    for member_name, declaration in topology_members(design_type):
        identity = member_name if declaration.stable_name is None else declaration.stable_name
        what = f"topology {identity!r}"
        active = _boolean_ref(design_type, compiled, declaration.when, what)
        if isinstance(declaration, NetworkEdge):
            gate = _gate(active, f"{compiled.namespace}.{identity}")
            sinks: list[_CompiledSink] = []
            for index, sink in enumerate(declaration.sinks):
                if sink.position_map is None:
                    sinks.append(
                        _CompiledSink(
                            _endpoint(design_type, by_declaration, sink.endpoint, what), None
                        )
                    )
                    continue
                source = resolve_value_source(
                    compiled, cast("ValueSource[object]", sink.position_map), what
                )
                map_path = QualifiedPath(
                    f"semantic.{compiled.namespace}.{identity}.position_map.{index}"
                )
                maps.append(_position_map_property(map_path, source, gate))
                sinks.append(
                    _CompiledSink(
                        _endpoint(design_type, by_declaration, sink.endpoint, what),
                        _Ref(map_path, DependencyKind.PROPERTY, source.semantics),
                    )
                )
            connections.append(
                _CompiledConnection(
                    identity,
                    _endpoint(design_type, by_declaration, declaration.source, what),
                    tuple(sinks),
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
    return tuple(connections), tuple(boundaries), tuple(maps)


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


def _declared_segments(
    design_type: type[DataflowDesign],
) -> tuple[tuple[str, str, KernelChoice], ...]:
    """Every Kernel segment's role and node id, read from the class body alone.

    The same derivation ``_segment`` performs during lowering, available before
    compilation because it consults only what the author wrote.  Keeping the two
    in step matters: a synthesized constraint that named a different node id
    from the one the Network is built with would compare a Design against a
    Design that does not exist.
    """

    resolved: list[tuple[str, str, KernelChoice]] = []
    for member_name, declaration in declared_members(design_type):
        if not isinstance(declaration, KernelChoice):
            continue
        # The candidate check the compiler would run later, run now.  Synthesis
        # is about to ask each alternative for its exported Region, and a
        # non-Kernel candidate would fail that with "does not export 'region'"
        # -- true, unhelpful, and not the thing the author got wrong.
        for _alternative_id, subspace in declaration.alternatives:
            declaration._admit_candidate(design_type, member_name, subspace)
        role = member_name if declaration.stable_name is None else declaration.stable_name
        resolved.append((role, declaration.node_id or role, declaration))
    return tuple(resolved)


def _segment_region(declaration: KernelChoice) -> ValueSource[object]:
    """One segment's selected-Region handle, past mypy's descriptor rule.

    ``KernelChoice.region`` returns a ``ValueSource``, and a type checker applies the
    descriptor protocol to any attribute whose declared type defines
    ``__get__`` -- so the handle types as the resolved Region it would produce
    on an attached occurrence.  Here it is a declaration being wired into
    another declaration, and no occurrence exists yet.
    """

    return cast("ValueSource[object]", declaration.region)


def _network_valid_constraint(network: ValueSource[object]) -> DeclaredConstraint:
    """Adapt every canonical network issue, preserving its code.

    Every issue, not the first.  ``validate_network`` reports a list because a
    malformed topology is routinely malformed in several ways at once, and a
    constraint that surfaced one at a time would turn one report into a queue of
    apparent regressions.
    """

    def evaluate(*, network: object) -> object:
        report = validate_network(cast(DataflowNetwork, network))
        if not report.issues:
            return True
        return reject_all(
            reject(
                f"design-network-{issue.code}",
                issue.message,
                values={"network_path": issue.path},
            )
            for issue in report.issues
        )

    return DeclaredConstraint((("network", network),), evaluate)


def _correspondence_constraint(
    design_type: type[DataflowDesign], network: ValueSource[object]
) -> DeclaredConstraint:
    """The one Design-specific supplement to canonical Network validation.

    Canonical validation already proves every semantic topology claim.  What it
    cannot know is that these nodes are exactly this Design's active segments and
    hold exactly their selected Regions -- which is the whole relationship a
    coverage object would otherwise have to carry.

    Its dependencies are the ordinary declared handles the author already has:
    each segment's selected ``region`` branch output and, where one exists, the
    segment's own ``when``.  Both are ``allow_absent`` because an inactive role
    contributes no node, and reporting that as "the Design refused" rather than
    "this role is not here" is exactly the confusion the marker exists to stop.
    """

    segments = _declared_segments(design_type)
    dependencies: list[tuple[str, ValueSource[object]]] = [("network", network)]
    # Positional keys, not roles: a dependency name is an evaluator parameter,
    # and a role is only required to be a single path segment.
    plan: list[tuple[str, str, str, str | None]] = []
    for index, (role, node_id, declaration) in enumerate(segments):
        region_key = f"region_{index}"
        dependencies.append((region_key, allow_absent(_segment_region(declaration))))
        active_key: str | None = None
        if declaration.when is not None:
            active_key = f"active_{index}"
            dependencies.append(
                (active_key, allow_absent(cast("ValueSource[object]", declaration.when)))
            )
        plan.append((role, node_id, region_key, active_key))

    def evaluate(**values: object) -> object:
        resolved = cast(DataflowNetwork, values["network"])
        expected: dict[str, DataflowRegion] = {}
        for _role, node_id, region_key, active_key in plan:
            if active_key is not None and not _active(values[active_key]):
                continue
            region = values[region_key]
            if region is not ABSENT:
                expected[node_id] = cast(DataflowRegion, region)
        reasons = []
        if not expected:
            reasons.append(
                reject(
                    "design-no-active-segment",
                    "a Design point must leave at least one segment active",
                )
            )
        actual = {node.id: node.region for node in resolved.nodes}
        for node_id in sorted(set(expected) ^ set(actual)):
            reasons.append(
                reject(
                    "design-segment-node-mismatch",
                    "active segments and Network nodes must correspond exactly",
                    values={"node": node_id},
                )
            )
        for node_id in sorted(set(expected) & set(actual)):
            if expected[node_id] != actual[node_id]:
                reasons.append(
                    reject(
                        "design-node-region-mismatch",
                        "a Network node does not hold its segment's selected Region",
                        values={"node": node_id},
                    )
                )
        return True if not reasons else reject_all(reasons)

    # Generated evaluator, generated parameter list: stated explicitly rather
    # than weakening the compiler's signature check for every author.
    evaluate.__signature__ = Signature(  # type: ignore[attr-defined]
        [_SignatureParameter(name, _SignatureParameter.KEYWORD_ONLY) for name, _ in dependencies]
    )
    return DeclaredConstraint(tuple(dependencies), evaluate)


def _synthesize_design_projection(design_type: type[DataflowDesign]) -> None:
    """Write the dataflow projection and its parts onto one concrete Design class.

    The Design counterpart of ``_synthesize_projections``, and for the same
    reason: the output of the projection is this class's Network over this
    class's segments, and a base class has neither.

    An abstract intermediate -- a Design with no ``KernelChoice`` segment yet -- is
    left alone.  ``_finalize_design`` is where a class that is actually compiled
    is required to be complete, so an incomplete base is an ordinary thing to
    write and not an error until someone tries to use it.
    """

    shadowed = sorted(RESERVED_DESIGN_NAMES & set(design_type.__dict__))
    if shadowed:
        raise AuthoringError(
            f"{design_type.__name__} declares {shadowed[0]!r}, which the Design layer "
            "synthesizes from its Kernel segments, its topology and its support group"
        )

    segments = _declared_segments(design_type)
    if not segments:
        return

    declarations = dict(declared_members(design_type))
    network = declarations.get("network")
    if not isinstance(network, SelectedNetwork):
        raise AuthoringError(
            f"{design_type.__name__} has no SelectedNetwork member named 'network'; "
            "it is declared by DataflowDesign and must not be replaced"
        )
    network_source = cast("ValueSource[object]", network)

    support = declarations.get("dataflow_support")
    if support is not None and not isinstance(support, ConstraintGroup):
        raise AuthoringError(
            f"{design_type.__name__}.dataflow_support is a {type(support).__name__}; it names "
            "one ConstraintGroup of the constraints that gate the Network projection"
        )

    network_valid = _network_valid_constraint(network_source)
    correspondence = _correspondence_constraint(design_type, network_source)
    dataflow_accepts = ConstraintGroup(
        network_valid,
        correspondence,
        *(support.constraints if isinstance(support, ConstraintGroup) else ()),
    )
    # The Network and every role's selected Region.  Nothing physical: see
    # ``_readiness_properties``.  The Design's own Decisions cannot be named
    # here -- they include selectors and Decisions nested in helper Spaces --
    # and are added to the compiled profile by ``_finalize_design``.
    dataflow_ready = Readiness(
        properties=(
            network_source,
            *(_segment_region(declaration) for _r, _n, declaration in segments),
        ),
        constraints=dataflow_accepts,
    )

    setattr(design_type, "network_structurally_valid", network_valid)
    setattr(design_type, "segments_match_network", correspondence)
    setattr(design_type, "dataflow_accepts", dataflow_accepts)
    setattr(design_type, "dataflow_ready", dataflow_ready)
    setattr(
        design_type,
        "dataflow",
        Projection(
            cast("ValueSource[DataflowNetwork]", network),
            readiness=dataflow_ready,
            constraints=(dataflow_accepts,),
            name="dataflow",
        ),
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
        if isinstance(declaration, KernelChoice)
    )
    if not segments:
        raise AuthoringError(f"{design_type.__name__} must declare at least one Kernel segment")
    _check_unique(design_type, segments)

    connections, boundaries, map_properties = _topology(design_type, design, segments)
    network = _substituted_network(design_type, design)
    network_property = _network_property(network.path, segments, connections, boundaries)
    _check_constraints_are_classified(design_type, declarations)
    # Every constraint in the flat fragment except the ones a candidate Kernel
    # declared as gating its build unit alone.  This is where U3's claim stops
    # being a hope: a DotpAxi that cannot pump at one SIMD lane, or has no
    # realization for this DSP generation, must not make the Network refuse --
    # it did not change a Region, and something else may build the same Region.
    physical_only = frozenset(
        path
        for segment in segments
        for case in segment.cases
        for path in case.metadata.physical_only_constraints
    )
    # ...and every *other* constraint a candidate declares, which is the half of
    # the same claim that was missing: excluding the physical ones only means
    # something if the rest are included.  Until this line the Design's
    # feasibility set held its own two constraints alone, so a Kernel that
    # refused its Region -- for a reason it had explicitly classified as
    # semantic -- left the Network accepted, and the exclusion below had nothing
    # to exclude from.
    #
    # An unselected candidate is not a problem and does not need guarding here:
    # its constraints depend on a branch this point did not take, so the engine
    # answers them Absent and reports them as *not applicable* rather than
    # refused.  That is the same machinery a conditional segment already relies
    # on, and it is why this can be a flat union over every case.
    candidate_dataflow = frozenset(
        declaration.path
        for segment in segments
        for case in segment.cases
        for declaration in case.compiled.spec.constraints
        if declaration.path not in physical_only
    )
    feasibility_name = design.engine_name(
        design.constraint_set_names, "dataflow_accepts", "ConstraintGroup"
    )
    readiness_name = design.engine_name(design.readiness_names, "dataflow_ready", "Readiness")
    internal_decisions = frozenset(item.path for item in design.spec.decisions)
    kernel_decisions = frozenset(
        item.path
        for segment in segments
        for case in segment.cases
        for item in case.compiled.spec.decisions
    )
    # Everything the Design fragment declares that a contained Kernel does not:
    # its own Decisions, those nested in Design-owned helper Spaces and generic
    # branches, and the segment selectors, which live in the Design namespace.
    design_decisions = internal_decisions - kernel_decisions
    accepted = (
        *(
            path
            for item in design.spec.constraint_sets
            if item.name == feasibility_name
            for path in item.constraints
            if path not in physical_only
        ),
        *sorted(candidate_dataflow, key=str),
    )
    specification = replace(
        design.spec,
        properties=(
            *(
                # The one substitution: the declared placeholder keeps its path,
                # its semantics and its member identity, and gains the evaluator
                # that only lowering could write.
                network_property if declaration.path == network.path else declaration
                for declaration in design.spec.properties
            ),
            *map_properties,
        ),
        constraint_sets=tuple(
            replace(item, constraints=accepted) if item.name == feasibility_name else item
            for item in design.spec.constraint_sets
        ),
        readiness_profiles=tuple(
            # The author's profile named the Network and the selected Regions.
            # The Design's own Decisions are added here because they cannot be
            # named in a class body: they include the segment selectors and any
            # Decision nested in a Design-owned helper Space.
            replace(item, decisions=tuple(sorted(design_decisions)), constraints=accepted)
            if item.name == readiness_name
            else item
            for item in design.spec.readiness_profiles
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
        design_decisions,
        kernel_decisions,
        internal_decisions,
    )
    exports = dict(design.exports)
    exports.setdefault("network", cast("_Ref[object]", network))
    return replace(design, spec=specification, exports=tuple(exports.items()), extension=metadata)


def _substituted_network(design_type: type[D], design: _CompiledSpace[D]) -> _Ref[DataflowNetwork]:
    """Locate the declared placeholder this Design's real Network replaces.

    By member identity, never by rebuilding ``semantic.<ns>.network``.  A path
    string reconstructed here is a second copy of a convention only the compiler
    is entitled to state, and the failure it produces -- a property quietly
    written at a path nothing reads -- looks like an unresolved Network rather
    than like a lowering bug.

    Each refusal below is a way the substitution could silently do nothing.
    """

    declarations = dict(declared_members(design_type))
    if not isinstance(declarations.get("network"), SelectedNetwork):
        raise AuthoringError(
            f"{design_type.__name__} has no SelectedNetwork member named 'network'"
        )
    reference = design.member("network")
    if reference.kind is not DependencyKind.PROPERTY:
        raise AuthoringError(f"{design_type.__name__}.network did not compile to a property")
    if reference.semantics.type_token is not DATAFLOW_NETWORK_SEMANTICS.type_token:
        raise AuthoringError(
            f"{design_type.__name__}.network is not a DataflowNetwork; it compiled as "
            f"{reference.semantics.name}"
        )
    occupants = tuple(
        declaration for declaration in design.spec.properties if declaration.path == reference.path
    )
    if len(occupants) != 1:
        raise AuthoringError(
            f"{design_type.__name__} has {len(occupants)} compiled properties at "
            f"{reference.path}; the Network placeholder must be exactly one"
        )
    return cast("_Ref[DataflowNetwork]", reference)


def _check_constraints_are_classified(
    design_type: type[DataflowDesign], declarations: Mapping[str, object]
) -> None:
    """Every authored Design Constraint gates the dataflow projection, explicitly.

    The same rule the Kernel layer states, for the same reason: a constraint in
    no group is compiled, evaluated and consulted by nothing.  A Design has one
    projection today, so there is one group to be in; when U5 adds the physical
    one, this is the place that will make each existing constraint say which.
    """

    grouped: set[int] = set()
    for name in ("dataflow_accepts", "dataflow_support"):
        group = declarations.get(name)
        if (
            name == "dataflow_support"
            and group is not None
            and not isinstance(group, ConstraintGroup)
        ):
            raise AuthoringError(
                f"{design_type.__name__}.dataflow_support is a {type(group).__name__}; it names "
                "one ConstraintGroup of the constraints that gate the Network projection"
            )
        if isinstance(group, ConstraintGroup):
            grouped.update(id(item) for item in group.constraints)
    ungrouped = sorted(
        name
        for name, declaration in declarations.items()
        if isinstance(declaration, DeclaredConstraint) and id(declaration) not in grouped
    )
    if ungrouped:
        raise AuthoringError(
            f"{design_type.__name__} declares Constraint {ungrouped[0]!r} outside "
            "dataflow_support; a Design says which projection each of its constraints "
            "gates, because a constraint in no group refuses nothing"
        )


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


def design_dataflow(
    engine: Engine,
    compiled: _CompiledSpace[D],
    point: DesignPoint,
) -> ProjectionAssessment[DataflowNetwork]:
    """Ask one compiled Design fragment for its Network at one point.

    The direct-fragment form of ``design.dataflow``, and the exact counterpart
    of ``kernel_dataflow``: for a caller holding a compiled record rather than
    an attached occurrence -- evidence driving a composed fragment from a flat
    engine point.  Same compiled projection, same reduction.

    Unlike the string-keyed projection it used to resolve, ``"dataflow"`` here
    is the ordinary member name of a declared ``Projection``, registered by the
    generic compiler like any other.
    """

    return cast(
        "ProjectionAssessment[DataflowNetwork]",
        evaluate_projection(engine, point, compiled.projection("dataflow")),
    )


__all__ = [
    "RESERVED_DESIGN_NAMES",
    "TOPOLOGY_TYPES",
    "NetworkBoundary",
    "NetworkEdge",
    "DataflowDesign",
    "KernelChoice",
    "SegmentEndpoint",
    "SelectedNetwork",
    "EdgeSink",
    "TopologyDeclaration",
    "design_dataflow",
    "topology_members",
]
