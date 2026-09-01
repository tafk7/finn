# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The physical Kernel: a hardware design family that covers selected semantics.

A ``HardwareKernel`` is a reusable physical hardware design with declared
coverage over one or more Region families and the edges between them.  It is
the unit an artifact is generated from and cached against.  It owns
microarchitecture, target coverage, physical-only decisions, physical parameter
derivation, and a source manifest -- and it owns none of the logical dataflow,
which the Region declarations upstream of it already fixed.

The name is deliberate and temporary.  ``finn.dataflow.kernels.Kernel`` still
means the *semantic* thing during the migration, so this one is spelled out
until that one is renamed and this becomes simply the Kernel.

**Coverage names a declaration, not a shape.**  A Kernel does not say "I
realize some Region in the compute role"; it says "I realize *this* declared
Region, which is required to compute *this* contract".  Binding resolves both
and checks them.  Two Regions can carry identical traffic and mean different
arithmetic -- ``REGION.md`` §3.6 is explicit that computation is a binding
concern -- so a coverage claim that rested on schedules and beat maps would
admit a maximum-reduction core into a dot-product position and never notice.

**A bound Kernel cannot see the point it was bound at.**  It gets its covered
Regions, its own committed choices, and its resolved parameters.  Elaboration
that could reach an undeclared problem field would be making a design decision
out of band, which is the one thing elaboration may never do.

This module adds no engine primitive.  A Kernel's decisions, derived
parameters, and coverage conditions are ordinary ``Decision``,
``DerivedProperty``, and ``Constraint`` values, and ``Engine.validate()``
remains the only compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from finn.dataflow.authoring.scope import Ref
from finn.dataflow.design import (
    Answer,
    Decided,
    DependencyKind,
    DesignPoint,
    Engine,
    Finding,
    FindingKind,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.hardware._declaration import (
    HardwareKernelDeclaration,
    check_declared_references,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.region import DataflowRegion
from finn.dataflow.spec_algebra import SpecAuthoringError, SpecAuthoringIssue, duplicate_values

if TYPE_CHECKING:  # the authoring scope imports this module, not the reverse
    from finn.dataflow.hardware.authoring import HardwareDesign

#: Where a binding failure is reported when it belongs to no single Kernel.
BINDING_PATH = QualifiedPath("hardware.binding")


def _finding(code: str, message: str, values: tuple[tuple[str, object], ...] = ()) -> Finding:
    return Finding(FindingKind.LIMITATION, code, BINDING_PATH, message, values)


def _resolve(engine: Engine, point: DesignPoint, path: QualifiedPath) -> Answer[object]:
    """Read one declared property, turning an engine refusal into an answer.

    ``query_property`` raises when a path is not a declared property of this
    design space -- which is exactly what an unvalidated imported reference
    looks like.  A binding that let that escape would report a stack trace
    where it owes a finding.
    """

    try:
        return engine.query_property(point, path)
    except RequestError as error:
        return Unresolved(
            (
                _finding(
                    "hardware-reference-not-declared",
                    f"{path} is not a derived property of this design space",
                    (("detail", tuple(item.message for item in error.findings)),),
                ),
            )
        )


# -- what a Region is required to compute ------------------------------------


@dataclass(frozen=True)
class ComputationContract:
    """What the traffic crossing a Region's boundary is required to mean.

    Equal Region values do not imply equal computation: a dot product, a
    maximum, and a population count over the same operands produce the same
    schedule, the same beat grouping, and the same availability.  The contract
    is what distinguishes them, so it is declared on both sides and compared.
    """

    id: str
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a computation contract must be named")


# -- coverage ----------------------------------------------------------------


@dataclass(frozen=True)
class RegionCoverage:
    """One Region this Kernel realizes, named by declaration.

    ``region`` and ``computation`` are handles the covered semantics wired in.
    ``implements`` is this Kernel's own claim about what it computes.  Binding
    resolves the first two and refuses unless the supplied Region *is* the
    declared one and the required contract *is* the implemented one.
    """

    role: str
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    implements: ComputationContract
    description: str = ""

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("a Region coverage role must be named")
        for label, handle in (("region", self.region), ("computation", self.computation)):
            if handle.kind is not DependencyKind.PROPERTY:
                raise ValueError(
                    f"coverage {label!r} must name a derived property, not a "
                    f"{handle.kind.value}; a Region is derived from the design, "
                    f"never supplied beside it"
                )


@dataclass(frozen=True)
class EdgeCoverage:
    """One connecting edge this Kernel absorbs internally.

    A fused Kernel takes the connection between two Regions it covers and
    implements it as wiring rather than as a pair of exposed interfaces.
    Declaring the roles it runs between is what lets binding check that the
    selected Network really has that edge, in that direction, between those
    two nodes -- rather than trusting an id.
    """

    role: str
    network: Ref[DataflowNetwork]
    source_role: str
    sink_role: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("an edge coverage role must be named")
        if not self.source_role or not self.sink_role:
            raise ValueError(f"edge {self.role!r} must name its source and sink roles")
        if self.source_role == self.sink_role:
            raise ValueError(f"edge {self.role!r} cannot run from a role to itself")
        if self.network.kind is not DependencyKind.PROPERTY:
            raise ValueError("edge coverage must name a derived Network property")


@dataclass(frozen=True)
class CoveragePattern:
    """The semantic shape one Kernel claims to realize."""

    regions: tuple[RegionCoverage, ...]
    edges: tuple[EdgeCoverage, ...] = ()

    def __post_init__(self) -> None:
        issues: list[SpecAuthoringIssue] = []
        if not self.regions:
            issues.append(
                SpecAuthoringIssue(
                    "coverage-pattern-empty",
                    "coverage",
                    "a Kernel must cover at least one Region",
                )
            )
        for duplicate in duplicate_values(tuple(item.role for item in self.regions)):
            issues.append(
                SpecAuthoringIssue(
                    "coverage-region-role-duplicate", duplicate, "Region role declared twice"
                )
            )
        for duplicate in duplicate_values(tuple(item.role for item in self.edges)):
            issues.append(
                SpecAuthoringIssue(
                    "coverage-edge-role-duplicate", duplicate, "edge role declared twice"
                )
            )
        known = {item.role for item in self.regions}
        for edge in self.edges:
            for endpoint in (edge.source_role, edge.sink_role):
                if endpoint not in known:
                    issues.append(
                        SpecAuthoringIssue(
                            "coverage-edge-role-unknown",
                            f"{edge.role}.{endpoint}",
                            "an absorbed edge must run between Regions this Kernel covers",
                        )
                    )
        if issues:
            raise SpecAuthoringError(tuple(issues))

    @property
    def region_roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.regions)

    @property
    def edge_roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.edges)

    def region(self, role: str) -> RegionCoverage:
        return next(item for item in self.regions if item.role == role)

    @property
    def signature(self) -> SemanticCoverageSignature:
        """Exactly what two Kernels must share to be alternatives.

        Matching role *names* is not enough, and treating it as enough was a
        real defect: two Kernels could agree on ``("compute",)`` while pointing
        at different Region declarations, or while implementing different
        arithmetic over the same one.  Either would let the physical choice
        change the semantics, which is the one thing selection may never do.

        Roles are sorted, so declaring them in a different order does not make
        two otherwise identical Kernels look like alternatives for nothing.
        """

        return SemanticCoverageSignature(
            tuple(
                sorted(
                    (
                        item.role,
                        str(item.region.path),
                        str(item.computation.path),
                        item.implements.id,
                        item.implements.version,
                    )
                    for item in self.regions
                )
            ),
            tuple(
                sorted(
                    (item.role, str(item.network.path), item.source_role, item.sink_role)
                    for item in self.edges
                )
            ),
        )


@dataclass(frozen=True)
class SemanticCoverageSignature:
    """The semantics one Kernel covers, as a comparable value.

    Two Kernels are alternatives for one another exactly when these are equal:
    same roles, over the same Region and computation declarations, implementing
    the same contracts, with the same absorbed edges running the same way.
    """

    regions: tuple[tuple[str, str, str, str, str], ...]
    edges: tuple[tuple[str, str, str, str], ...]

    def difference(self, other: SemanticCoverageSignature) -> tuple[str, ...]:
        """A short account of why two signatures are not the same.

        Returned rather than formatted so the caller decides how loud to be;
        an equality failure with no account of what differs is the kind of
        authoring error people work around instead of fixing.
        """

        reasons: list[str] = []
        mine = {item[0]: item[1:] for item in self.regions}
        theirs = {item[0]: item[1:] for item in other.regions}
        for role in sorted(set(mine) | set(theirs)):
            if role not in mine or role not in theirs:
                reasons.append(f"Region role {role!r} is covered by only one of them")
            elif mine[role] != theirs[role]:
                region, computation, contract, version = mine[role]
                other_region, other_computation, other_contract, other_version = theirs[role]
                if region != other_region:
                    reasons.append(f"{role!r} covers {region} versus {other_region}")
                if computation != other_computation:
                    reasons.append(f"{role!r} reads {computation} versus {other_computation}")
                if (contract, version) != (other_contract, other_version):
                    reasons.append(
                        f"{role!r} implements {contract}:{version} versus "
                        f"{other_contract}:{other_version}"
                    )
        if self.edges != other.edges:
            reasons.append("the absorbed edges differ")
        return tuple(reasons)


# -- physical parameters -----------------------------------------------------


@dataclass(frozen=True)
class KernelParameter:
    """One physical parameter and the declaration its value comes from.

    A parameter either names a declaration -- a problem field, a committed
    decision, or a derived property -- or is a Kernel constant that must say
    why it is one.  There is no fourth route, which is what stops a value
    reaching an artifact without ever appearing in the design point.

    Ownership is read off the handle rather than restated beside it.  Declaring
    a parameter a decision and then pointing it at a property was expressible
    before, and it is now not.

    The value arrives as it was declared.  There is no projection step: a
    parameter that needs a bit width names a property that computes one, so the
    width is in the design point and in the audit rather than happening on the
    way out.
    """

    name: str
    source: Ref[object] | None = None
    value: object | None = None
    why: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a physical parameter must be named")
        if self.source is None and not self.why:
            raise ValueError(f"{self.name} is a Kernel constant and must say why")
        if self.source is not None and self.value is not None:
            raise ValueError(f"{self.name} names a source, so it cannot also fix a value")

    @property
    def is_constant(self) -> bool:
        return self.source is None

    @property
    def ownership(self) -> str:
        """Where this value comes from, in the engine's own vocabulary."""

        return "constant" if self.source is None else self.source.kind.value


def _parameter_value(
    engine: Engine, point: DesignPoint, parameter: KernelParameter
) -> Answer[object]:
    """One declared parameter's value, read the way its handle says to."""

    if parameter.source is None:
        return Decided(parameter.value)
    source = parameter.source
    if source.kind is DependencyKind.PROBLEM:
        if source.path not in point.problem:
            return _missing(
                "hardware-parameter-problem-field-absent",
                f"{parameter.name} names a problem field this point does not carry",
                source.path,
            )
        return Decided(point.problem[source.path])
    if source.kind is DependencyKind.DECISION:
        if source.path not in point.assignments:
            return _missing(
                "hardware-parameter-decision-unassigned",
                f"{parameter.name} needs a committed decision that is not assigned",
                source.path,
            )
        return Decided(point.assignments[source.path])
    answer = _resolve(engine, point, source.path)
    if isinstance(answer, Decided):
        return Decided(answer.value)
    return Unresolved(
        answer.findings
        or (
            Finding(
                FindingKind.BLOCKER,
                "hardware-parameter-property-unresolved",
                BINDING_PATH,
                f"{parameter.name} names a property that did not resolve",
                trace=(source.path,),
            ),
        )
    )


def _missing(code: str, message: str, path: QualifiedPath) -> Unresolved:
    return Unresolved((Finding(FindingKind.BLOCKER, code, BINDING_PATH, message, trace=(path,)),))


def scalar_parameters(
    values: Mapping[str, object],
) -> tuple[tuple[str, bool | int | float | str], ...]:
    """The parameter table as sorted scalar pairs, for a physical component.

    A non-scalar would silently stringify into generated HDL, so it is refused
    here rather than discovered in a synthesis log.  An element type reaching
    this point is the usual cause, and it means a width property was owed and
    not declared.
    """

    bad = tuple(
        name for name, value in values.items() if type(value) not in (bool, int, float, str)
    )
    if bad:
        raise SpecAuthoringError(
            (
                SpecAuthoringIssue(
                    "hardware-parameter-not-scalar",
                    ", ".join(sorted(bad)),
                    "physical parameters must be scalar; declare a property that projects one",
                ),
            )
        )
    return tuple((name, cast("bool | int | float | str", values[name])) for name in sorted(values))


# -- physical structure ------------------------------------------------------


@dataclass(frozen=True)
class PhysicalComponent:
    """One instantiated hardware module a Kernel elaborates to.

    Deliberately thin: a Kernel may elaborate to one component or to several,
    and the forcing cases need to tell those apart.  Interfaces, connections,
    and semantic associations stay with the richer MVAU physical model until a
    second consumer justifies lifting them here.
    """

    id: str
    module: str
    parameters: tuple[tuple[str, bool | int | float | str], ...] = ()
    parent: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.module:
            raise ValueError("a physical component needs an id and a module")


@dataclass(frozen=True)
class SourceFile:
    """One HDL file the Kernel compiles, relative to a named root."""

    root: str
    path: str

    def __post_init__(self) -> None:
        if not self.root or not self.path:
            raise ValueError("a source file needs a root and a path")


# -- the Kernel --------------------------------------------------------------


@dataclass(frozen=True)
class KernelOrigin:
    """What a bound Kernel is, for evidence and provenance.

    This is what an association ledger quotes: which Kernel, covering which
    nodes and edges *of its local semantic assembly*, configured how.

    **It is binding provenance, not graph-occurrence identity**, and the
    distinction was got wrong here once.  The names it carries are local: the
    declaration namespace, and the node and edge ids of the Network the Kernel
    was bound against.  For the decomposed MVAU those node ids are the fixed
    ``replay`` and ``compute``, so two MVAUs at different graph positions
    produce byte-equal origins.  Anything needing to tell two *occurrences*
    apart has to read the enclosing scope, which this does not record.

    **It is still not an artifact identity and must not become one.**  It is
    sensitive to declaration paths and to field ordering, neither of which a
    reusable artifact may depend on, and it says nothing about source content.
    :mod:`finn.dataflow.hardware.identity` carries the staged artifact keys.
    """

    kernel_id: str
    kernel_version: str
    namespace: str
    covered_nodes: tuple[str, ...]
    covered_edges: tuple[str, ...]
    computations: tuple[tuple[str, str], ...]
    assignments: tuple[tuple[str, object], ...]
    parameters: tuple[tuple[str, object], ...]
    sources: tuple[tuple[str, str], ...]


class HardwareKernel:
    """One physical Kernel family, and one bound instance of it.

    Subclass it to declare a hardware design.  A subclass carries a stable
    ``id`` and ``version`` and one ``define_design`` hook; passing it to
    ``declare_hardware_kernel`` runs that hook under a namespace and returns the
    ordinary engine declarations it produced.

    An *instance* is that Kernel as bound at one point: the Regions it was told
    it covers, its own committed choices, and its resolved parameters.  It does
    not hold the point.  Everything an elaboration may use had to be declared,
    which is what makes "elaboration invents no value" a property of the type
    rather than a rule someone remembers.
    """

    #: Stable identity of the family.  A subclass sets both.
    id: str = ""
    version: str = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[Any]) -> object:
        """Declare this Kernel's coverage, choices, parameters, and sources.

        The scope carries the typed inputs the covered semantics wired in, so a
        subclass narrows the parameter to its own input bundle.  The base cannot
        know that type, which is what the ``Any`` records.
        """

        raise NotImplementedError(f"{cls.__name__} does not define a design")

    @classmethod
    def elaborate(cls, kernel: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        """The physical components this Kernel becomes at one binding.

        Elaboration makes no design choice.  Every value it may use is in the
        configured Kernel, and nothing else is reachable from it.
        """

        raise NotImplementedError(f"{cls.__name__} does not elaborate")

    def __init__(
        self,
        declaration: HardwareKernelDeclaration,
        regions: Mapping[str, BoundRegion],
        edges: Mapping[str, str],
        assignments: Mapping[QualifiedPath, object],
        parameters: Mapping[str, object],
    ) -> None:
        self.declaration = declaration
        self.regions = MappingProxyType(dict(regions))
        self.edges = MappingProxyType(dict(edges))
        #: Only this Kernel's own committed choices.  A Kernel has no business
        #: reading another's, and no way to.
        self.assignments = MappingProxyType(dict(assignments))
        self.parameters = MappingProxyType(dict(parameters))
        # Instance attributes shadow the class-level family identity, so a
        # generically bound Kernel still answers correctly.
        self.id = declaration.id
        self.version = declaration.version

    @property
    def sources(self) -> tuple[SourceFile, ...]:
        return self.declaration.sources

    def region(self, role: str) -> DataflowRegion:
        """The Region bound into one covered role."""

        return self.regions[role].region

    @property
    def kernel_id(self) -> str:
        return self.id

    @property
    def kernel_version(self) -> str:
        return self.version

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.node_id for item in self.regions.values()))

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.edges.values()))

    def origin(self) -> KernelOrigin:
        """The durable record of what this binding is."""

        return KernelOrigin(
            self.id,
            self.version,
            self.declaration.namespace,
            tuple(sorted(item.node_id for item in self.regions.values())),
            tuple(sorted(self.edges.values())),
            tuple(
                sorted(
                    (item.role, f"{item.implements.id}:{item.implements.version}")
                    for item in self.declaration.coverage.regions
                )
            ),
            tuple(sorted((str(path), value) for path, value in self.assignments.items())),
            tuple(sorted(self.parameters.items())),
            # Compile order, not sorted: ``dotp_axi`` instantiates ``dotp``,
            # which instantiates the DSP core.  The manifest is a sequence, and
            # sorting it would record a different manifest that happens to name
            # the same files.
            tuple((item.root, item.path) for item in self.declaration.sources),
        )

    def components(self) -> tuple[PhysicalComponent, ...]:
        """Elaborate this configured Kernel and audit every emitted parameter."""

        return audit_elaboration(self, type(self).elaborate(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, covers={sorted(self.regions)})"


# -- configured coverage -----------------------------------------------------


@dataclass(frozen=True)
class BoundRegion:
    """One covered role, and the exact semantic node filling it."""

    role: str
    node_id: str
    region: DataflowRegion


# -- the elaboration audit ----------------------------------------------------


def audit_elaboration(
    kernel: HardwareKernel, components: tuple[PhysicalComponent, ...]
) -> tuple[PhysicalComponent, ...]:
    """Refuse a component carrying a parameter the Kernel did not declare.

    What this establishes, precisely: **every parameter value leaving an
    elaboration is one the design point already answered.**  A name the Kernel
    never declared is refused, and a declared name carrying anything other than
    its resolved value is refused -- so the four allowed ownership routes
    (projected fact, committed decision, derived property, documented constant)
    stay the only ways a number reaches hardware.

    What it does *not* establish, and cannot: that elaboration makes no choice
    at all.  A passing check catches specific violations, and an elaboration is
    free to choose an instance name, a hierarchy, or how many components to emit
    -- none of which is a design-space coordinate.  "No design choice inside
    elaboration" is not provable by a test; this is the falsifiable part of it.

    Deliberately not a ``Finding``: a Kernel whose elaboration invents a value
    is an authoring defect in the Kernel, discovered at the moment it is used
    rather than a point this Kernel does not cover.
    """

    resolved = dict(kernel.parameters)
    undeclared: list[str] = []
    disagreeing: list[str] = []
    for component in components:
        for name, value in component.parameters:
            if name not in resolved:
                undeclared.append(f"{component.id}.{name}")
            elif value != resolved[name]:
                disagreeing.append(f"{component.id}.{name}={value!r} not {resolved[name]!r}")
    issues = []
    if undeclared:
        issues.append(
            SpecAuthoringIssue(
                "hardware-elaboration-parameter-undeclared",
                ", ".join(sorted(undeclared)),
                f"{kernel.kernel_id} elaborated a parameter it never declared; "
                "declare it so the value has an owner in the design point",
            )
        )
    if disagreeing:
        issues.append(
            SpecAuthoringIssue(
                "hardware-elaboration-parameter-recomputed",
                ", ".join(sorted(disagreeing)),
                f"{kernel.kernel_id} elaborated a declared parameter with a value "
                "the binding did not resolve",
            )
        )
    if issues:
        raise SpecAuthoringError(tuple(issues))
    return components


def _role_findings(
    declaration: HardwareKernelDeclaration,
    regions: Mapping[str, BoundRegion],
    edges: Mapping[str, str],
) -> list[Finding]:
    findings: list[Finding] = []
    for label, expected, given in (
        ("region", set(declaration.coverage.region_roles), set(regions)),
        ("edge", set(declaration.coverage.edge_roles), set(edges)),
    ):
        if expected != given:
            findings.append(
                _finding(
                    f"hardware-coverage-{label}-roles-mismatch",
                    f"{declaration.id} covers {label} roles this binding does not fill exactly",
                    (
                        ("unfilled", tuple(sorted(expected - given))),
                        ("undeclared", tuple(sorted(given - expected))),
                    ),
                )
            )
    for role, bound in regions.items():
        if bound.role != role:
            findings.append(
                _finding(
                    "hardware-coverage-role-mislabelled",
                    f"a Region bound under {role!r} names itself {bound.role!r}",
                )
            )
    return findings


def _semantic_findings(
    engine: Engine,
    point: DesignPoint,
    declaration: HardwareKernelDeclaration,
    regions: Mapping[str, BoundRegion],
) -> list[Finding]:
    """Check every covered role against the declaration it names."""

    findings: list[Finding] = []
    for coverage in declaration.coverage.regions:
        supplied = regions[coverage.role]
        declared = _resolve(engine, point, coverage.region.path)
        if not isinstance(declared, Decided):
            findings.extend(declared.findings)
        elif declared.value != supplied.region:
            findings.append(
                _finding(
                    "hardware-coverage-region-not-the-declared-one",
                    f"the Region bound into {coverage.role!r} is not the one "
                    f"{coverage.region.path} derives",
                    (("node", supplied.node_id), ("declared_at", str(coverage.region.path))),
                )
            )
        required = _resolve(engine, point, coverage.computation.path)
        if not isinstance(required, Decided):
            findings.extend(required.findings)
        elif required.value != coverage.implements:
            findings.append(
                _finding(
                    "hardware-computation-contract-mismatch",
                    f"{declaration.id} implements {coverage.implements.id!r} where "
                    f"{coverage.role!r} requires {required.value!r}",
                    (("role", coverage.role), ("implements", coverage.implements.id)),
                )
            )
    return findings


def _edge_findings(
    engine: Engine,
    point: DesignPoint,
    declaration: HardwareKernelDeclaration,
    regions: Mapping[str, BoundRegion],
    edges: Mapping[str, str],
) -> list[Finding]:
    """Check every absorbed edge really connects the two nodes it claims to."""

    findings: list[Finding] = []
    for coverage in declaration.coverage.edges:
        edge_id = edges[coverage.role]
        answer = _resolve(engine, point, coverage.network.path)
        if not isinstance(answer, Decided):
            findings.extend(answer.findings)
            continue
        network = cast(DataflowNetwork, answer.value)
        found = next((item for item in network.edges if item.id == edge_id), None)
        if found is None:
            findings.append(
                _finding(
                    "hardware-covered-edge-absent",
                    f"the selected Network has no edge {edge_id!r} for {coverage.role!r}",
                    (("edges", tuple(item.id for item in network.edges)),),
                )
            )
            continue
        source_node = regions[coverage.source_role].node_id
        sink_node = regions[coverage.sink_role].node_id
        sinks = tuple(item.endpoint.node_id for item in found.sinks)
        if found.source.node_id != source_node or sink_node not in sinks:
            findings.append(
                _finding(
                    "hardware-covered-edge-misconnected",
                    f"edge {edge_id!r} does not run from {source_node!r} to {sink_node!r}",
                    (("source", found.source.node_id), ("sinks", sinks)),
                )
            )
    return findings


def _coverage_findings(
    engine: Engine, point: DesignPoint, declaration: HardwareKernelDeclaration
) -> list[Finding]:
    """Evaluate the Kernel's own coverage conditions.

    Two ways to be refused, and both mean the same thing.  ``False`` is the
    flat answer; ``Absent`` is what ``reject(...)`` produces, which is how an
    author says no *with a reason*.  A rejection's own findings are carried
    through -- replacing "the fused core stops at 8" with "does not cover this
    point" would throw away the only part anyone can act on.
    """

    try:
        assessment = engine.evaluate_constraints(point, declaration.coverage_constraints)
    except RequestError as error:
        return [
            _finding(
                "hardware-coverage-constraint-not-declared",
                f"{declaration.id} names a coverage constraint this design space lacks",
                (("detail", tuple(item.message for item in error.findings)),),
            )
        ]
    findings: list[Finding] = []
    refused: list[QualifiedPath] = []
    for path, answer in assessment.answers.items():
        if isinstance(answer, Decided):
            if answer.value is False:
                refused.append(path)
        elif isinstance(answer, Unresolved):
            findings.extend(answer.findings)
        else:
            refused.append(path)
            findings.extend(answer.findings)
    if refused:
        findings.append(
            _finding(
                "hardware-coverage-refused",
                f"{declaration.id} does not cover this point",
                (("constraints", tuple(str(path) for path in sorted(refused, key=str))),),
            )
        )
    return findings


def bind_hardware_kernel(
    engine: Engine,
    declaration: HardwareKernelDeclaration,
    point: DesignPoint,
    regions: Mapping[str, BoundRegion],
    edges: Mapping[str, str] | None = None,
) -> Answer[HardwareKernel]:
    """Bind one physical Kernel to the exact semantics it declared it covers.

    ``regions`` and ``edges`` map this Kernel's declared roles onto real node
    and edge identities.  They are supplied by whoever owns the Network,
    because that is the only thing entitled to say what a role means here.  The
    binding then *checks* the claim rather than trusting it: the Region must be
    the one the coverage handle derives, the computation contract must be the
    one the Kernel implements, and an absorbed edge must really run between the
    two covered nodes in the selected Network.

    Every declared role must be filled and no undeclared one supplied, every
    coverage condition must hold, and every declared parameter must resolve.  A
    Kernel bound with a parameter missing is indistinguishable from one that
    never declared it, and a consumer would read the absence as "no such value"
    rather than "the value could not be answered".
    """

    supplied_edges = dict(edges or {})
    findings = _role_findings(declaration, regions, supplied_edges)
    if findings:
        # Everything below indexes by role, so a role mismatch has to stop here
        # rather than be reported alongside consequences of itself.
        return Unresolved(tuple(findings))

    findings.extend(_semantic_findings(engine, point, declaration, regions))
    findings.extend(_edge_findings(engine, point, declaration, regions, supplied_edges))
    findings.extend(_coverage_findings(engine, point, declaration))
    if findings:
        return Unresolved(tuple(findings))

    values: dict[str, object] = {}
    for parameter in declaration.parameters:
        resolved = _parameter_value(engine, point, parameter)
        if not isinstance(resolved, Decided):
            return Unresolved(resolved.findings)
        values[parameter.name] = resolved.value

    owned = {item.path for item in declaration.spec.decisions}
    local = {path: value for path, value in point.assignments.items() if path in owned}
    bound_type = declaration.owner or HardwareKernel
    instance = bound_type(declaration, dict(regions), supplied_edges, local, values)
    return Decided(instance)


def bound_regions(pairs: Sequence[tuple[str, str, DataflowRegion]]) -> dict[str, BoundRegion]:
    """Build a role map from ``(role, node id, Region)`` triples."""

    return {role: BoundRegion(role, node_id, region) for role, node_id, region in pairs}


__all__ = [
    "BINDING_PATH",
    "BoundRegion",
    "ComputationContract",
    "CoveragePattern",
    "EdgeCoverage",
    "HardwareKernel",
    "HardwareKernelDeclaration",
    "KernelOrigin",
    "KernelParameter",
    "PhysicalComponent",
    "RegionCoverage",
    "SourceFile",
    "audit_elaboration",
    "bind_hardware_kernel",
    "bound_regions",
    "check_declared_references",
    "scalar_parameters",
]
