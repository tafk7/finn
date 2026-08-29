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

**Coverage is declared, never inferred.**  A Kernel says it realizes a Region
in the ``compute`` role; the assembly that owns the Network says which node
holds that role.  Two Regions with equal values are not thereby the same
Region, so a binding that guessed from shape would associate hardware with
semantics it was never told about.

This module adds no engine primitive.  A Kernel's decisions, derived
parameters, and coverage conditions are ordinary ``Decision``,
``DerivedProperty``, and ``Constraint`` values, and ``Engine.validate()``
remains the only compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from finn.dataflow.authoring.scope import Ref
from finn.dataflow.design import (
    Answer,
    Decided,
    DependencyKind,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.region import DataflowRegion, NumericElementType
from finn.dataflow.spec_algebra import SpecAuthoringError, SpecAuthoringIssue, duplicate_values

if TYPE_CHECKING:  # the authoring scope imports this module, not the reverse
    from finn.dataflow.hardware.authoring import HardwareDesign

#: Where a binding failure is reported when it belongs to no single Kernel.
BINDING_PATH = QualifiedPath("hardware.binding")


# -- coverage ----------------------------------------------------------------


@dataclass(frozen=True)
class RegionCoverage:
    """One Region role this Kernel realizes.

    The role is the Kernel's own vocabulary -- ``compute``, ``replay``,
    ``delivery`` -- not a node id.  What fills the role is stated at binding
    time by whoever owns the Network.
    """

    role: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("a Region coverage role must be named")


@dataclass(frozen=True)
class EdgeCoverage:
    """One connecting edge this Kernel realizes internally.

    A fused Kernel absorbs the edge between the Regions it covers: the
    connection still exists semantically, but no physical interface is exposed
    for it.  Declaring that is what distinguishes a fused Kernel from two
    Kernels that happen to be adjacent.
    """

    role: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("an edge coverage role must be named")


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
        if issues:
            raise SpecAuthoringError(tuple(issues))

    @property
    def region_roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.regions)

    @property
    def edge_roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.edges)


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
        raw = point.problem[source.path]
        # An element type reaches the RTL as its width.  Projecting here rather
        # than in each declaration keeps the table naming the *fact*, which is
        # what makes it readable as an audit.
        return Decided(raw.bit_width if isinstance(raw, NumericElementType) else raw)
    if source.kind is DependencyKind.DECISION:
        if source.path not in point.assignments:
            return _missing(
                "hardware-parameter-decision-unassigned",
                f"{parameter.name} needs a committed decision that is not assigned",
                source.path,
            )
        return Decided(point.assignments[source.path])
    answer = engine.query_property(point, source.path)
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
    here rather than discovered in a synthesis log.
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
                    "physical parameters must be scalar",
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


# -- the declaration ---------------------------------------------------------


@dataclass(frozen=True)
class HardwareKernelDeclaration:
    """Everything one physical Kernel owns locally, as engine declarations.

    The static half: what the Kernel declares, before any point exists.  A
    contributor writes a ``HardwareKernel`` subclass and never constructs this;
    it is what ``declare_hardware_kernel`` returns.
    """

    id: str
    version: str
    #: The path namespace this placement owns.  A Kernel placed twice has two
    #: namespaces over one authored design, so the namespace is what says which
    #: local paths are this placement's.
    namespace: str
    spec: DesignSpaceSpec
    coverage: CoveragePattern
    parameters: tuple[KernelParameter, ...] = ()
    coverage_constraints: tuple[QualifiedPath, ...] = ()
    sources: tuple[SourceFile, ...] = ()
    owner: type[HardwareKernel] | None = None

    def __post_init__(self) -> None:
        issues: list[SpecAuthoringIssue] = []
        if not self.id:
            issues.append(
                SpecAuthoringIssue("hardware-kernel-id-empty", "kernel", "Kernel id is empty")
            )
        if not self.version:
            issues.append(
                SpecAuthoringIssue("hardware-kernel-version-empty", self.id, "version is empty")
            )
        declared = {item.path for item in self.spec.constraints}
        for path in self.coverage_constraints:
            if path not in declared:
                issues.append(
                    SpecAuthoringIssue(
                        "hardware-coverage-constraint-missing",
                        str(path),
                        "the Kernel does not declare this coverage constraint",
                    )
                )
        for duplicate in duplicate_values(tuple(item.name for item in self.parameters)):
            issues.append(
                SpecAuthoringIssue(
                    "hardware-parameter-duplicate",
                    f"{self.id}.{duplicate}",
                    f"physical parameter {duplicate!r} is declared twice",
                )
            )
        issues.extend(self._parameter_source_issues())
        if issues:
            raise SpecAuthoringError(tuple(issues))

    def _parameter_source_issues(self) -> list[SpecAuthoringIssue]:
        """A parameter may only name a declaration that exists.

        A decision or property the Kernel imports is legitimately not in its
        own spec -- that is what importing means -- so only paths under this
        Kernel's own namespace are checked.  The rest are checked when the
        design space is assembled, where the other declarations are visible.
        """

        owned = {
            *(item.path for item in self.spec.decisions),
            *(item.path for item in self.spec.properties),
        }
        # A decision lands at ``<namespace>.<name>`` and a derived property at
        # ``semantic.<namespace>.<name>``, so locality is either of those forms.
        local_prefixes = (f"{self.namespace}.", f"semantic.{self.namespace}.")
        issues: list[SpecAuthoringIssue] = []
        for parameter in self.parameters:
            source = parameter.source
            if source is None or source.kind is DependencyKind.PROBLEM:
                continue
            is_local = str(source.path).startswith(local_prefixes)
            if is_local and source.path not in owned:
                issues.append(
                    SpecAuthoringIssue(
                        "hardware-parameter-source-undeclared",
                        str(source.path),
                        f"{parameter.name} names a local path this Kernel does not declare",
                    )
                )
        return issues

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    def parameter(self, name: str) -> KernelParameter | None:
        return next((item for item in self.parameters if item.name == name), None)


# -- the Kernel --------------------------------------------------------------


class HardwareKernel:
    """One physical Kernel family, and one bound instance of it.

    Subclass it to declare a hardware design.  A subclass carries a stable
    ``id`` and ``version`` and one ``define_design`` hook; passing it to
    ``declare_hardware_kernel`` runs that hook under a namespace and returns the
    ordinary engine declarations it produced.

    An *instance* is that Kernel as bound at one point: the Regions it was told
    it covers, the local choices that configured it, and its resolved physical
    parameters.  Binding returns an instance of the subclass that declared it,
    so a consumer that knows which Kernel it asked for gets the type it expects.
    """

    #: Stable identity of the family.  A subclass sets both.
    id: str = ""
    version: str = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[Any]) -> None:
        """Declare this Kernel's coverage, choices, parameters, and sources.

        The scope carries the typed inputs the covered semantics wired in, so a
        subclass narrows the parameter to its own input bundle.  The base cannot
        know that type, which is what the ``Any`` records.
        """

        raise NotImplementedError(f"{cls.__name__} does not define a design")

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        """The physical components this Kernel becomes at one binding.

        Elaboration makes no design choice.  Every value it uses is already in
        the binding, which is why this takes the binding and nothing else.
        """

        raise NotImplementedError(f"{cls.__name__} does not elaborate")

    def __init__(
        self,
        declaration: HardwareKernelDeclaration,
        point: DesignPoint,
        regions: Mapping[str, BoundRegion],
        edges: Mapping[str, str],
        parameters: Mapping[str, object],
    ) -> None:
        self.declaration = declaration
        self.point = point
        self.regions = regions
        self.edges = edges
        self.parameters = parameters
        # Instance attributes shadow the class-level family identity, so a
        # generically bound Kernel still answers correctly.
        self.id = declaration.id
        self.version = declaration.version

    def region(self, role: str) -> DataflowRegion:
        """The Region bound into one covered role."""

        return self.regions[role].region

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, covers={sorted(self.regions)})"


# -- the binding -------------------------------------------------------------


@dataclass(frozen=True)
class BoundRegion:
    """One covered role, and the exact semantic node filling it."""

    role: str
    node_id: str
    region: DataflowRegion


@dataclass(frozen=True)
class KernelBinding:
    """The selected relationship between semantics and one physical Kernel.

    It records what was covered, by which Kernel identity, with which values --
    everything an artifact identity and an association ledger need, and nothing
    that would have to be recomputed to read it back.
    """

    kernel: HardwareKernel
    regions: tuple[BoundRegion, ...]
    edges: tuple[tuple[str, str], ...]
    parameters: tuple[tuple[str, object], ...]

    @property
    def kernel_id(self) -> str:
        return self.kernel.id

    @property
    def kernel_version(self) -> str:
        return self.kernel.version

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.node_id for item in self.regions))

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(sorted(edge_id for _, edge_id in self.edges))

    def components(self) -> tuple[PhysicalComponent, ...]:
        return type(self.kernel).elaborate(self)


def _finding(code: str, message: str, values: tuple[tuple[str, object], ...] = ()) -> Finding:
    return Finding(FindingKind.LIMITATION, code, BINDING_PATH, message, values)


def bind_hardware_kernel(
    engine: Engine,
    declaration: HardwareKernelDeclaration,
    point: DesignPoint,
    regions: Mapping[str, BoundRegion],
    edges: Mapping[str, str] | None = None,
) -> Answer[KernelBinding]:
    """Bind one physical Kernel to the exact semantics it was told it covers.

    ``regions`` and ``edges`` map this Kernel's declared roles onto real node
    and edge identities.  They are supplied by whoever owns the Network,
    because that is the only thing entitled to say what a role means here --
    inferring it from Region shape would associate hardware with semantics
    nobody stated.

    Every declared role must be filled and no undeclared one supplied, every
    coverage constraint must hold, and every declared parameter must resolve.
    A Kernel bound with a parameter missing is indistinguishable from one that
    never declared it, and a consumer would read the absence as "no such
    value" rather than "the value could not be answered".
    """

    supplied_edges = dict(edges or {})
    findings: list[Finding] = []

    expected_regions, given_regions = set(declaration.coverage.region_roles), set(regions)
    if expected_regions != given_regions:
        findings.append(
            _finding(
                "hardware-coverage-region-roles-mismatch",
                f"{declaration.id} covers Region roles this binding does not fill exactly",
                (
                    ("unfilled", tuple(sorted(expected_regions - given_regions))),
                    ("undeclared", tuple(sorted(given_regions - expected_regions))),
                ),
            )
        )
    expected_edges, given_edges = set(declaration.coverage.edge_roles), set(supplied_edges)
    if expected_edges != given_edges:
        findings.append(
            _finding(
                "hardware-coverage-edge-roles-mismatch",
                f"{declaration.id} covers edge roles this binding does not fill exactly",
                (
                    ("unfilled", tuple(sorted(expected_edges - given_edges))),
                    ("undeclared", tuple(sorted(given_edges - expected_edges))),
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
    if findings:
        return Unresolved(tuple(findings))

    assessment = engine.evaluate_constraints(point, declaration.coverage_constraints)
    # Two ways to be refused, and both mean the same thing here.  ``False`` is
    # the flat answer; ``Absent`` is what ``reject(...)`` produces, which is how
    # an author says no *with a reason*.  Treating only the first as refusal
    # would let a rejection with a message through as coverage.  ``Unresolved``
    # is different in kind -- the question could not be answered -- so its own
    # findings propagate rather than being restated as a refusal.
    refused: list[QualifiedPath] = []
    unresolved: list[Finding] = []
    for path, answer in assessment.answers.items():
        if isinstance(answer, Decided):
            if answer.value is False:
                refused.append(path)
        elif isinstance(answer, Unresolved):
            unresolved.extend(answer.findings)
        else:
            refused.append(path)
    if unresolved:
        return Unresolved(tuple(unresolved))
    if refused:
        return Unresolved(
            (
                _finding(
                    "hardware-coverage-refused",
                    f"{declaration.id} does not cover this point",
                    (("constraints", tuple(str(path) for path in sorted(refused, key=str))),),
                ),
            )
        )

    values: dict[str, object] = {}
    for parameter in declaration.parameters:
        resolved = _parameter_value(engine, point, parameter)
        if not isinstance(resolved, Decided):
            return Unresolved(resolved.findings)
        values[parameter.name] = resolved.value

    bound_type = declaration.owner or HardwareKernel
    instance = bound_type(declaration, point, dict(regions), supplied_edges, values)
    return Decided(
        KernelBinding(
            instance,
            tuple(regions[role] for role in declaration.coverage.region_roles),
            tuple((role, supplied_edges[role]) for role in declaration.coverage.edge_roles),
            tuple((name, values[name]) for name in sorted(values)),
        )
    )


def bound_regions(pairs: Sequence[tuple[str, str, DataflowRegion]]) -> dict[str, BoundRegion]:
    """Build a role map from ``(role, node id, Region)`` triples."""

    return {role: BoundRegion(role, node_id, region) for role, node_id, region in pairs}


__all__ = [
    "BINDING_PATH",
    "BoundRegion",
    "CoveragePattern",
    "EdgeCoverage",
    "HardwareKernel",
    "HardwareKernelDeclaration",
    "KernelBinding",
    "KernelParameter",
    "PhysicalComponent",
    "RegionCoverage",
    "SourceFile",
    "bind_hardware_kernel",
    "bound_regions",
    "scalar_parameters",
]
