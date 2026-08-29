# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Kernel model: what a Kernel is, and how a pool of them assembles.

A ``Kernel`` names one meaningful microarchitecture family.  A
``KernelSelection`` is one static, Op-class-owned pool of candidate Kernels
behind a single decision.  Selecting a Kernel selects its Region, its parameter
demands, and its provider inventory at once; there is no second selectable
identity between Kernel and provider.

This module adds no engine primitive.  A selection assembles ordinary problem
fields, decisions, derived properties, constraints, constraint sets, and
readiness profiles into one flat ``DesignSpaceSpec``.

**Which of these you write.**  Author a ``Kernel`` subclass and hand it to
``finn.dataflow.authoring.declare_kernel``; that is the supported contribution
path and it is what ``finn.dataflow.authoring`` re-exports.
``KernelDeclaration`` is the *compiled* form ``declare_kernel`` returns -- a
Kernel class with its scoped declarations already built -- and it stays
exported here because pool assembly and the engine both take it by name.
Constructing one by hand means writing the declarations that
``KernelDesign`` exists to write for you.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
    AbsenceMode,
    Absent,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
    validate_region,
)
from finn.dataflow.region import DataflowRegion, Port
from finn.dataflow.region_validation import RegionValidationReport
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
    duplicate_values,
    gate_spec,
    prefixed,
    rebase_spec,
    spec_declaration_paths,
)

if TYPE_CHECKING:  # the authoring scope imports this module, not the reverse
    from finn.dataflow.authoring.kernel_design import KernelDesign

#: Reserved pool member naming "no Kernel is selected for this interface".
NO_KERNEL = "none"

_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_REPORT_SEMANTICS = as_object_semantics(REGION_VALIDATION_REPORT_SEMANTICS)
_PORT_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
KERNEL_ID_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(str, name="KernelId"))


@dataclass(frozen=True)
class KernelProvider:
    """One concrete realization mechanism for exactly one Kernel."""

    id: str
    kernel_id: str
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.id or not self.kernel_id or not self.version:
            raise ValueError("provider id, Kernel id, and version must not be empty")


@dataclass(frozen=True)
class KernelDemand:
    """One parameter interface a Kernel needs supplied from outside itself."""

    interface: str
    port_path: QualifiedPath

    def __post_init__(self) -> None:
        if not self.interface:
            raise ValueError("demand interface name must not be empty")


@dataclass(frozen=True)
class KernelExport:
    """One Kernel-derived value the enclosing scope reads by a stable name.

    Exports let a pool present one path for a value every member derives its
    own way.  They carry no selection semantics of their own.
    """

    name: str
    path: QualifiedPath
    value_semantics: ValueSemantics[object]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("export name must not be empty")


@dataclass(frozen=True)
class SelectedKernel:
    """The committed *identity* of one Kernel inside one selection.

    This is the small durable value: what a design point records and what
    provenance and evidence quote.  It is deliberately not the working object
    -- for the Kernel as selected and configured in one operation, with its
    Region, demands, and providers attached, see :class:`Kernel`.
    """

    selection: str
    kernel_id: str
    version: str


SELECTED_KERNEL_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(SelectedKernel, name="SelectedKernel")
)


@dataclass(frozen=True)
class KernelDeclaration:
    """Everything one Kernel owns locally, as ordinary engine declarations.

    This is the static half of a Kernel: what it declares, before any point
    exists.  It is a value rather than the class so that the same Kernel can
    be placed under several namespaces, each placement being a different set
    of paths over the same authored design.  ``owner`` records the class that
    authored it, so a resolved selection can hand back an instance of exactly
    that class.

    It is an assembly record, not an authoring noun: a contributor writes a
    ``Kernel`` subclass and never names this type.  It is deliberately absent
    from ``finn.dataflow.authoring`` for that reason, and the pool machinery
    that does need it imports it from here.
    """

    id: str
    version: str
    spec: DesignSpaceSpec
    region_path: QualifiedPath
    feasibility_constraints: tuple[QualifiedPath, ...] = ()
    source_admission_constraints: tuple[QualifiedPath, ...] = ()
    demands: tuple[KernelDemand, ...] = ()
    exports: tuple[KernelExport, ...] = ()
    providers: tuple[KernelProvider, ...] = ()
    owner: type[Kernel] | None = None

    def __post_init__(self) -> None:
        issues: list[SpecAuthoringIssue] = []
        if not self.id:
            issues.append(SpecAuthoringIssue("kernel-id-empty", "kernel", "Kernel id is empty"))
        if not self.version:
            issues.append(
                SpecAuthoringIssue("kernel-version-empty", self.id, "Kernel version is empty")
            )
        if self.id == NO_KERNEL:
            issues.append(
                SpecAuthoringIssue(
                    "kernel-id-reserved",
                    self.id,
                    f"{NO_KERNEL!r} is reserved for unselected optional pools",
                )
            )
        properties = {item.path for item in self.spec.properties}
        constraints = {item.path for item in self.spec.constraints}
        if self.region_path not in properties:
            issues.append(
                SpecAuthoringIssue(
                    "kernel-region-path-missing",
                    str(self.region_path),
                    "the Kernel Region path is not one of its derived properties",
                )
            )
        for path in (*self.feasibility_constraints, *self.source_admission_constraints):
            if path not in constraints:
                issues.append(
                    SpecAuthoringIssue(
                        "kernel-constraint-path-missing",
                        str(path),
                        "the Kernel does not declare this constraint",
                    )
                )
        issues.extend(self._source_admission_issues())
        for duplicate in duplicate_values(tuple(item.interface for item in self.demands)):
            issues.append(
                SpecAuthoringIssue(
                    "kernel-demand-interface-duplicate",
                    f"{self.id}.{duplicate}",
                    f"demand interface {duplicate!r} is declared twice",
                )
            )
        for duplicate in duplicate_values(tuple(item.name for item in self.exports)):
            issues.append(
                SpecAuthoringIssue(
                    "kernel-export-name-duplicate",
                    f"{self.id}.{duplicate}",
                    f"export name {duplicate!r} is declared twice",
                )
            )
        for export in self.exports:
            if export.path not in properties:
                issues.append(
                    SpecAuthoringIssue(
                        "kernel-export-path-missing",
                        str(export.path),
                        "the exported path is not one of the Kernel's properties",
                    )
                )
        for demand in self.demands:
            if demand.port_path not in properties:
                issues.append(
                    SpecAuthoringIssue(
                        "kernel-demand-path-missing",
                        str(demand.port_path),
                        "the demanded Port path is not one of the Kernel's properties",
                    )
                )
        for provider in self.providers:
            if provider.kernel_id != self.id:
                issues.append(
                    SpecAuthoringIssue(
                        "provider-kernel-mismatch",
                        provider.id,
                        f"provider names Kernel {provider.kernel_id!r}, not {self.id!r}",
                    )
                )
        if issues:
            raise SpecAuthoringError(tuple(issues))

    def _source_admission_issues(self) -> list[SpecAuthoringIssue]:
        """Source admission must be answerable from problem data alone.

        Admission is asked before anything is decided, so a constraint that
        reads one of this Kernel's own decisions could never be more than a
        guess.  Such a constraint belongs in full feasibility instead.
        """

        own_decisions = {item.path for item in self.spec.decisions}
        by_path = {item.path: item for item in self.spec.constraints}
        issues: list[SpecAuthoringIssue] = []
        for path in self.source_admission_constraints:
            constraint = by_path.get(path)
            if constraint is None:
                continue
            dependencies = list(constraint.evaluator.dependencies)
            if constraint.applies_if is not None:
                dependencies.extend(constraint.applies_if.dependencies)
            for dependency in dependencies:
                if dependency.path in own_decisions:
                    issues.append(
                        SpecAuthoringIssue(
                            "source-admission-reads-a-decision",
                            str(path),
                            f"source admission cannot depend on {dependency.path}",
                        )
                    )
        return issues

    @property
    def demand_interfaces(self) -> tuple[str, ...]:
        return tuple(demand.interface for demand in self.demands)

    def demand(self, interface: str) -> KernelDemand | None:
        return next((item for item in self.demands if item.interface == interface), None)

    @property
    def export_names(self) -> tuple[str, ...]:
        return tuple(export.name for export in self.exports)

    def export(self, name: str) -> KernelExport | None:
        return next((item for item in self.exports if item.name == name), None)

    def place(
        self,
        prefix: QualifiedPath | str,
        *,
        shared_problem_paths: Mapping[QualifiedPath, QualifiedPath] | None = None,
    ) -> KernelDeclaration:
        """Return the same Kernel with every local path rebased under a prefix.

        Problem fields stay Kernel-local unless the caller explicitly maps one
        onto an outer-scope field it already declares.
        """

        shared = dict(shared_problem_paths or {})
        problem_paths = {item.path for item in self.spec.problem_schema.fields}
        unknown = sorted(set(shared) - problem_paths)
        if unknown:
            raise SpecAuthoringError(
                tuple(
                    SpecAuthoringIssue(
                        "shared-problem-path-unknown",
                        str(path),
                        "shared path is not a problem field of this Kernel",
                    )
                    for path in unknown
                )
            )
        base = QualifiedPath.parse(prefix)
        mapping = {
            path: shared.get(path, prefixed(base, path))
            for path in spec_declaration_paths(self.spec)
        }
        return KernelDeclaration(
            self.id,
            self.version,
            rebase_spec(
                self.spec,
                mapping,
                name_prefix=str(base),
                dropped_problem_paths=frozenset(shared),
            ),
            mapping[self.region_path],
            tuple(mapping[path] for path in self.feasibility_constraints),
            tuple(mapping[path] for path in self.source_admission_constraints),
            tuple(KernelDemand(item.interface, mapping[item.port_path]) for item in self.demands),
            tuple(
                KernelExport(item.name, mapping[item.path], item.value_semantics)
                for item in self.exports
            ),
            self.providers,
            self.owner,
        )


@dataclass(frozen=True)
class KernelSelectionPaths:
    """Every path one Kernel selection owns in the assembled flat scope."""

    kernel: QualifiedPath
    selected_kernel: QualifiedPath
    region: QualifiedPath
    region_validation: QualifiedPath
    region_structurally_well_formed: QualifiedPath
    demands: Mapping[str, QualifiedPath] = field(default_factory=dict)
    exports: Mapping[str, QualifiedPath] = field(default_factory=dict)

    def demand(self, interface: str) -> QualifiedPath:
        return self.demands[interface]

    def export(self, name: str) -> QualifiedPath:
        return self.exports[name]


def _selection_paths(
    name: str,
    interfaces: tuple[str, ...],
    exports: tuple[str, ...],
) -> KernelSelectionPaths:
    return KernelSelectionPaths(
        QualifiedPath(f"{name}.kernel"),
        QualifiedPath(f"semantic.{name}.selected_kernel"),
        QualifiedPath(f"semantic.{name}.region"),
        QualifiedPath(f"semantic.{name}.region_validation"),
        QualifiedPath(f"constraint.{name}.region_structurally_well_formed"),
        {
            interface: QualifiedPath(f"semantic.{name}.demand.{interface}")
            for interface in interfaces
        },
        {export: QualifiedPath(f"semantic.{name}.export.{export}") for export in exports},
    )


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


@dataclass(frozen=True)
class KernelSelection:
    """One static pool of candidate Kernels behind a single decision.

    The pool is owned by the Op class.  Graph and build facts make candidates
    applicable or infeasible through problem data and constraints; they never
    add or remove pool members.
    """

    name: str
    kernels: tuple[KernelDeclaration, ...]
    optional: bool = False
    applies_if: EvaluatorSpec[Answer[bool]] | None = None

    def __post_init__(self) -> None:
        issues: list[SpecAuthoringIssue] = []
        if not self.name:
            issues.append(
                SpecAuthoringIssue("selection-name-empty", "selection", "selection name is empty")
            )
        if not self.kernels:
            issues.append(
                SpecAuthoringIssue(
                    "selection-pool-empty",
                    self.name,
                    "a Kernel selection must declare at least one candidate",
                )
            )
        for duplicate in duplicate_values(tuple(item.id for item in self.kernels)):
            issues.append(
                SpecAuthoringIssue(
                    "kernel-id-duplicate",
                    f"{self.name}.{duplicate}",
                    f"Kernel id {duplicate!r} is duplicated in this pool",
                )
            )
        if issues:
            raise SpecAuthoringError(tuple(issues))

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.kernels)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return (*self.kernel_ids, *((NO_KERNEL,) if self.optional else ()))

    @property
    def demand_interfaces(self) -> tuple[str, ...]:
        seen: list[str] = []
        for kernel in self.kernels:
            for interface in kernel.demand_interfaces:
                if interface not in seen:
                    seen.append(interface)
        return tuple(seen)

    @property
    def export_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for kernel in self.kernels:
            for name in kernel.export_names:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    @property
    def paths(self) -> KernelSelectionPaths:
        return _selection_paths(self.name, self.demand_interfaces, self.export_names)

    @property
    def structural_constraint_set(self) -> str:
        return f"{self.name}.structural"

    @property
    def feasibility_constraint_set(self) -> str:
        return f"{self.name}.feasibility"

    def kernel(self, kernel_id: str) -> KernelDeclaration:
        for candidate in self.kernels:
            if candidate.id == kernel_id:
                return candidate
        raise KeyError(f"{kernel_id!r} is not a member of the {self.name!r} Kernel pool")

    def providers(self, kernel_id: str) -> tuple[KernelProvider, ...]:
        return self.kernel(kernel_id).providers

    def feasibility_constraints(self) -> tuple[QualifiedPath, ...]:
        paths = self.paths
        return tuple(
            dict.fromkeys(
                (
                    paths.region_structurally_well_formed,
                    *(path for kernel in self.kernels for path in kernel.feasibility_constraints),
                )
            )
        )

    def source_admission_constraints(self) -> tuple[QualifiedPath, ...]:
        return tuple(
            dict.fromkeys(
                path for kernel in self.kernels for path in kernel.source_admission_constraints
            )
        )

    # -- applicability -------------------------------------------------------

    def _kernel_ref(self, name: str = "selected") -> DependencyRef:
        return DependencyRef.decision(name, self.paths.kernel, KERNEL_ID_SEMANTICS)

    def _gate(self, inner: EvaluatorSpec[Answer[bool]]) -> EvaluatorSpec[Answer[bool]]:
        if self.applies_if is None:
            return inner
        outer = self.applies_if
        dependencies = outer.dependencies + inner.dependencies

        def evaluate(values: DependencyView) -> Answer[bool]:
            outer_answer = outer.evaluator(
                DependencyView({item.name: values[item.name] for item in outer.dependencies})
            )
            if not isinstance(outer_answer, Decided) or not outer_answer.value:
                return outer_answer
            return inner.evaluator(
                DependencyView({item.name: values[item.name] for item in inner.dependencies})
            )

        return EvaluatorSpec(dependencies, evaluate)

    def kernel_applies(self, kernel_id: str) -> EvaluatorSpec[Answer[bool]]:
        """Return the applicability of declarations owned by one pool member."""

        reference = self._kernel_ref(f"{self.name}.selected")

        def evaluate(values: DependencyView) -> Answer[bool]:
            return Decided(values[reference.name] == kernel_id)

        return self._gate(EvaluatorSpec((reference,), evaluate))

    def any_kernel_applies(self) -> EvaluatorSpec[Answer[bool]]:
        """Return the applicability of declarations shared by every member."""

        reference = self._kernel_ref(f"{self.name}.selected")

        def evaluate(values: DependencyView) -> Answer[bool]:
            return Decided(values[reference.name] != NO_KERNEL)

        return self._gate(EvaluatorSpec((reference,), evaluate))

    # -- assembly ------------------------------------------------------------

    def _region_property(self, paths: KernelSelectionPaths) -> DerivedProperty:
        dependencies = tuple(
            DependencyRef.property(
                f"region_{index}",
                kernel.region_path,
                _REGION_SEMANTICS,
                absence=AbsenceMode.ALLOWS_ABSENT,
            )
            for index, kernel in enumerate(self.kernels)
        )

        def select(values: DependencyView) -> Answer[object]:
            present = tuple(value for value in values.values() if value is not ABSENT)
            if len(present) > 1:
                raise AssertionError(
                    f"at most one Kernel of {self.name!r} may derive a Region, got {len(present)}"
                )
            if not present:
                # The selected Kernel refused to derive a Region for this
                # problem.  That is a rejection to report, not an authoring bug.
                return Absent(
                    (
                        Finding(
                            FindingKind.REJECTION,
                            "kernel-region-unavailable",
                            paths.region,
                            "the selected Kernel derives no Region for this problem",
                        ),
                    )
                )
            return Decided(cast(DataflowRegion, present[0]))

        return DerivedProperty(
            paths.region,
            _REGION_SEMANTICS,
            EvaluatorSpec(dependencies, select),
            applies_if=self.any_kernel_applies(),
        )

    def _demand_property(self, paths: KernelSelectionPaths, interface: str) -> DerivedProperty:
        owners = tuple(kernel for kernel in self.kernels if kernel.demand(interface) is not None)
        dependencies = tuple(
            DependencyRef.property(
                f"demand_{index}",
                cast(KernelDemand, kernel.demand(interface)).port_path,
                _PORT_SEMANTICS,
                absence=AbsenceMode.ALLOWS_ABSENT,
            )
            for index, kernel in enumerate(owners)
        )

        def select(values: DependencyView) -> Answer[object]:
            present = tuple(value for value in values.values() if value is not ABSENT)
            if not present:
                return Absent(
                    (
                        Finding(
                            FindingKind.LIMITATION,
                            "kernel-demand-not-declared",
                            paths.demand(interface),
                            f"the selected Kernel does not demand {interface!r}",
                        ),
                    )
                )
            if len(present) != 1:
                raise AssertionError("at most one Kernel may derive one demanded interface")
            return Decided(cast(Port, present[0]))

        return DerivedProperty(
            paths.demand(interface),
            _PORT_SEMANTICS,
            EvaluatorSpec(dependencies, select),
            applies_if=self.any_kernel_applies(),
        )

    def _export_property(self, paths: KernelSelectionPaths, name: str) -> DerivedProperty:
        owners = tuple(kernel for kernel in self.kernels if kernel.export(name) is not None)
        semantics = cast(KernelExport, owners[0].export(name)).value_semantics
        dependencies = tuple(
            DependencyRef.property(
                f"export_{index}",
                cast(KernelExport, kernel.export(name)).path,
                semantics,
                absence=AbsenceMode.ALLOWS_ABSENT,
            )
            for index, kernel in enumerate(owners)
        )

        def select(values: DependencyView) -> Answer[object]:
            present = tuple(value for value in values.values() if value is not ABSENT)
            if not present:
                return Absent(
                    (
                        Finding(
                            FindingKind.LIMITATION,
                            "kernel-export-not-declared",
                            paths.export(name),
                            f"the selected Kernel does not export {name!r}",
                        ),
                    )
                )
            if len(present) != 1:
                raise AssertionError("at most one Kernel may derive one export")
            return Decided(present[0])

        return DerivedProperty(
            paths.export(name),
            semantics,
            EvaluatorSpec(dependencies, select),
            applies_if=self.any_kernel_applies(),
        )

    def _selected_kernel_property(self, paths: KernelSelectionPaths) -> DerivedProperty:
        reference = self._kernel_ref("kernel_id")
        versions = {kernel.id: kernel.version for kernel in self.kernels}

        def derive(values: DependencyView) -> Answer[object]:
            kernel_id = cast(str, values["kernel_id"])
            return Decided(SelectedKernel(self.name, kernel_id, versions[kernel_id]))

        return DerivedProperty(
            paths.selected_kernel,
            SELECTED_KERNEL_SEMANTICS,
            EvaluatorSpec((reference,), derive),
            applies_if=self.any_kernel_applies(),
        )

    def build_spec(self) -> DesignSpaceSpec:
        """Assemble this pool into one ordinary flat specification."""

        paths = self.paths
        gated = tuple(
            gate_spec(kernel.spec, self.kernel_applies(kernel.id)) for kernel in self.kernels
        )
        region_ref = DependencyRef.property("region", paths.region, _REGION_SEMANTICS)
        report_ref = DependencyRef.property("report", paths.region_validation, _REPORT_SEMANTICS)

        def validate(values: DependencyView) -> Answer[object]:
            return Decided(validate_region(cast(DataflowRegion, values["region"])))

        def well_formed(values: DependencyView) -> Answer[bool]:
            return Decided(not cast(RegionValidationReport, values["report"]))

        selection_decision = Decision(
            paths.kernel,
            KERNEL_ID_SEMANTICS,
            _finite_domain(cast(tuple[object, ...], self.candidate_ids)),
            applies_if=self.applies_if,
        )
        own = DesignSpaceSpec(
            decisions=(selection_decision,),
            properties=(
                self._region_property(paths),
                DerivedProperty(
                    paths.region_validation,
                    _REPORT_SEMANTICS,
                    EvaluatorSpec((region_ref,), validate),
                    applies_if=self.any_kernel_applies(),
                ),
                self._selected_kernel_property(paths),
                *(self._demand_property(paths, interface) for interface in self.demand_interfaces),
                *(self._export_property(paths, name) for name in self.export_names),
            ),
            constraints=(
                Constraint(
                    paths.region_structurally_well_formed,
                    EvaluatorSpec((report_ref,), well_formed),
                    applies_if=self.any_kernel_applies(),
                ),
            ),
        )
        assembled = assemble_specs((*gated, own))
        return DesignSpaceSpec(
            assembled.problem_schema,
            assembled.decisions,
            assembled.properties,
            assembled.constraints,
            (
                *assembled.constraint_sets,
                ConstraintSet(
                    self.structural_constraint_set,
                    (paths.region_structurally_well_formed,),
                ),
                ConstraintSet(self.feasibility_constraint_set, self.feasibility_constraints()),
            ),
            (
                *assembled.readiness_profiles,
                ReadinessProfile(
                    self.structural_constraint_set,
                    decisions=(paths.kernel,),
                    properties=(paths.region, paths.region_validation),
                    constraints=(paths.region_structurally_well_formed,),
                ),
                ReadinessProfile(
                    self.feasibility_constraint_set,
                    decisions=(
                        paths.kernel,
                        *(item.path for kernel in self.kernels for item in kernel.spec.decisions),
                    ),
                    properties=(
                        paths.region,
                        paths.region_validation,
                        paths.selected_kernel,
                    ),
                    constraints=self.feasibility_constraints(),
                ),
            ),
        )


class Kernel:
    """One Kernel: the family a contributor declares, and one selected instance.

    Subclass it to declare a microarchitecture family.  A subclass carries a
    stable ``id`` and ``version`` and one ``define_design`` hook; calling
    ``declare`` runs that hook under a namespace and returns the ordinary
    engine declarations it produced.

    An *instance* is that Kernel as selected and configured in one operation
    point: the choices that configured it, the one Region it derives, what it
    demands, what it exports, and who can build it.  Binding a resolved
    selection returns an instance of the subclass that declared it, so a
    consumer that knows which family it asked for gets the type it expects and
    nothing has to be downcast from a generic bag.

    There is one identity axis, named by :class:`SelectedKernel`; an instance
    is derived from a point, never selected.
    """

    #: Stable identity of the family.  A subclass sets both.
    id: str = ""
    version: str = "1"

    # -- declaration authority ---------------------------------------------

    @classmethod
    def define_design(cls, design: KernelDesign[Any]) -> None:
        """Declare this Kernel's decisions, properties, and constraints.

        The scope carries the typed inputs the owning operation wired in, so a
        subclass narrows the parameter to its own input bundle.  The base
        cannot know that type -- a pool is the thing that pairs Kernels with
        inputs -- which is what the ``Any`` records.
        """

        raise NotImplementedError(f"{cls.__name__} does not define a design")

    # -- one bound instance -------------------------------------------------

    def __init__(
        self,
        declaration: KernelDeclaration,
        selection: str,
        point: DesignPoint,
        assignments: Mapping[QualifiedPath, object],
        region: DataflowRegion,
        demands: Mapping[str, Port],
        exports: Mapping[str, object],
    ) -> None:
        self.declaration = declaration
        self.selection = selection
        self.point = point
        self.assignments = assignments
        self.region = region
        self.demands = demands
        self.exports = exports
        # Instance attributes shadow the class-level family identity, so a
        # generically bound Kernel still answers correctly.
        self.id = declaration.id
        self.version = declaration.version

    @property
    def providers(self) -> tuple[KernelProvider, ...]:
        return self.declaration.providers

    @property
    def identity(self) -> SelectedKernel:
        """The durable identity this instance persists as."""

        return SelectedKernel(self.selection, self.id, self.version)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, selection={self.selection!r})"


def bind_kernel(engine: Engine, selection: KernelSelection, point: DesignPoint) -> Answer[Kernel]:
    """Bind the selected Kernel of one pool to everything it derived.

    The returned object is an instance of the class that declared the selected
    Kernel, or of ``Kernel`` itself for a declaration authored without one.

    Every demand and export the *selected* declaration owns must resolve.  An
    unresolved one propagates rather than being dropped: a Kernel bound with a
    declared demand missing is indistinguishable from a Kernel that never
    demanded it, and a consumer would read the absence as "no such requirement"
    instead of "the requirement could not be answered".  Only the paths this
    declaration itself owns are required, because a pool path belonging to a
    different member is legitimately absent when this one is selected.
    """

    identity = selected_kernel(engine, selection, point)
    if not isinstance(identity, Decided):
        return identity
    declaration = selection.kernel(identity.value.kernel_id)
    paths = selection.paths
    region = engine.query_property(point, paths.region)
    if not isinstance(region, Decided):
        return region
    demands: dict[str, Port] = {}
    for interface in declaration.demand_interfaces:
        answer = engine.query_property(point, paths.demand(interface))
        if not isinstance(answer, Decided):
            return answer
        demands[interface] = cast(Port, answer.value)
    exports: dict[str, object] = {}
    for name in declaration.export_names:
        answer = engine.query_property(point, paths.export(name))
        if not isinstance(answer, Decided):
            return answer
        exports[name] = answer.value
    own = {item.path for item in declaration.spec.decisions}
    bound = declaration.owner or Kernel
    return Decided(
        bound(
            declaration,
            selection.name,
            point,
            {path: value for path, value in point.assignments.items() if path in own},
            cast(DataflowRegion, region.value),
            demands,
            exports,
        )
    )


def admissible_kernels(
    engine: Engine, selection: KernelSelection, point: DesignPoint
) -> tuple[str, ...]:
    """Return the pool members whose source-admission constraints hold.

    Every source-admission constraint is decision-free by construction (see
    ``KernelDeclaration._source_admission_issues``), so evaluating them against the
    problem is the whole existential question -- there is no local completion
    left to witness.  Target feasibility is deliberately not asked here; it is
    a selection-time query, and asking it now would make inference coverage
    depend on the board.
    """

    admitted: list[str] = []
    for kernel in selection.kernels:
        if not kernel.source_admission_constraints:
            admitted.append(kernel.id)
            continue
        result = engine.commit_assignments(point, {selection.paths.kernel: kernel.id})
        if any(
            outcome.disposition not in {"committed", "unchanged"} for outcome in result.outcomes
        ):
            continue
        assessment = engine.evaluate_constraints(result.point, kernel.source_admission_constraints)
        refused = any(
            isinstance(answer, Decided) and answer.value is False
            for answer in assessment.answers.values()
        )
        if not refused:
            admitted.append(kernel.id)
    return tuple(admitted)


def provider_of(
    selection: KernelSelection, kernel_id: str, provider_id: str
) -> KernelProvider | None:
    """Look one provider up in the Kernel's own inventory.

    Elaboration dispatches through this so the inventory is the single answer
    to "what can build this Kernel".  A miss is an ordinary negative answer --
    that Kernel simply has no such realization path -- so the caller decides
    how to report it.
    """

    kernel = selection.kernel(kernel_id)
    return next((item for item in kernel.providers if item.id == provider_id), None)


def selected_kernel(
    engine: Engine, selection: KernelSelection, point: DesignPoint
) -> Answer[SelectedKernel]:
    """Return the committed Kernel identity of one selection, if any."""

    answer = engine.query_property(point, selection.paths.selected_kernel)
    if isinstance(answer, Decided):
        return Decided(cast(SelectedKernel, answer.value))
    if isinstance(answer, Absent):
        return Absent(answer.findings)
    return Unresolved(answer.findings)


__all__ = [
    "KERNEL_ID_SEMANTICS",
    "NO_KERNEL",
    "SELECTED_KERNEL_SEMANTICS",
    "Kernel",
    "KernelDemand",
    "KernelDeclaration",
    "KernelExport",
    "KernelProvider",
    "KernelSelection",
    "KernelSelectionPaths",
    "SelectedKernel",
    "admissible_kernels",
    "bind_kernel",
    "provider_of",
    "selected_kernel",
]
