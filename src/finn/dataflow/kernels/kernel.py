# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One-Region Kernel specialization of the declarative Space frontend.

A ``Kernel`` answers exactly two questions about itself, and they are asked
separately::

    kernel.dataflow   -> ProjectionAssessment[DataflowRegion]
    kernel.physical   -> ProjectionAssessment[ModuleBuildSpec]

The separation is the point.  The Region is the logical contract a peer or an
enclosing Design reads, and it must resolve from semantic facts alone: no
physical Decision, no parameter, no ABI, no source, no target support and no
artifact.  The physical result is the detached build unit, and a Kernel is
permitted to have a perfectly valid Region and an explicitly unsupported
physical realization -- an unavailable target is not a broken Region.

Both projections are synthesized per concrete subclass, because their output is
that subclass's own ``region`` member and the base class has no such
declaration to name.  A subclass therefore writes the parts and gets the
projections: the Region, the physical ``ModuleParameter`` table, and at most two
``ConstraintGroup`` members saying which of its constraints gate which
question.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import wraps
from inspect import Parameter as _SignatureParameter, Signature, signature
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from finn.dataflow._engine import (
    Answer,
    Decided,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DesignPoint,
    Engine,
    EvaluatorSpec,
    QualifiedPath,
)
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.contributions import (
    Contribution,
    CopiedSource,
    DataSlot,
    RenderedSource,
)
from finn.dataflow.artifacts.derivation import Scalar
from finn.dataflow.space.compiler import _CompiledSpace, _Ref
from finn.dataflow.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Problem,
    Projection,
    Readiness,
    Space,
    ValueSource,
    _declaration_name,
    declared_members,
    reject,
    resolve_declared_value,
    semantics_for,
)
from finn.dataflow.space.occurrence import ProjectionAssessment, evaluate_projection
from finn.dataflow.space.dataflow_value_semantics import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.model.region import DataflowRegion, RegionRefused
from finn.dataflow.model.region_validation import validate_region

T = TypeVar("T")
K = TypeVar("K", bound="Kernel")

_MISSING = object()

#: The member names ``Kernel.__init_subclass__`` writes onto every concrete
#: subclass.  A declaration under one of these would be silently replaced, so it
#: is refused in the class body instead -- the same rule and the same reason as
#: the generic reserved names, stated by the layer that owns these seven.
RESERVED_KERNEL_NAMES: frozenset[str] = frozenset(
    {
        "region_structurally_valid",
        "dataflow_accepts",
        "dataflow_ready",
        "dataflow",
        "physical_result",
        "physical_ready",
        "physical",
    }
)


class PhysicallyUnsupported(ValueError):
    """This Kernel cannot realize the configuration its Region already accepts.

    Raised by ``component_abi``, or declared as ``physical_unavailable`` when no
    point of the implementation can supply a standalone module. Its existence
    is the concrete form of the rule
    that a valid dataflow projection does not oblige a valid physical one: an
    implementation that has no wiring for a resolved Region says so here and the
    physical projection becomes a rejecting absence, while the Region carries on
    being exactly what it was.  The alternative -- a dummy ABI over invented
    widths -- would make an unbuildable point look buildable right up until
    synthesis.
    """


@dataclass(frozen=True, slots=True)
class ModuleBuildSpec:
    """The complete detached input to module artifact derivation.

    Detached means what it says: no Engine, no point, no ``_Ref``, no compiled
    record and no attached occurrence.  Everything here is a resolved value, so
    an artifact function cannot reach back into the design space through it, and
    an artifact key built from it cannot accidentally depend on where in a
    namespace tree the Kernel happened to sit.

    ``region`` is carried for witness association -- a physical result must be
    attributable to the exact logical contract it realizes -- and for nothing
    else.  Artifact code reads identity, parameters, ABI and contributions.

    ``imported_decisions`` contains stable root-relative declaration names.
    These strings are provenance, never Engine references or query handles.
    Both scalar mappings are copied and frozen at construction, so a helper
    retaining its render-context dictionary cannot change this spec later.
    """

    implementation_id: str
    implementation_version: str
    region: DataflowRegion
    parameters: Mapping[str, Scalar]
    abi: ComponentABI
    contributions: tuple[Contribution, ...]
    render_context: Mapping[str, Scalar]
    imported_decisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("parameters", "render_context"):
            table = dict(getattr(self, name))
            if any(
                type(key) is not str
                or not (type(value) in (bool, int, float, str) or isinstance(value, Enum))
                for key, value in table.items()
            ):
                raise TypeError(f"ModuleBuildSpec.{name} must map strings to scalars")
            object.__setattr__(self, name, MappingProxyType(table))
        if any(type(name) is not str for name in self.imported_decisions):
            raise TypeError("ModuleBuildSpec.imported_decisions must contain declaration names")
        object.__setattr__(self, "imported_decisions", tuple(self.imported_decisions))
        object.__setattr__(self, "contributions", tuple(self.contributions))


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class RegionDeclaration(Derived[DataflowRegion]):
    """The one canonical Region a Kernel promises to realize.

    A *declaration*, and the name says so.  ``RegionDeclaration`` is the
    authoring recipe -- a family, a version, a constructor and the facts it is
    fed; ``DataflowRegion`` is the detached normalized value that recipe
    produces.  The two used to share the word ``Region``, which made every
    sentence about the lifecycle need a clarifying clause.

    It is an ordinary ``Derived`` that also carries the semantic family it
    belongs to.  The generic decorator is mechanically sufficient, but it
    cannot say *which* compact semantic family produced the resolved value, and
    a family field parked beside a separate ``@derived`` can drift away from the
    value it labels.  Keeping both in one declaration also gives the Kernel
    compiler a single place to run the no-local-Decision dependency audit.

    It defines no second evaluator, wraps no resolved value, and owns no port
    schema: ports, operands, and beat sequences remain fields of the resolved
    ``DataflowRegion``.
    """

    family: str
    version: str
    construct: Callable[..., DataflowRegion]

    def __init__(
        self,
        *,
        family: str,
        version: str,
        construct: Callable[..., DataflowRegion],
        name: str | None = None,
        **dependencies: ValueSource[object],
    ) -> None:
        if not family:
            raise AuthoringError("a Region declaration needs a non-empty family")
        if not version:
            raise AuthoringError("a Region declaration needs a non-empty version")
        if not callable(construct):
            raise AuthoringError("a Region declaration needs a callable constructor")
        _check_constructor(family, construct, tuple(dependencies))

        @wraps(construct)
        def evaluate(**values: object) -> object:
            try:
                return construct(**values)
            except RegionRefused as error:
                # A deliberate refusal is a refusal, not a crash: the facts
                # reached the constructor through Inputs its supplier owns, and
                # the point that supplied them should be told so.  Any other
                # exception is a defect and stays an EvaluationError.
                return reject(
                    "kernel-region-refused",
                    f"{family} cannot be constructed from these facts: {error}",
                    values={"family": family, "version": version},
                )

        object.__setattr__(self, "value_semantics", semantics_for(DATAFLOW_REGION_SEMANTICS))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a Region"))
        object.__setattr__(self, "dependencies", tuple(dependencies.items()))
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "construct", construct)


def _check_constructor(
    family: str,
    construct: Callable[..., DataflowRegion],
    dependencies: tuple[str, ...],
) -> None:
    parameters = signature(construct).parameters
    if any(
        parameter.kind
        in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD, parameter.POSITIONAL_ONLY)
        for parameter in parameters.values()
    ):
        raise AuthoringError(f"the {family} Region constructor must take only named parameters")
    accepted = set(parameters)
    declared = set(dependencies)
    if accepted != declared:
        missing = sorted(declared - accepted)
        extra = sorted(accepted - declared)
        raise AuthoringError(
            f"the {family} Region constructor signature does not match its dependency "
            f"mapping; unused dependencies {missing}, unbound parameters {extra}"
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ModuleParameter(Generic[T]):
    """One scalar physical parameter sourced from a declaration or constant."""

    source: ValueSource[T] | None
    fixed_value: object
    why: str
    stable_name: str | None

    def __init__(self, source: ValueSource[T], *, name: str | None = None) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "fixed_value", _MISSING)
        object.__setattr__(self, "why", "")
        object.__setattr__(self, "stable_name", _declaration_name(name, "a ModuleParameter"))

    @classmethod
    def constant(
        cls,
        value: T,
        *,
        why: str,
        name: str | None = None,
    ) -> ModuleParameter[T]:
        if not why:
            raise AuthoringError("a constant physical parameter must say why it is constant")
        built = object.__new__(cls)
        object.__setattr__(built, "source", None)
        object.__setattr__(built, "fixed_value", value)
        object.__setattr__(built, "why", why)
        object.__setattr__(built, "stable_name", _declaration_name(name, "a ModuleParameter"))
        return built

    def __get__(self, instance: object | None, owner: type[object]) -> object:
        """Class access is the declaration; instance access is the resolved scalar.

        A constant answers from the declaration itself.  A sourced ModuleParameter is
        a *view* on the declaration it names rather than a value of its own, so
        it resolves through the same dispatcher every other value descriptor
        uses -- which is what makes ``kernel.PE`` mean the same thing on an
        attached occurrence as ``kernel.pe`` does.
        """

        if instance is None:
            return self
        if self.source is None:
            return self.fixed_value
        return resolve_declared_value(instance, cast("ValueSource[object]", self.source))


@dataclass(frozen=True)
class _CompiledParameter:
    member_name: str
    physical_name: str
    template: ModuleParameter[object]
    source: ValueSource[object] | None
    constant: object = _MISSING
    why: str = ""


@dataclass(frozen=True)
class _KernelCompilation(Generic[K]):
    """Kernel-only metadata attached to a generic compiled Space.

    Private, and deliberately so: the Decision classification below exists for
    diagnostics and for the review that U2b was asked to make possible, not as
    a capability.  Publishing it would invite a caller to act on it, and the
    only rule that currently acts on it is the authoring refusal.
    """

    owner: type[K]
    kernel_id: str
    kernel_version: str
    region: _Ref[DataflowRegion]
    region_template: RegionDeclaration
    region_family: str
    region_version: str
    parameters: tuple[_CompiledParameter, ...]
    contributions: tuple[Contribution, ...]
    physical_result: _Ref[ModuleBuildSpec]
    #: Local Decisions inside the Region's value/applicability closure.  Always
    #: empty while the ownership refusal stands; kept because "the rule held"
    #: and "the rule was never tested" are different facts.
    region_closure_decisions: tuple[QualifiedPath, ...]
    #: Local Decisions outside that closure: the physical axes this Kernel owns.
    physical_decisions: tuple[QualifiedPath, ...]
    #: Outside Decisions this fragment reads.  Provenance, never ownership.
    imported_decisions: tuple[QualifiedPath, ...]
    #: Compiled paths of the constraints that gate only the *physical*
    #: projection.  An enclosing Design reads this to keep them out of its own
    #: dataflow question: a Kernel that cannot be built here has not thereby
    #: stopped contributing a Region.
    physical_only_constraints: frozenset[QualifiedPath]


class Kernel(Space):
    """One semantic Region and one reusable physical top-module family."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    sources: ClassVar[tuple[Contribution, ...]] = ()
    #: A permanent refusal has no parameter or Decision dependencies. Subclasses
    #: that add an implementation explicitly reset this to None and supply an ABI.
    physical_unavailable: ClassVar[PhysicallyUnsupported | None] = None

    #: The Region is the one automatic Kernel output to a containing Design.
    _implicit_exports = ("region",)

    if TYPE_CHECKING:
        dataflow: Projection[DataflowRegion]
        physical: Projection[ModuleBuildSpec]
        physical_result: Derived[ModuleBuildSpec]
        region_structurally_valid: Constraint
        dataflow_accepts: ConstraintGroup
        dataflow_ready: Readiness
        physical_ready: Readiness

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _synthesize_projections(cls)

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        """The external ABI this Kernel presents at one resolved parameter table.

        Takes the resolved scalars rather than a configured object, because that
        is all an ABI can legitimately read and because an artifact-facing value
        must not be reachable from one.  Raise :class:`PhysicallyUnsupported` to
        say the configuration has no realization here.
        """

        raise NotImplementedError(f"{cls.__name__} does not declare a component ABI")

    @classmethod
    def render_context(
        cls, parameters: Mapping[str, bool | int | float | str]
    ) -> Mapping[str, Scalar]:
        """Flat scalar context for this Kernel's rendered source contributions."""

        del parameters
        return MappingProxyType({})

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        if not isinstance(compiled, _CompiledSpace):
            raise AuthoringError(f"{cls.__name__} received an invalid Space compilation")
        return _finalize_kernel(cls, cast("_CompiledSpace[Kernel]", compiled))


def _parameter_members(
    kernel_type: type[Kernel],
) -> tuple[tuple[str, ModuleParameter[object]], ...]:
    ordered: dict[str, ModuleParameter[object]] = {}
    for base in reversed(kernel_type.__mro__):
        if not issubclass(base, Kernel) or base is Kernel:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, ModuleParameter):
                ordered[name] = cast("ModuleParameter[object]", value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces a ModuleParameter with {type(value).__name__}"
                )
    return tuple(ordered.items())


def _physical_names(
    kernel_type: type[Kernel],
) -> tuple[tuple[str, ModuleParameter[object], str], ...]:
    seen: set[str] = set()
    resolved: list[tuple[str, ModuleParameter[object], str]] = []
    for member_name, template in _parameter_members(kernel_type):
        physical_name = member_name if template.stable_name is None else template.stable_name
        if physical_name in seen:
            raise AuthoringError(
                f"{kernel_type.__name__} declares physical parameter {physical_name!r} twice"
            )
        seen.add(physical_name)
        resolved.append((member_name, template, physical_name))
    return tuple(resolved)


def _region_valid(region: RegionDeclaration) -> Constraint:
    """The one constraint every Kernel gets: its Region is canonically valid.

    Generated rather than asked for.  A Kernel that had to remember to validate
    its own Region would be a Kernel that could forget, and the failure mode --
    a structurally invalid Region travelling into a Network as though it were
    fine -- is exactly the one the canonical validator exists to prevent.
    """

    def evaluate(*, region: DataflowRegion) -> object:
        report = validate_region(region)
        if not report.issues:
            return True
        first = report.issues[0]
        return reject(
            f"kernel-region-{first.code}",
            first.message,
            values={
                "region_path": first.path,
                "issues": tuple(issue.code for issue in report.issues),
            },
        )

    return Constraint((("region", cast("ValueSource[object]", region)),), evaluate)


def _physical_result_property(
    kernel_type: type[Kernel],
    region: RegionDeclaration,
    parameters: tuple[tuple[str, ModuleParameter[object], str], ...],
) -> Derived[ModuleBuildSpec]:
    """Assemble the detached build unit from resolved values and nothing else.

    Every input is an ordinary dependency, so the property is ``Unresolved``
    exactly when one of them is and there is no second notion of "configured".
    The ABI is built here rather than stored on an object because it is a
    function of the resolved parameter table, and a Kernel that says the table
    is unrealizable does so by raising, which becomes a rejecting absence.
    """

    if kernel_type.physical_unavailable is not None:
        # Capture the reason at compilation; even inherited physical parameters,
        # Decisions and support rules are irrelevant to an unavailable module.
        reason = str(kernel_type.physical_unavailable)

        def unavailable() -> object:
            return _physical_refusal(kernel_type.id, reason)

        return Derived(semantics_for(ModuleBuildSpec), None, (), unavailable)

    dependencies: list[tuple[str, ValueSource[object]]] = [
        ("region", cast("ValueSource[object]", region))
    ]
    for member_name, template, _physical_name in parameters:
        if template.source is not None:
            dependencies.append((f"parameter_{member_name}", template.source))

    def evaluate(**values: object) -> object:
        table: dict[str, bool | int | float | str] = {}
        for member_name, template, physical_name in parameters:
            value = (
                template.fixed_value
                if template.source is None
                else values[f"parameter_{member_name}"]
            )
            if type(value) not in (bool, int, float, str):
                raise AuthoringError(
                    f"{kernel_type.__name__} physical parameter {physical_name!r} resolved to "
                    f"{type(value).__name__}, which is not a scalar"
                )
            table[physical_name] = cast("bool | int | float | str", value)
        frozen_table = MappingProxyType(dict(table))
        try:
            abi = kernel_type.component_abi(frozen_table)
        except PhysicallyUnsupported as error:
            return _physical_refusal(kernel_type.id, str(error))
        if not isinstance(abi, ComponentABI):
            raise AuthoringError(
                f"{kernel_type.__name__}.component_abi() did not return ComponentABI"
            )
        expected = tuple(
            sorted(
                (name, str(int(value)) if isinstance(value, bool) else str(value))
                for name, value in frozen_table.items()
            )
        )
        if abi.parameters != expected:
            raise AuthoringError(
                f"{kernel_type.__name__}.component_abi() must expose its exact resolved "
                "physical parameter table"
            )
        return ModuleBuildSpec(
            kernel_type.id,
            kernel_type.version,
            cast(DataflowRegion, values["region"]),
            frozen_table,
            abi,
            tuple(kernel_type.sources),
            kernel_type.render_context(frozen_table),
        )

    # The compiler checks an evaluator's *signature* against its dependency
    # mapping, and rightly refuses a ``**kwargs`` one: a typo in an authored
    # dependency name would otherwise be silently accepted.  This evaluator is
    # generated, and its parameter list is generated from the same tuple the
    # mapping is, so the declared signature is stated explicitly rather than
    # the check being weakened for everyone.
    evaluate.__signature__ = Signature(  # type: ignore[attr-defined]
        [
            _SignatureParameter(name, _SignatureParameter.KEYWORD_ONLY)
            for name, _source in dependencies
        ]
    )
    return Derived(
        semantics_for(ModuleBuildSpec),
        None,
        tuple(dependencies),
        evaluate,
    )


def _physical_refusal(implementation_id: str, reason: str) -> object:
    return reject(
        "kernel-physically-unsupported",
        f"{implementation_id} has no realization for this configuration: {reason}",
        values={"kernel": implementation_id},
    )


def _synthesize_projections(kernel_type: type[Kernel]) -> None:
    """Write the two projections and their parts onto one concrete Kernel class.

    Synthesis rather than authoring, because the output of each projection is
    this class's own ``region`` -- a declaration the base class cannot name --
    and because a Kernel that had to hand-assemble a readiness profile over its
    own parameter table would be hand-assembling something with exactly one
    correct answer.  What the author still owns is the classification: which of
    their constraints gate the Region and which gate the build unit.

    An abstract intermediate Kernel -- one with no ``region`` yet -- is left
    alone.  ``_finalize_kernel`` is where a class that is actually compiled is
    required to be complete, so an incomplete base is a perfectly ordinary
    thing to write and not an error until someone tries to use it.
    """

    shadowed = sorted(RESERVED_KERNEL_NAMES & set(kernel_type.__dict__))
    if shadowed:
        raise AuthoringError(
            f"{kernel_type.__name__} declares {shadowed[0]!r}, which the Kernel layer "
            "synthesizes from the Region, the ModuleParameter table and the support groups"
        )

    unavailable = kernel_type.physical_unavailable
    if unavailable is not None and not isinstance(unavailable, PhysicallyUnsupported):
        raise AuthoringError(
            f"{kernel_type.__name__}.physical_unavailable must be PhysicallyUnsupported or None"
        )

    declarations = dict(declared_members(kernel_type))
    region = declarations.get("region")
    if not isinstance(region, RegionDeclaration):
        return

    parameters = _physical_names(kernel_type)
    decisions = tuple(
        (name, declaration)
        for name, declaration in declarations.items()
        if isinstance(declaration, Decision)
    )

    dataflow_support = declarations.get("dataflow_support")
    physical_support = declarations.get("physical_support")
    for label, group in (
        ("dataflow_support", dataflow_support),
        ("physical_support", physical_support),
    ):
        if group is not None and not isinstance(group, ConstraintGroup):
            raise AuthoringError(
                f"{kernel_type.__name__}.{label} is a {type(group).__name__}; it names one "
                "ConstraintGroup of the constraints that gate that projection"
            )

    region_valid = _region_valid(region)
    dataflow_accepts = ConstraintGroup(
        region_valid,
        *(dataflow_support.constraints if isinstance(dataflow_support, ConstraintGroup) else ()),
    )
    dataflow_ready = Readiness(
        properties=(cast("ValueSource[object]", region),),
        constraints=dataflow_accepts,
    )
    physical_result = _physical_result_property(kernel_type, region, parameters)
    physical_constraints = (
        ()
        if unavailable is not None
        else (
            *dataflow_accepts.constraints,
            *(
                physical_support.constraints
                if isinstance(physical_support, ConstraintGroup)
                else ()
            ),
        )
    )
    physical_ready = Readiness(
        decisions=(
            ()
            if unavailable is not None
            else tuple(declaration for _name, declaration in decisions)
        ),
        properties=(cast("ValueSource[object]", physical_result),),
        constraints=physical_constraints,
    )
    projection_constraints: tuple[ConstraintGroup, ...] = (
        () if unavailable is not None else (dataflow_accepts,)
    )
    if unavailable is None and isinstance(physical_support, ConstraintGroup):
        projection_constraints = (*projection_constraints, physical_support)

    setattr(kernel_type, "region_structurally_valid", region_valid)
    setattr(kernel_type, "dataflow_accepts", dataflow_accepts)
    setattr(kernel_type, "dataflow_ready", dataflow_ready)
    setattr(
        kernel_type,
        "dataflow",
        Projection(
            cast("ValueSource[DataflowRegion]", region),
            readiness=dataflow_ready,
            constraints=(dataflow_accepts,),
            name="dataflow",
        ),
    )
    setattr(kernel_type, "physical_result", physical_result)
    setattr(kernel_type, "physical_ready", physical_ready)
    setattr(
        kernel_type,
        "physical",
        Projection(
            cast("ValueSource[ModuleBuildSpec]", physical_result),
            readiness=physical_ready,
            constraints=projection_constraints,
            name="physical",
        ),
    )


def _region_closure_decisions(
    compiled: _CompiledSpace[K],
    region: _Ref[DataflowRegion],
) -> tuple[QualifiedPath, ...]:
    """Classify local Decisions by whether they can change the Region.

    A Kernel-local Decision is physical only: it may reorganize the hardware,
    never the logical contract a peer or a Design reads.  The mechanical form of
    that rule is the Region property's transitive closure, and it must not reach
    a decision this Kernel declares -- including one nested in a helper ``Space``
    the Kernel uses.  A Decision reached through an ``Input`` belongs to the
    supplier, which is exactly the intended arrangement.

    Applicability counts as reaching.  A gate is not "whether the Region is
    asked for" when the thing holding the gate is inside the Kernel: a local
    Decision that gates a helper whose output feeds the Region makes the Region
    present or absent, which is a change to the logical contract a peer reads and
    not a reorganization of hardware.  So both the value graph and the
    applicability graph are walked.  An *outer* gate -- a Design's segment
    condition or branch selector -- is still fine, because it is not owned here.

    The classification is returned rather than enforced here.  U2b asked for
    the analysis to exist without the ownership rule relaxing, so the two are
    separated: this function computes the set and ``_finalize_kernel`` refuses a
    non-empty one.  A set that is computed and proven empty is a different piece
    of evidence from a set nobody ever computed, and a synthetic local dataflow
    Decision is explicitly *not* an argument for removing the refusal.
    """

    owned = {declaration.path for declaration in compiled.spec.decisions}
    properties = {declaration.path: declaration for declaration in compiled.spec.properties}
    decisions = {declaration.path: declaration for declaration in compiled.spec.decisions}
    reached: list[QualifiedPath] = []
    pending = [region.path]
    visited: set[QualifiedPath] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        edges: list[DependencyRef] = []
        for declaration in (properties.get(path), decisions.get(path)):
            if declaration is None:
                continue
            evaluator = getattr(declaration, "evaluator", None)
            if evaluator is not None:
                edges.extend(evaluator.dependencies)
            domain = getattr(declaration, "domain", None)
            if domain is not None:
                edges.extend(domain.dependencies)
            applies_if = declaration.applies_if
            if applies_if is not None:
                edges.extend(applies_if.dependencies)
        for dependency in edges:
            if dependency.kind is DependencyKind.DECISION and dependency.path in owned:
                reached.append(dependency.path)
            if dependency.kind in (DependencyKind.DECISION, DependencyKind.PROPERTY):
                pending.append(dependency.path)
    return tuple(dict.fromkeys(reached))


def _external_decisions(compiled: _CompiledSpace[K]) -> tuple[QualifiedPath, ...]:
    """Decision paths this fragment reads but does not own.

    Static, unlike the point-filtered form it replaces.  Provenance that changes
    as a point is filled in is provenance a persisted artifact key cannot use;
    "which outside choices can this Kernel's values depend on" is a property of
    the compiled fragment and is the same question anyone actually asks.
    """

    owned = {declaration.path for declaration in compiled.spec.decisions}
    local = {declaration.path for declaration in compiled.spec.properties}
    found: list[QualifiedPath] = []
    for declaration in (
        *compiled.spec.decisions,
        *compiled.spec.properties,
        *compiled.spec.constraints,
    ):
        edges: list[DependencyRef] = []
        evaluator = getattr(declaration, "evaluator", None)
        if evaluator is not None:
            edges.extend(evaluator.dependencies)
        domain = getattr(declaration, "domain", None)
        if domain is not None:
            edges.extend(domain.dependencies)
        applies_if = getattr(declaration, "applies_if", None)
        if applies_if is not None:
            edges.extend(applies_if.dependencies)
        for dependency in edges:
            if dependency.kind is DependencyKind.DECISION and dependency.path not in owned:
                found.append(dependency.path)
    for _name, reference in compiled.inputs:
        if reference.kind is DependencyKind.DECISION and reference.path not in owned:
            found.append(reference.path)
        elif reference.kind is DependencyKind.PROPERTY and reference.path not in local:
            # An Input bound to an outside property is an outside dependency, but
            # the decisions behind it belong to whoever owns that property and
            # are named there; recording the property would confuse the two.
            continue
    return tuple(dict.fromkeys(found))


def _with_provenance(
    specification: EvaluatorSpec[Answer[object]],
    provenance: tuple[str, ...],
) -> EvaluatorSpec[Answer[object]]:
    """Stamp the compiled fragment's import provenance onto the physical result.

    Done here and not in the declaration because the same Kernel class placed at
    two roles reads two different sets of outside paths, and a closure written
    in the class body would have to pretend otherwise.
    """

    inner = specification.evaluator

    def evaluate(values: DependencyView) -> Answer[object]:
        answer = inner(values)
        if isinstance(answer, Decided) and isinstance(answer.value, ModuleBuildSpec):
            return Decided(replace(answer.value, imported_decisions=provenance))
        return answer

    return EvaluatorSpec(specification.dependencies, evaluate)


def _module_provenance(
    compiled: _CompiledSpace[K], paths: tuple[QualifiedPath, ...]
) -> tuple[str, ...]:
    """Translate compiler paths to the same root-relative names persistence uses.

    Input references retain their supplier's root for direct fragment compilation.
    A composed tree shares one root, including a multi-segment occurrence namespace.
    No occurrence or point is needed to name an imported declaration.
    """

    roots = {reference.path: reference.root_namespace for _name, reference in compiled.inputs}
    return tuple(
        str(path).removeprefix(f"{roots.get(path) or compiled.root_namespace}.") for path in paths
    )


def _finalize_kernel(kernel_type: type[K], compiled: _CompiledSpace[K]) -> _CompiledSpace[K]:
    if not kernel_type.id:
        raise AuthoringError(f"{kernel_type.__name__} must declare a non-empty id")
    if not kernel_type.version:
        raise AuthoringError(f"{kernel_type.__name__} must declare a non-empty version")
    abi_owner = next(base for base in kernel_type.__mro__ if "component_abi" in base.__dict__)
    if abi_owner is Kernel and kernel_type.physical_unavailable is None:
        raise AuthoringError(f"{kernel_type.__name__} must declare a component_abi()")

    declarations = dict(declared_members(kernel_type))
    problem_members = tuple(
        name for name, declaration in declarations.items() if isinstance(declaration, Problem)
    )
    if problem_members:
        raise AuthoringError(
            f"{kernel_type.__name__} must consume external facts through Input; "
            f"Kernel-owned Problem members are {problem_members}"
        )
    if kernel_type.exports:
        raise AuthoringError(
            f"{kernel_type.__name__} may not publish exports besides its Region; "
            "a value a peer Kernel needs is a Design-owned semantic fact"
        )
    region_template = declarations.get("region")
    if not isinstance(region_template, RegionDeclaration):
        raise AuthoringError(
            f"{kernel_type.__name__} must declare exactly one Region member named 'region'"
        )
    if region_template.value_semantics.type_token is not DATAFLOW_REGION_SEMANTICS.type_token:
        raise AuthoringError(f"{kernel_type.__name__}.region is not a DataflowRegion")
    _check_constraints_are_classified(kernel_type, declarations)

    region_ref = cast("_Ref[DataflowRegion]", compiled.member("region"))
    region_paths = tuple(
        declaration.path
        for declaration in compiled.spec.properties
        if declaration.value_semantics.type_token is DATAFLOW_REGION_SEMANTICS.type_token
    )
    if region_paths != (region_ref.path,):
        raise AuthoringError(
            f"{kernel_type.__name__} must declare exactly one DataflowRegion; "
            f"compiled Region properties are {tuple(str(path) for path in region_paths)}"
        )
    in_closure = _region_closure_decisions(compiled, region_ref)
    if in_closure:
        raise AuthoringError(
            f"{kernel_type.__name__} lets its own Decision {in_closure[0]} reach "
            f"{region_ref.path}; a choice that changes the Region -- including whether it "
            "applies at all -- belongs to the enclosing Design and arrives as an Input"
        )

    contributions = tuple(kernel_type.sources)
    if any(
        not isinstance(item, (CopiedSource, RenderedSource, DataSlot)) for item in contributions
    ):
        raise AuthoringError(f"{kernel_type.__name__}.sources contains a non-Contribution")

    physical_ref = cast("_Ref[ModuleBuildSpec]", compiled.member("physical_result"))
    physical_only = _physical_only_constraints(declarations, compiled)
    provenance = _external_decisions(compiled)
    specification = replace(
        compiled.spec,
        properties=tuple(
            replace(
                declaration,
                evaluator=_with_provenance(
                    declaration.evaluator, _module_provenance(compiled, provenance)
                ),
            )
            if declaration.path == physical_ref.path
            else declaration
            for declaration in compiled.spec.properties
        ),
    )
    metadata = _KernelCompilation(
        kernel_type,
        kernel_type.id,
        kernel_type.version,
        region_ref,
        region_template,
        region_template.family,
        region_template.version,
        tuple(
            _CompiledParameter(
                member_name,
                physical_name,
                template,
                template.source,
                template.fixed_value,
                template.why,
            )
            for member_name, template, physical_name in _physical_names(kernel_type)
        ),
        contributions,
        physical_ref,
        in_closure,
        tuple(declaration.path for declaration in compiled.spec.decisions),
        provenance,
        physical_only,
    )
    exports = dict(compiled.exports)
    exports.setdefault("region", cast("_Ref[object]", region_ref))
    return replace(
        compiled,
        spec=specification,
        exports=tuple(exports.items()),
        extension=metadata,
    )


def _physical_only_constraints(
    declarations: Mapping[str, object], compiled: _CompiledSpace[K]
) -> frozenset[QualifiedPath]:
    """The compiled paths of the constraints that gate the build unit *alone*.

    "Alone" is the whole content of this function, and it is a **difference**
    rather than a membership test.  An author may put one Constraint in both
    groups -- a folding rule that is simultaneously a semantic requirement and a
    build feasibility one is the ordinary case, not a corner -- and reading only
    ``physical_support`` would classify it as physical-only.  The enclosing
    Design excludes physical-only constraints from its Network question, so that
    reading would silently stop a constraint from gating the Network its author
    explicitly said it gates.  Declaring it twice must *add* a projection, never
    remove one.
    """

    group = declarations.get("physical_support")
    if not isinstance(group, ConstraintGroup):
        return frozenset()
    shared = declarations.get("dataflow_support")
    also_dataflow = (
        {id(item) for item in shared.constraints} if isinstance(shared, ConstraintGroup) else set()
    )
    owned = {id(item) for item in group.constraints} - also_dataflow
    by_member = dict(compiled.constraint_members)
    return frozenset(
        by_member[name]
        for name, declaration in declarations.items()
        if isinstance(declaration, Constraint) and id(declaration) in owned and name in by_member
    )


def _check_constraints_are_classified(
    kernel_type: type[Kernel], declarations: Mapping[str, object]
) -> None:
    """Every authored Constraint gates one projection or the other, explicitly.

    An unclassified constraint is the failure this rule exists to prevent: it
    would be compiled, evaluated, and consulted by nothing, so a Kernel would
    silently stop refusing what its author wrote a refusal for.  Which of the
    two questions it answers is a real decision and is not inferable from what
    it reads -- a physical feasibility constraint and a semantic one routinely
    depend on exactly the same folding facts.
    """

    grouped: set[int] = set()
    for name in ("dataflow_accepts", "dataflow_support", "physical_support"):
        group = declarations.get(name)
        if isinstance(group, ConstraintGroup):
            grouped.update(id(item) for item in group.constraints)
    ungrouped = sorted(
        name
        for name, declaration in declarations.items()
        if isinstance(declaration, Constraint) and id(declaration) not in grouped
    )
    if ungrouped:
        raise AuthoringError(
            f"{kernel_type.__name__} declares Constraint {ungrouped[0]!r} in neither "
            "dataflow_support nor physical_support; a Kernel says which projection each "
            "of its constraints gates, because a constraint in no group refuses nothing"
        )


def kernel_dataflow(
    engine: Engine,
    compiled: _CompiledSpace[K],
    point: DesignPoint,
) -> ProjectionAssessment[DataflowRegion]:
    """Ask one compiled Kernel fragment for its Region at one point.

    The direct-fragment form of ``kernel.dataflow``, for a caller that holds a
    compiled record rather than an attached occurrence -- an enclosing Design
    resolving a selected candidate, or evidence configuring a Kernel from a flat
    engine point.  It runs the same compiled projection and the same reduction.
    """

    return cast(
        "ProjectionAssessment[DataflowRegion]",
        evaluate_projection(engine, point, compiled.projection("dataflow")),
    )


def kernel_physical(
    engine: Engine,
    compiled: _CompiledSpace[K],
    point: DesignPoint,
) -> ProjectionAssessment[ModuleBuildSpec]:
    """Ask one compiled Kernel fragment for its detached build unit at one point."""

    return cast(
        "ProjectionAssessment[ModuleBuildSpec]",
        evaluate_projection(engine, point, compiled.projection("physical")),
    )


__all__ = [
    "Kernel",
    "ModuleBuildSpec",
    "ModuleParameter",
    "PhysicallyUnsupported",
    "RegionDeclaration",
    "kernel_dataflow",
    "kernel_physical",
]
