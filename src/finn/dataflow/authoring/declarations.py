# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable class-local declarations for the dataflow adapter compiler.

The public authoring unit is an Op, Design, or Kernel subclass.  Objects in
this module are immutable relative templates stored on those classes.  Binding
creates fresh ``Ref`` values beneath one concrete namespace and never writes a
path or compiler handle back onto the template.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

from finn.dataflow.authoring.op_design import OpDesign, Provenance
from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Ref,
    Scope,
    semantics_for,
)
from finn.dataflow.design import (
    Answer,
    Decided,
    DecisionDomain,
    DependencyView,
    EvaluatorSpec,
    QualifiedPath,
    ValueSemantics,
)

T = TypeVar("T")


class DeclarationLayer(str, Enum):
    """Contributor layer allowed to contain a declaration template."""

    OP = "op"
    DESIGN = "design"
    KERNEL = "kernel"


class DeclarationKind(str, Enum):
    """Stable declaration kinds used for override compatibility."""

    IMPORT = "import"
    PROBLEM = "problem"
    DECISION = "decision"
    PROPERTY = "property"
    CONSTRAINT = "constraint"


ALL_LAYERS = frozenset(DeclarationLayer)


@dataclass(frozen=True, slots=True, eq=False)
class DeclarationTemplate(Generic[T]):
    """Base for one unbound declaration stored in a class dictionary."""

    kind: DeclarationKind
    value_type: type[T] | ValueSemantics[T]
    stable_name: str | None = None
    layers: frozenset[DeclarationLayer] = ALL_LAYERS

    def __post_init__(self) -> None:
        if self.stable_name is not None and not self.stable_name:
            raise ValueError("a declaration stable name must not be empty")
        if not self.layers:
            raise ValueError("a declaration must be available to at least one layer")


class DeclarationGroup:
    """Immutable composite that contributes several leaf declarations."""

    layers: frozenset[DeclarationLayer] = ALL_LAYERS

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True, eq=False)
class Imported(DeclarationTemplate[T]):
    """A typed handle supplied by the enclosing compiler layer."""

    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        *,
        stable_name: str | None = None,
        layers: Iterable[DeclarationLayer] = (
            DeclarationLayer.DESIGN,
            DeclarationLayer.KERNEL,
        ),
    ) -> None:
        object.__setattr__(self, "kind", DeclarationKind.IMPORT)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "stable_name", stable_name)
        object.__setattr__(self, "layers", frozenset(layers))
        DeclarationTemplate.__post_init__(self)


@dataclass(frozen=True, slots=True)
class Condition:
    """Pure Boolean expression over declaration templates."""

    dependencies: tuple[DeclarationTemplate[Any], ...]
    evaluate: Callable[..., bool]


def present(declaration: DeclarationTemplate[Any]) -> Condition:
    """Apply when an optional declaration is present."""

    return Condition((declaration,), lambda value: value is not None)


def not_(condition: DeclarationTemplate[bool] | Condition) -> Condition:
    """Negate a Boolean declaration or condition."""

    if isinstance(condition, DeclarationTemplate):
        return Condition((cast("DeclarationTemplate[Any]", condition),), lambda value: not value)
    return Condition(condition.dependencies, lambda *values: not condition.evaluate(*values))


@dataclass(frozen=True, slots=True)
class FiniteDomain:
    values: tuple[object, ...]

    def __init__(self, values: Iterable[object]) -> None:
        object.__setattr__(self, "values", tuple(values))


@dataclass(frozen=True, slots=True)
class DivisorsDomain:
    extent: DeclarationTemplate[int]


@dataclass(frozen=True, slots=True)
class DependentDomain:
    dependencies: tuple[DeclarationTemplate[Any], ...]
    accepts: Callable[..., bool]
    candidates: Callable[..., tuple[object, ...]] | None = None


DomainTemplate = FiniteDomain | DivisorsDomain | DependentDomain


def finite_values(*values: object) -> FiniteDomain:
    return FiniteDomain(values)


def divisors_of(declaration: DeclarationTemplate[int]) -> DivisorsDomain:
    return DivisorsDomain(declaration)


@dataclass(frozen=True, slots=True, eq=False)
class Problem(DeclarationTemplate[T]):
    """One typed problem field and its projection provenance."""

    provenance: Provenance = Provenance.UPSTREAM
    required: bool = True
    validate: Callable[[object], bool] | None = None
    description: str = ""
    path: QualifiedPath | None = None

    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        *,
        provenance: Provenance,
        stable_name: str | None = None,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
        path: QualifiedPath | str | None = None,
    ) -> None:
        object.__setattr__(self, "kind", DeclarationKind.PROBLEM)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "stable_name", stable_name)
        object.__setattr__(self, "layers", frozenset({DeclarationLayer.OP}))
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "validate", validate)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "path", None if path is None else QualifiedPath.parse(path))
        DeclarationTemplate.__post_init__(self)


@dataclass(frozen=True, slots=True, eq=False)
class Choice(DeclarationTemplate[T]):
    """One engine decision with a class-local domain declaration."""

    domain: DomainTemplate = FiniteDomain(())
    when: DeclarationTemplate[bool] | Condition | None = None

    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: DomainTemplate | Iterable[object],
        stable_name: str | None = None,
        when: DeclarationTemplate[bool] | Condition | None = None,
        layers: Iterable[DeclarationLayer] = ALL_LAYERS,
    ) -> None:
        object.__setattr__(self, "kind", DeclarationKind.DECISION)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "stable_name", stable_name)
        object.__setattr__(self, "layers", frozenset(layers))
        object.__setattr__(
            self,
            "domain",
            domain
            if isinstance(domain, (FiniteDomain, DivisorsDomain, DependentDomain))
            else FiniteDomain(domain),
        )
        object.__setattr__(self, "when", when)
        DeclarationTemplate.__post_init__(self)


@dataclass(frozen=True, slots=True, eq=False)
class Derived(DeclarationTemplate[T]):
    """One derived property backed by a decorated pure method."""

    dependencies: tuple[DeclarationTemplate[Any], ...] = ()
    evaluate: Callable[..., object] = lambda: None
    when: DeclarationTemplate[bool] | Condition | None = None

    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        dependencies: Sequence[DeclarationTemplate[Any]],
        evaluate: Callable[..., object],
        *,
        stable_name: str | None = None,
        when: DeclarationTemplate[bool] | Condition | None = None,
        layers: Iterable[DeclarationLayer] = ALL_LAYERS,
    ) -> None:
        object.__setattr__(self, "kind", DeclarationKind.PROPERTY)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "stable_name", stable_name)
        object.__setattr__(self, "layers", frozenset(layers))
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "when", when)
        DeclarationTemplate.__post_init__(self)


def derived(
    *dependencies: DeclarationTemplate[Any],
    value_type: type[Any] | ValueSemantics[Any] = object,
    stable_name: str | None = None,
    when: DeclarationTemplate[bool] | Condition | None = None,
    layers: Iterable[DeclarationLayer] = ALL_LAYERS,
) -> Callable[[Callable[..., T]], Derived[T]]:
    """Decorate a pure class-body method as a derived declaration."""

    def decorate(function: Callable[..., T]) -> Derived[T]:
        return Derived(
            value_type,
            dependencies,
            function,
            stable_name=stable_name,
            when=when,
            layers=layers,
        )

    return decorate


@dataclass(frozen=True, slots=True, eq=False)
class Rule(DeclarationTemplate[bool]):
    """One constraint backed by a decorated pure method."""

    dependencies: tuple[DeclarationTemplate[Any], ...] = ()
    evaluate: Callable[..., object] = lambda: True
    when: DeclarationTemplate[bool] | Condition | None = None
    sets: tuple[str, ...] = ()

    def __init__(
        self,
        dependencies: Sequence[DeclarationTemplate[Any]],
        evaluate: Callable[..., object],
        *,
        stable_name: str | None = None,
        when: DeclarationTemplate[bool] | Condition | None = None,
        sets: Sequence[str] = (),
        layers: Iterable[DeclarationLayer] = ALL_LAYERS,
    ) -> None:
        object.__setattr__(self, "kind", DeclarationKind.CONSTRAINT)
        object.__setattr__(self, "value_type", bool)
        object.__setattr__(self, "stable_name", stable_name)
        object.__setattr__(self, "layers", frozenset(layers))
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "sets", tuple(sets))
        DeclarationTemplate.__post_init__(self)


def constraint(
    *dependencies: DeclarationTemplate[Any],
    stable_name: str | None = None,
    when: DeclarationTemplate[bool] | Condition | None = None,
    sets: Sequence[str] = (),
    layers: Iterable[DeclarationLayer] = ALL_LAYERS,
) -> Callable[[Callable[..., object]], Rule]:
    """Decorate a pure class-body method as a constraint declaration."""

    def decorate(function: Callable[..., object]) -> Rule:
        return Rule(
            dependencies,
            function,
            stable_name=stable_name,
            when=when,
            sets=sets,
            layers=layers,
        )

    return decorate


@dataclass(frozen=True, slots=True)
class Readiness:
    """One class-local readiness-profile declaration."""

    name: str
    decisions: tuple[DeclarationTemplate[Any], ...] = ()
    properties: tuple[DeclarationTemplate[Any], ...] = ()
    constraints: tuple[Rule, ...] = ()
    layers: frozenset[DeclarationLayer] = ALL_LAYERS


@dataclass(frozen=True, slots=True)
class CollectedDeclaration:
    member_name: str
    stable_name: str
    template: DeclarationTemplate[Any]
    declaring_class: type[object]


@dataclass(frozen=True, slots=True)
class CompiledClassDeclarations:
    """One fresh binding of immutable templates beneath a namespace."""

    owner: type[object]
    layer: DeclarationLayer
    namespace: str
    scope: Scope
    members: Mapping[str, Ref[object] | ConstraintRef]
    groups: Mapping[str, DeclarationGroup]
    template_members: Mapping[int, str]
    declaring_classes: Mapping[str, type[object]]

    def ref(self, member_name: str) -> Ref[object]:
        try:
            value = self.members[member_name]
        except KeyError:
            raise AuthoringError(
                f"{self.owner.__name__} declares no member {member_name!r}"
            ) from None
        if not isinstance(value, Ref):
            raise AuthoringError(f"{self.owner.__name__}.{member_name} is a constraint")
        return value

    def constraint(self, member_name: str) -> ConstraintRef:
        try:
            value = self.members[member_name]
        except KeyError:
            raise AuthoringError(
                f"{self.owner.__name__} declares no member {member_name!r}"
            ) from None
        if not isinstance(value, ConstraintRef):
            raise AuthoringError(f"{self.owner.__name__}.{member_name} is not a constraint")
        return value


def _value_signature(template: DeclarationTemplate[Any]) -> tuple[object, str]:
    semantics = semantics_for(template.value_type)
    return semantics.type_token, semantics.name


def _collect_class_declarations(
    owner: type[object], layer: DeclarationLayer
) -> tuple[
    tuple[CollectedDeclaration, ...],
    Mapping[int, str],
    Mapping[str, DeclarationGroup],
]:
    """Collect effective declarations in deterministic base-to-leaf order."""

    effective: OrderedDict[
        str, tuple[DeclarationTemplate[Any] | DeclarationGroup, type[object]]
    ] = OrderedDict()
    aliases: dict[int, str] = {}
    for declaring_class in reversed(owner.__mro__):
        if declaring_class is object:
            continue
        for member_name, value in declaring_class.__dict__.items():
            previous = effective.get(member_name)
            if previous is not None and not isinstance(
                value, (DeclarationTemplate, DeclarationGroup)
            ):
                raise AuthoringError(
                    f"{declaring_class.__name__}.{member_name} replaces a declaration with "
                    f"{type(value).__name__}"
                )
            if not isinstance(value, (DeclarationTemplate, DeclarationGroup)):
                continue
            template_or_group = value
            if layer not in template_or_group.layers:
                raise AuthoringError(
                    f"{declaring_class.__name__}.{member_name} is unavailable on the "
                    f"{layer.value} layer"
                )
            if previous is not None:
                old, old_class = previous
                compatible = (
                    isinstance(old, DeclarationTemplate)
                    and isinstance(template_or_group, DeclarationTemplate)
                    and old.kind is template_or_group.kind
                ) or (
                    isinstance(old, DeclarationGroup)
                    and isinstance(template_or_group, DeclarationGroup)
                    and type(old) is type(template_or_group)
                )
                if not compatible:
                    old_kind = (
                        old.kind.value
                        if isinstance(old, DeclarationTemplate)
                        else type(old).__name__
                    )
                    new_kind = (
                        template_or_group.kind.value
                        if isinstance(template_or_group, DeclarationTemplate)
                        else type(template_or_group).__name__
                    )
                    raise AuthoringError(
                        f"{declaring_class.__name__}.{member_name} cannot override "
                        f"{old_class.__name__}.{member_name}: {old_kind} and "
                        f"{new_kind} are incompatible declaration kinds"
                    )
                if (
                    isinstance(old, DeclarationTemplate)
                    and isinstance(template_or_group, DeclarationTemplate)
                    and _value_signature(old) != _value_signature(template_or_group)
                ):
                    raise AuthoringError(
                        f"{declaring_class.__name__}.{member_name} cannot override "
                        f"{old_class.__name__}.{member_name} with incompatible value semantics"
                    )
                if isinstance(old, DeclarationTemplate):
                    aliases[id(old)] = member_name
                else:
                    for child_name, child in old.declaration_items(member_name):
                        aliases[id(child)] = child_name
            effective[member_name] = (template_or_group, declaring_class)
            if isinstance(template_or_group, DeclarationTemplate):
                aliases[id(template_or_group)] = member_name
            else:
                for child_name, child in template_or_group.declaration_items(member_name):
                    aliases[id(child)] = child_name

    stable_names: dict[str, str] = {}
    collected: list[CollectedDeclaration] = []
    groups: dict[str, DeclarationGroup] = {}
    for member_name, (template_or_group, declaring_class) in effective.items():
        leaves = (
            ((member_name, template_or_group),)
            if isinstance(template_or_group, DeclarationTemplate)
            else template_or_group.declaration_items(member_name)
        )
        if isinstance(template_or_group, DeclarationGroup):
            groups[member_name] = template_or_group
        for leaf_name, template in leaves:
            stable_name = template.stable_name or leaf_name
            previous_member = stable_names.get(stable_name)
            if previous_member is not None and previous_member != leaf_name:
                raise AuthoringError(
                    f"{owner.__name__}.{leaf_name} and {owner.__name__}.{previous_member} "
                    f"both declare stable name {stable_name!r}"
                )
            stable_names[stable_name] = leaf_name
            collected.append(
                CollectedDeclaration(leaf_name, stable_name, template, declaring_class)
            )

    return (
        tuple(collected),
        MappingProxyType(dict(aliases)),
        MappingProxyType(groups),
    )


def collect_class_declarations(
    owner: type[object], layer: DeclarationLayer
) -> tuple[CollectedDeclaration, ...]:
    """Collect effective declarations without retaining compiler state."""

    declarations, _aliases, _groups = _collect_class_declarations(owner, layer)
    return declarations


def _parameter_names(function: Callable[..., object], count: int) -> tuple[str, ...]:
    parameters = tuple(signature(function).parameters.values())
    if len(parameters) != count or any(
        parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        for parameter in parameters
    ):
        raise AuthoringError(
            f"{getattr(function, '__qualname__', function)} must declare exactly {count} "
            "named dependency parameters"
        )
    return tuple(parameter.name for parameter in parameters)


def _condition_spec(
    condition: DeclarationTemplate[bool] | Condition | None,
    *,
    aliases: Mapping[int, str],
    bound: Mapping[str, Ref[object] | ConstraintRef],
) -> EvaluatorSpec[Answer[bool]] | None:
    if condition is None:
        return None
    expression = (
        Condition((cast("DeclarationTemplate[Any]", condition),), lambda value: bool(value))
        if isinstance(condition, DeclarationTemplate)
        else condition
    )
    refs = tuple(
        _resolve_ref(item, aliases=aliases, bound=bound) for item in expression.dependencies
    )
    names = tuple(f"value_{index}" for index in range(len(refs)))

    def evaluate_condition(view: DependencyView) -> Answer[bool]:
        return Decided(expression.evaluate(*(view[name] for name in names)))

    return EvaluatorSpec(
        tuple(ref.dependency(name) for name, ref in zip(names, refs)),
        evaluate_condition,
    )


def _resolve_ref(
    template: DeclarationTemplate[Any],
    *,
    aliases: Mapping[int, str],
    bound: Mapping[str, Ref[object] | ConstraintRef],
) -> Ref[object]:
    member_name = aliases.get(id(template))
    if member_name is None:
        raise AuthoringError("a declaration dependency is not a member of the compiled class")
    value = bound.get(member_name)
    if not isinstance(value, Ref):
        raise AuthoringError(f"dependency {member_name!r} is not a value declaration")
    return value


def _domain(
    template: DomainTemplate,
    *,
    aliases: Mapping[int, str],
    bound: Mapping[str, Ref[object] | ConstraintRef],
) -> Callable[[QualifiedPath], DecisionDomain]:
    if isinstance(template, FiniteDomain):
        ordered = template.values
        allowed = frozenset(ordered)

        def finite_domain(_owner: QualifiedPath) -> DecisionDomain:
            return DecisionDomain(
                (),
                lambda value, _view: Decided(value in allowed),
                EvaluatorSpec((), lambda _view: Decided(ordered)),
            )

        return finite_domain
    if isinstance(template, DivisorsDomain):
        extent = _resolve_ref(
            cast("DeclarationTemplate[Any]", template.extent), aliases=aliases, bound=bound
        )

        def divisors_domain(_owner: QualifiedPath) -> DecisionDomain:
            dependency = extent.dependency("extent")

            def accepts(value: object, view: DependencyView) -> Answer[bool]:
                limit = cast(int, view["extent"])
                return Decided(type(value) is int and value > 0 and limit % value == 0)

            def candidates(view: DependencyView) -> Answer[tuple[object, ...]]:
                limit = cast(int, view["extent"])
                return Decided(tuple(value for value in range(1, limit + 1) if limit % value == 0))

            return DecisionDomain(
                (dependency,),
                accepts,
                EvaluatorSpec((dependency,), candidates),
            )

        return divisors_domain

    refs = tuple(_resolve_ref(item, aliases=aliases, bound=bound) for item in template.dependencies)
    accepts_names = _parameter_names(template.accepts, len(refs) + 1)
    candidate_name = accepts_names[0]
    dependency_names = accepts_names[1:]
    if candidate_name != "candidate":
        raise AuthoringError(
            "a dependent-domain accepts method must name its first parameter candidate"
        )
    if template.candidates is not None:
        candidate_names = _parameter_names(template.candidates, len(refs))
        if candidate_names != dependency_names:
            raise AuthoringError("dependent-domain accepts and candidates parameters must agree")

    def dependent_domain(_owner: QualifiedPath) -> DecisionDomain:
        dependency_refs = tuple(ref.dependency(name) for name, ref in zip(dependency_names, refs))

        def accepts(value: object, view: DependencyView) -> Answer[bool]:
            return Decided(
                bool(template.accepts(value, *(view[name] for name in dependency_names)))
            )

        candidates_spec: EvaluatorSpec[Answer[tuple[object, ...]]] | None = None
        if template.candidates is not None:

            def candidates(view: DependencyView) -> Answer[tuple[object, ...]]:
                assert template.candidates is not None
                return Decided(
                    tuple(template.candidates(*(view[name] for name in dependency_names)))
                )

            candidates_spec = EvaluatorSpec(dependency_refs, candidates)
        return DecisionDomain(dependency_refs, accepts, candidates_spec)

    return dependent_domain


def compile_class_declarations(
    owner: type[object],
    *,
    layer: DeclarationLayer,
    namespace: str,
    imports: Mapping[str, Ref[object]] | None = None,
) -> CompiledClassDeclarations:
    """Bind one class's immutable declaration templates under ``namespace``."""

    declarations, aliases, groups = _collect_class_declarations(owner, layer)
    scope: Scope = OpDesign(namespace) if layer is DeclarationLayer.OP else Scope(namespace)
    bound: OrderedDict[str, Ref[object] | ConstraintRef] = OrderedDict()
    imported = imports or {}

    for item in declarations:
        template = item.template
        if isinstance(template, Imported):
            try:
                supplied = imported[item.member_name]
            except KeyError:
                raise AuthoringError(
                    f"{owner.__name__}.{item.member_name} requires an imported handle"
                ) from None
            if supplied.semantics.type_token is not semantics_for(template.value_type).type_token:
                raise AuthoringError(
                    f"{owner.__name__}.{item.member_name} received incompatible imported semantics"
                )
            bound[item.member_name] = supplied
        elif isinstance(template, Problem):
            assert isinstance(scope, OpDesign)
            bound[item.member_name] = scope.fact(
                item.stable_name,
                template.value_type,
                provenance=template.provenance,
                path=template.path,
                required=template.required,
                validate=template.validate,
                description=template.description,
            )

    for item in declarations:
        template = item.template
        if isinstance(template, Choice):
            bound[item.member_name] = scope.decision(
                item.stable_name,
                template.value_type,
                domain=_domain(template.domain, aliases=aliases, bound=bound),
                applies_if=_condition_spec(template.when, aliases=aliases, bound=bound),
            )

    for item in declarations:
        template = item.template
        if isinstance(template, Derived):
            names = _parameter_names(template.evaluate, len(template.dependencies))
            dependencies = {
                name: _resolve_ref(dependency, aliases=aliases, bound=bound)
                for name, dependency in zip(names, template.dependencies)
            }
            bound[item.member_name] = scope.derived(
                item.stable_name,
                template.value_type,
                dependencies=dependencies,
                evaluate=template.evaluate,
                applies_if=_condition_spec(template.when, aliases=aliases, bound=bound),
            )

    for item in declarations:
        template = item.template
        if isinstance(template, Rule):
            names = _parameter_names(template.evaluate, len(template.dependencies))
            dependencies = {
                name: _resolve_ref(dependency, aliases=aliases, bound=bound)
                for name, dependency in zip(names, template.dependencies)
            }
            bound[item.member_name] = scope.constraint(
                item.stable_name,
                dependencies=dependencies,
                evaluate=template.evaluate,
                applies_if=_condition_spec(template.when, aliases=aliases, bound=bound),
                sets=template.sets,
            )

    for declaring_class in reversed(owner.__mro__):
        if declaring_class is object:
            continue
        for member_name, value in declaring_class.__dict__.items():
            if not isinstance(value, Readiness):
                continue
            if layer not in value.layers:
                raise AuthoringError(
                    f"{declaring_class.__name__}.{member_name} is unavailable on the "
                    f"{layer.value} layer"
                )
            scope.readiness_profile(
                value.name,
                decisions=tuple(
                    _resolve_ref(item, aliases=aliases, bound=bound) for item in value.decisions
                ),
                properties=tuple(
                    _resolve_ref(item, aliases=aliases, bound=bound) for item in value.properties
                ),
                constraints=tuple(
                    cast(
                        ConstraintRef,
                        bound[aliases[id(cast("DeclarationTemplate[Any]", item))]],
                    )
                    for item in value.constraints
                ),
            )

    return CompiledClassDeclarations(
        owner,
        layer,
        namespace,
        scope,
        MappingProxyType(dict(bound)),
        groups,
        aliases,
        MappingProxyType({item.member_name: item.declaring_class for item in declarations}),
    )


__all__ = [
    "ALL_LAYERS",
    "Choice",
    "CollectedDeclaration",
    "CompiledClassDeclarations",
    "Condition",
    "DeclarationKind",
    "DeclarationLayer",
    "DeclarationGroup",
    "DeclarationTemplate",
    "DependentDomain",
    "Derived",
    "DivisorsDomain",
    "FiniteDomain",
    "Imported",
    "Problem",
    "Readiness",
    "Rule",
    "collect_class_declarations",
    "compile_class_declarations",
    "constraint",
    "derived",
    "divisors_of",
    "finite_values",
    "not_",
    "present",
]
