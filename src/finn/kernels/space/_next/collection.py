# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Effective-member collection and signature binding without callback execution."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Union, cast, get_args, get_origin, get_type_hints

from .declarations import (
    AcceptedViewRef,
    Constraint,
    Declaration,
    Decision,
    DecisionRef,
    Dependency,
    Derived,
    ScopedValueRef,
    Space,
    Subspace,
    SubspaceChoice,
    Param,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
)
from .errors import DefinitionError
from .results import Decided, Inapplicable, MissingInput, NotApplicable, Rejected, Unresolved
from .semantics import ValueSemantics, default_semantics

_RESERVED = frozenset(
    {
        "start",
        "answer",
        "assign",
        "assess",
        "root",
        "edit",
        "refine",
        "decision_state",
        "candidates",
        "_state",
        "_scope",
        "exports",
        "when",
    }
)


@dataclass(frozen=True, slots=True)
class BoundArgument:
    name: str
    source: ValueRef[object]
    mode: Literal["required", "optional", "answer"] = "required"
    annotation: object = object


@dataclass(frozen=True, slots=True)
class BoundFunction:
    function: Callable[..., object]
    dependencies: tuple[BoundArgument, ...]
    return_type: object
    semantics: ValueSemantics[object]


@dataclass(frozen=True, slots=True)
class MemberRecord:
    name: str
    declaration: Declaration
    defined_on: type[object]


@dataclass(frozen=True, slots=True)
class EffectiveSpace:
    space_type: type[Space]
    members: Mapping[str, MemberRecord]
    functions: Mapping[str, BoundFunction]
    semantics: Mapping[Declaration, ValueSemantics[object]]
    aliases: Mapping[Declaration, str]
    exports: Mapping[ValueKey[object] | ViewKey[object], Declaration]
    guards: Mapping[Declaration, ValueRef[bool]] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    parameter: Param[object]
    supplier: object
    kind: Literal["literal", "reference", "exposed-param", "local-decision"]


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    placement: Subspace[Space]
    bindings: Mapping[str, PlacementBinding]


def collect_placement(placement: Subspace[Space]) -> PlacementPlan:
    """Validate one placement's four binding forms without descending into children."""
    namespace = _namespace(placement.space_type)
    parameters = {name: value for name, value in namespace.items() if isinstance(value, Param)}
    label = placement.name or placement.space_type.__qualname__
    missing = parameters.keys() - placement.bindings.keys()
    extra = placement.bindings.keys() - parameters.keys()
    if missing:
        raise DefinitionError(f"{label}: missing child parameter bindings {sorted(missing)}")
    if extra:
        raise DefinitionError(f"{label}: unknown child parameter bindings {sorted(extra)}")
    bindings: dict[str, PlacementBinding] = {}
    for name, parameter in parameters.items():
        supplier = placement.bindings[name]
        kind: Literal["literal", "reference", "exposed-param", "local-decision"]
        if isinstance(supplier, Param) and supplier.owner is None:
            kind = "exposed-param"
        elif isinstance(supplier, Decision) and supplier.owner is None:
            kind = "local-decision"
        elif isinstance(supplier, ValueRef):
            kind = "reference"
        else:
            kind = "literal"
        assert parameter.semantics is not None
        if isinstance(supplier, ValueRef):
            if supplier.semantics is not None and not parameter.semantics.is_compatible_with(
                supplier.semantics
            ):
                raise DefinitionError(f"{label}.{name}: binding has incompatible value semantics")
        else:
            try:
                supplier = parameter.semantics.freeze(supplier)
            except (TypeError, ValueError) as error:
                raise DefinitionError(f"{label}.{name}: {error}") from error
        bindings[name] = PlacementBinding(parameter, supplier, kind)
    return PlacementPlan(placement, MappingProxyType(bindings))


def resolve_decision_ref(reference: DecisionRef[object]) -> Decision[object]:
    """Check local choice ownership, including a fresh Decision bound to a Param."""
    placement = reference.placement
    if not isinstance(placement, Subspace):
        raise DefinitionError("a DecisionRef requires a concrete child placement")
    namespace = _namespace(placement.space_type)
    member = reference.member
    member_name: str | None = None
    for base in placement.space_type.__mro__:
        for name, value in vars(base).items():
            if value is member:
                member_name = name
                break
        if member_name is not None:
            break
    if member_name is None:
        raise DefinitionError("DecisionRef member is not part of this child family")
    effective = namespace[member_name]
    if isinstance(effective, Decision):
        return effective
    if isinstance(effective, Param):
        binding = collect_placement(placement).bindings[member_name]
        if binding.kind == "local-decision" and isinstance(binding.supplier, Decision):
            return binding.supplier
    raise DefinitionError(
        f"{placement.name or placement.space_type.__qualname__}.{member_name}: "
        "a Param alias is not a locally owned Decision"
    )


def _namespace(space_type: type[Space]) -> dict[str, object]:
    namespace: dict[str, object] = {}
    for base in reversed(space_type.__mro__):
        namespace.update(vars(base))
    namespace[space_type.__name__] = space_type
    return namespace


def _hints(
    function: Callable[..., object], namespace: Mapping[str, object], owner: str
) -> dict[str, object]:
    try:
        return get_type_hints(function, localns=namespace)
    except (NameError, TypeError, ValueError) as exc:
        raise DefinitionError(f"{owner}: cannot resolve annotations: {exc}") from exc


def _answer_value_type(annotation: object) -> object | None:
    origin = get_origin(annotation)
    if origin not in (Union, types.UnionType):
        return None
    arguments = get_args(annotation)
    value_types = [get_args(arg)[0] for arg in arguments if get_origin(arg) is Decided]
    if len(value_types) == 1 and all(
        get_origin(arg) is Decided or arg in (Inapplicable, Rejected, Unresolved)
        for arg in arguments
    ):
        return cast(object, value_types[0])
    return None


def _output_semantics(
    declaration: Derived[object] | View[object] | Constraint,
    hints: Mapping[str, object],
    owner: str,
) -> tuple[object, ValueSemantics[object]]:
    annotation = hints.get("return")
    if annotation is None:
        raise DefinitionError(f"{owner}: a return annotation is required")
    answer_type = _answer_value_type(annotation)
    semantics = cast("ValueSemantics[object] | None", declaration.semantics)
    if answer_type is not None and semantics is None:
        raise DefinitionError(f"{owner}: Answer[T] returns require explicit semantics=")
    value_type = answer_type if answer_type is not None else annotation
    if semantics is None:
        if not isinstance(value_type, type):
            raise DefinitionError(f"{owner}: output annotation needs explicit semantics=")
        try:
            semantics = default_semantics(value_type)
        except (TypeError, ValueError) as exc:
            raise DefinitionError(f"{owner}: {exc}") from exc
    if isinstance(value_type, type) and isinstance(semantics.type_token, type):
        if value_type is not semantics.type_token:
            raise DefinitionError(
                f"{owner}: output annotation {value_type.__name__} is incompatible with "
                f"{semantics.name} semantics"
            )
    return annotation, semantics


def _function(declaration: Declaration) -> Callable[..., object] | None:
    if isinstance(declaration, (Derived, Constraint, View)):
        return declaration.function
    return None


def _resolve_source(source: object, effective: EffectiveSpace, owner: str) -> ValueRef[object]:
    if not isinstance(source, ValueRef):
        raise DefinitionError(f"{owner}: dependency must be a value reference")
    name = effective.aliases.get(source)
    if name is not None:
        replacement = effective.members[name].declaration
        if not isinstance(replacement, ValueRef):
            raise DefinitionError(f"{owner}: overridden dependency {name} is not a value")
        return replacement
    if isinstance(source, (ScopedValueRef, AcceptedViewRef)):
        if isinstance(source, DecisionRef):
            resolve_decision_ref(source)
        return source
    raise DefinitionError(f"{owner}: dependency is not declared in this effective scope")


def _source_semantics(
    source: ValueRef[object] | View[object],
    effective: EffectiveSpace,
    known_spaces: Mapping[type[Space], EffectiveSpace] | None = None,
) -> ValueSemantics[object] | None:
    """Read already collected types iteratively; the linker checks unresolved edges."""
    current: Declaration = source
    seen: set[tuple[int, int]] = set()
    while (id(current), id(effective)) not in seen:
        seen.add((id(current), id(effective)))
        if isinstance(current, (ValueRef, View)):
            semantics = effective.semantics.get(current, current.semantics)
            if semantics is not None:
                return semantics
        if isinstance(current, (ScopedValueRef, AcceptedViewRef)):
            placement = current.placement
            if not isinstance(placement, Subspace) or known_spaces is None:
                return None
            child = known_spaces.get(placement.space_type)
            if child is None:
                return None
            member = current.member
            if isinstance(member, (ValueKey, ViewKey)):
                resolved = child.exports.get(member)
            else:
                name = child.aliases.get(member)
                resolved = child.members[name].declaration if name is not None else None
            if resolved is None:
                return None
            current, effective = resolved, child
        elif isinstance(current, View) and current.source is not None:
            current = current.source
        else:
            return None
    return None


def _argument_value_type(argument: BoundArgument, owner: str) -> object:
    annotation = argument.annotation
    if argument.mode == "answer":
        value_type = _answer_value_type(annotation)
        if value_type is None:
            raise DefinitionError(f"{owner}: full_answer dependency requires Answer[T]")
        return value_type
    if argument.mode == "optional":
        options = get_args(annotation)
        if MissingInput not in options or NotApplicable not in options:
            raise DefinitionError(
                f"{owner}: optional dependency annotation must include MissingInput "
                "and NotApplicable"
            )
        value_types = tuple(x for x in options if x not in (MissingInput, NotApplicable))
        return value_types
    return annotation


def _annotation_accepts(annotation: object, value_type: type[object]) -> bool:
    if isinstance(annotation, type):
        return issubclass(value_type, annotation)
    if isinstance(annotation, tuple):
        return any(_annotation_accepts(option, value_type) for option in annotation)
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_annotation_accepts(option, value_type) for option in get_args(annotation))
    if isinstance(origin, type):
        return issubclass(value_type, origin)
    return True


def validate_argument(
    argument: BoundArgument, semantics: ValueSemantics[object], *, owner: str
) -> None:
    """Check one linked supplier using the same dependency-mode policy as collection."""
    value_type = _argument_value_type(argument, owner)
    if isinstance(semantics.type_token, type) and not _annotation_accepts(
        value_type, semantics.type_token
    ):
        raise DefinitionError(
            f"{owner}: annotation {argument.annotation!r} cannot consume {semantics.name}"
        )


def bind_function(
    declaration: Derived[object] | Constraint | View[object],
    effective: EffectiveSpace,
    *,
    name: str,
    namespace: Mapping[str, object] | None = None,
    annotations: Mapping[str, object] | None = None,
    known_spaces: Mapping[type[Space], EffectiveSpace] | None = None,
) -> BoundFunction:
    """Bind all arguments against one complete table, respecting overrides."""
    owner = f"{effective.space_type.__qualname__}.{name}"
    function = declaration.function
    if function is None:
        raise DefinitionError(f"{owner}: value-authored view has no function to bind")
    hints = annotations
    if hints is None:
        hints = _hints(
            function,
            namespace if namespace is not None else _namespace(effective.space_type),
            owner,
        )
    annotation, semantics = _output_semantics(declaration, hints, owner)
    signature = inspect.signature(function)
    extra = declaration.aliases.keys() - signature.parameters.keys()
    if extra:
        raise DefinitionError(f"{owner}: aliases name unknown arguments {sorted(extra)}")
    arguments: list[BoundArgument] = []
    for parameter in signature.parameters.values():
        label = f"{owner} argument {parameter.name}"
        if parameter.name in ("self", "cls"):
            raise DefinitionError(f"{label}: implicit self/cls is unsupported")
        if parameter.name == "when":
            raise DefinitionError(f"{label}: when is a reserved authoring control argument")
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise DefinitionError(
                f"{label}: positional-only and variadic parameters are unsupported"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise DefinitionError(f"{label}: dependency parameters cannot have defaults")
        if parameter.name not in hints:
            raise DefinitionError(f"{label}: a dependency annotation is required")
        source = declaration.aliases.get(parameter.name)
        if parameter.name not in declaration.aliases:
            record = effective.members.get(parameter.name)
            if record is None:
                raise DefinitionError(f"{label}: no declaration with this name")
            source = record.declaration
        mode: Literal["required", "optional", "answer"] = "required"
        if isinstance(source, Dependency):
            mode = source.mode
            source = source.source
        resolved = _resolve_source(source, effective, label)
        source_semantics = _source_semantics(resolved, effective, known_spaces)
        input_type = hints[parameter.name]
        argument = BoundArgument(parameter.name, resolved, mode, input_type)
        if source_semantics is not None:
            validate_argument(argument, source_semantics, owner=label)
        else:
            _argument_value_type(argument, label)
        arguments.append(argument)
    return BoundFunction(function, tuple(arguments), annotation, semantics)


def collect_space(
    space_type: type[Space], *, known_spaces: Mapping[type[Space], EffectiveSpace] | None = None
) -> EffectiveSpace:
    """Collect one scope and snapshot its signatures; child linking is separate."""
    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError("collect_space expects a Space subclass")
    namespace = _namespace(space_type)
    hint_cache: dict[int, dict[str, object]] = {}

    def hints_for(function: Callable[..., object], owner: str) -> dict[str, object]:
        key = id(function)
        if key not in hint_cache:
            hint_cache[key] = _hints(function, namespace, owner)
        return hint_cache[key]

    members: dict[str, MemberRecord] = {}
    aliases: dict[Declaration, str] = {}
    inherited: dict[str, list[Declaration]] = {}
    for base in reversed(space_type.__mro__):
        for name, value in vars(base).items():
            if isinstance(value, Declaration):
                if name in _RESERVED or name.startswith("__"):
                    raise DefinitionError(f"{base.__qualname__}.{name}: reserved occurrence name")
                if value.owner is not base or value.name != name:
                    raise DefinitionError(
                        f"{base.__qualname__}.{name}: declaration was not bound at class creation"
                    )
                previous_name = aliases.get(value)
                if previous_name is not None and previous_name != name:
                    raise DefinitionError(
                        f"{space_type.__qualname__}.{name}: declaration also named {previous_name}"
                    )
                inherited.setdefault(name, []).append(value)
                aliases[value] = name
                members[name] = MemberRecord(name, value, base)
            elif name in members:
                raise DefinitionError(
                    f"{base.__qualname__}.{name}: ordinary attribute hides an inherited declaration"
                )
    semantics: dict[Declaration, ValueSemantics[object]] = {}
    for name, record in members.items():
        declaration = record.declaration
        owner = f"{space_type.__qualname__}.{name}"
        if isinstance(declaration, Subspace):
            collect_placement(declaration)
        providers = [
            base for base in space_type.__mro__ if isinstance(vars(base).get(name), Declaration)
        ]
        nearest = [
            base
            for base in providers
            if not any(other is not base and issubclass(other, base) for other in providers)
        ]
        if len(nearest) > 1:
            names = ", ".join(base.__qualname__ for base in nearest)
            raise DefinitionError(f"{owner}: conflicting inherited declarations from {names}")
        function = _function(declaration)
        if function is not None:
            function_decl = cast("Derived[object] | Constraint | View[object]", declaration)
            semantics[declaration] = _output_semantics(
                function_decl, hints_for(function, owner), owner
            )[1]
        elif isinstance(declaration, (ValueRef, View)) and declaration.semantics is not None:
            semantics[declaration] = declaration.semantics
        for prior in inherited[name][:-1]:
            if type(prior) is not type(declaration):
                raise DefinitionError(f"{owner}: override changes declaration kind")
            old_semantics = getattr(prior, "semantics", None)
            if old_semantics is None and isinstance(prior, (Derived, View)):
                prior_function = prior.function
                if prior_function is not None:
                    old_semantics = _output_semantics(
                        prior, hints_for(prior_function, owner), owner
                    )[1]
            current = semantics.get(declaration)
            if (
                isinstance(old_semantics, ValueSemantics)
                and current is not None
                and not old_semantics.is_compatible_with(current)
            ):
                raise DefinitionError(f"{owner}: override changes value semantics")
    preliminary = EffectiveSpace(
        space_type,
        MappingProxyType(members),
        MappingProxyType({}),
        MappingProxyType(semantics),
        MappingProxyType(aliases),
        MappingProxyType({}),
    )
    functions: dict[str, BoundFunction] = {}
    for name, record in members.items():
        if _function(record.declaration) is not None:
            functions[name] = bind_function(
                cast("Derived[object] | Constraint | View[object]", record.declaration),
                preliminary,
                name=name,
                namespace=namespace,
                annotations=hint_cache[id(_function(record.declaration))],
                known_spaces=known_spaces,
            )
    guard_candidates: list[tuple[str, Declaration]] = [
        (name, record.declaration) for name, record in members.items()
    ]
    for name, record in members.items():
        declaration = record.declaration
        placements: tuple[tuple[str, Subspace[Space]], ...] = ()
        if isinstance(declaration, Subspace):
            placements = ((name, declaration),)
        elif isinstance(declaration, SubspaceChoice):
            placements = tuple(
                (f"{name}.{case}", placement)
                for case, placement in declaration.alternatives.items()
            )
            guard_candidates.extend(placements)
        for placement_name, placement in placements:
            plan = collect_placement(placement)
            guard_candidates.extend(
                (f"{placement_name}.{parameter}", binding.supplier)
                for parameter, binding in plan.bindings.items()
                if binding.kind == "local-decision" and isinstance(binding.supplier, Decision)
            )
    guards: dict[Declaration, ValueRef[bool]] = {}
    for name, declaration in guard_candidates:
        if declaration.when is None:
            continue
        owner = f"{space_type.__qualname__}.{name} guard"
        guard = _resolve_source(declaration.when, preliminary, owner)
        guard_semantics = _source_semantics(guard, preliminary, known_spaces)
        if guard_semantics is not None and guard_semantics.type_token is not bool:
            raise DefinitionError(f"{owner}: when= requires Boolean value semantics")
        guards[declaration] = cast(ValueRef[bool], guard)
    for record in members.values():
        if isinstance(record.declaration, View):
            inferred = _source_semantics(record.declaration, preliminary, known_spaces)
            if inferred is not None:
                semantics[record.declaration] = inferred
    exports: dict[ValueKey[object] | ViewKey[object], Declaration] = {}
    declared_exports = namespace.get("exports", {})
    if not isinstance(declared_exports, Mapping):
        raise DefinitionError(
            f"{space_type.__qualname__}: exports must map typed keys to declarations"
        )
    export_names: set[str] = set()
    for key, declaration in declared_exports.items():
        if not isinstance(key, (ValueKey, ViewKey)):
            raise DefinitionError(f"{space_type.__qualname__}: an export needs a typed key")
        if key.name in export_names:
            raise DefinitionError(f"{space_type.__qualname__}: duplicate export key {key.name}")
        export_names.add(key.name)
        if not isinstance(declaration, Declaration) or declaration not in aliases:
            raise DefinitionError(f"{space_type.__qualname__}: export {key.name} is not a member")
        resolved = members[aliases[declaration]].declaration
        if (isinstance(key, ValueKey) and not isinstance(resolved, ValueRef)) or (
            isinstance(key, ViewKey) and not isinstance(resolved, View)
        ):
            raise DefinitionError(
                f"{space_type.__qualname__}: export {key.name} has the wrong kind"
            )
        exported_semantics = semantics.get(resolved)
        if (
            exported_semantics is None
            and isinstance(resolved, View)
            and resolved.source is not None
        ):
            exported_semantics = _source_semantics(resolved.source, preliminary, known_spaces)
        if exported_semantics is not None and not key.semantics.is_compatible_with(
            exported_semantics
        ):
            raise DefinitionError(
                f"{space_type.__qualname__}: export {key.name} has incompatible semantics"
            )
        exports[key] = resolved
    return EffectiveSpace(
        space_type,
        MappingProxyType(members),
        MappingProxyType(functions),
        MappingProxyType(semantics),
        MappingProxyType(aliases),
        MappingProxyType(exports),
        MappingProxyType(guards),
    )
