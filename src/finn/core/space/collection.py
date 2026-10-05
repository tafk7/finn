# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Effective-member collection and signature binding without callback execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType, UnionType
from typing import Union, cast, get_args, get_origin

from ._configuration import Space
from ._nodes import NodeDecision, NodeDecl, is_reference_input, node_record, slot_declaration
from ._signatures import (
    BoundFunction,
    bind_function,
    output_semantics,
    resolve_annotations,
    resolve_source,
    source_semantics,
)
from .declarations import (
    MISSING,
    Constraint,
    Decision,
    Declaration,
    Derived,
    Param,
    ValueRef,
    View,
    ViewKey,
    _describe_formal,
    at,
    class_namespace,
    declared_annotation,
    local_name,
)
from .errors import DefinitionError
from .semantics import ValueSemantics

_RESERVED = frozenset(
    {
        "inspect",
        "view",
        "field",
        "query",
        "root",
        "try_with_choices",
        "with_choices",
        "_state",
        "_scope",
        "exports",
        "when",
        "_space_path",
    }
)


@dataclass(frozen=True, slots=True)
class EffectiveSpace:
    space_type: type[Space]
    members: Mapping[str, Declaration]
    functions: Mapping[str, BoundFunction]
    semantics: Mapping[Declaration, ValueSemantics[object]]
    aliases: Mapping[Declaration, str]
    exports: Mapping[ViewKey[object], Declaration]
    guards: Mapping[Declaration, ValueRef[bool]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    # A per-input export: key -> (reference input name, view) in declared order.
    input_exports: Mapping[ViewKey[object], tuple[tuple[str, Declaration], ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _annotation_namespace(space_type: type[Space]) -> dict[str, object]:
    """Resolve local class and base annotations without adding structural members."""
    namespace: dict[str, object] = {base.__name__: base for base in reversed(space_type.__mro__)}
    namespace.update(class_namespace(space_type))
    return namespace


def _function(declaration: Declaration) -> Callable[..., object] | None:
    if isinstance(declaration, (Derived, Constraint, View)):
        return declaration.function
    return None


def _member_declaration(value: object) -> Declaration | None:
    """The declaration a class attribute contributes: a record for a node or choice."""
    return slot_declaration(value)


def _check_choice_annotation(decision: NodeDecision) -> None:
    """``heating: Boiler | HeatPump = Decision(values=...)``: the union names the candidates."""
    annotation = declared_annotation(decision, "Decision")
    if annotation is MISSING:
        return  # a Decision over nodes persists its key; its annotation only types it
    options = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else ()
    options = options or (annotation,)
    label = _describe_formal(decision, "Decision")
    for key, record in decision.candidates.items():
        if record is None:
            if type(None) not in options:
                raise DefinitionError(
                    f"{label}: candidate {key!r} is None, so the annotation must include None"
                )
        elif not any(
            isinstance(option, type) and issubclass(record.space_type, option) for option in options
        ):
            raise DefinitionError(
                f"{label}: candidate {key!r} is a {record.space_type.__qualname__} node, which the "
                f"annotation {annotation!r} does not admit"
            )


def _collect_members(
    space_type: type[Space],
) -> tuple[
    dict[str, Declaration],
    dict[Declaration, str],
    dict[str, list[Declaration]],
]:
    """Resolve MRO membership and retain declaration aliases for inherited references."""
    members: dict[str, Declaration] = {}
    aliases: dict[Declaration, str] = {}
    inherited: dict[str, list[Declaration]] = {}
    for base in reversed(space_type.__mro__):
        for name, value in vars(base).items():
            declaration = _member_declaration(value)
            if declaration is not None:
                local_name(name, "declaration name")
                if name in _RESERVED or name.startswith("__"):
                    raise DefinitionError(
                        f"{base.__qualname__}.{name}: reserved configuration name"
                    )
                record = node_record(value)
                if record is not None and record.candidate_of is not None:
                    # A handle naming a candidate of a Decision in this body.
                    decision = record.candidate_of
                    if decision.owner is not base:
                        raise DefinitionError(
                            f"{base.__qualname__}.{name}: names a candidate of a Decision "
                            f"that is not declared in {base.__qualname__}{at(record.origin)}"
                        )
                    continue
                if declaration.owner is not base or declaration.name != name:
                    raise DefinitionError(
                        f"{base.__qualname__}.{name}: declaration was not bound at class creation"
                    )
                previous_name = aliases.get(declaration)
                if previous_name is not None and previous_name != name:
                    raise DefinitionError(
                        f"{space_type.__qualname__}.{name}: declaration also named {previous_name}"
                    )
                inherited.setdefault(name, []).append(declaration)
                aliases[declaration] = name
                members[name] = declaration
            elif name in members:
                raise DefinitionError(
                    f"{base.__qualname__}.{name}: ordinary attribute hides an inherited declaration"
                )
    return members, aliases, inherited


def _collect_semantics(
    space_type: type[Space],
    members: Mapping[str, Declaration],
    inherited: Mapping[str, list[Declaration]],
    hints_for: Callable[[Callable[..., object], str], Mapping[str, object]],
) -> dict[Declaration, ValueSemantics[object]]:
    """Check overrides and infer the local semantics of each effective member."""
    semantics: dict[Declaration, ValueSemantics[object]] = {}
    for name, declaration in members.items():
        owner = f"{space_type.__qualname__}.{name}"
        providers = [
            base
            for base in space_type.__mro__
            if _member_declaration(vars(base).get(name)) is not None
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
            semantics[declaration] = output_semantics(
                function_decl, hints_for(function, owner), owner
            )
        elif isinstance(declaration, (ValueRef, View)) and declaration.semantics is not None:
            semantics[declaration] = declaration.semantics
        for prior in inherited[name][:-1]:
            if isinstance(prior, (NodeDecl, NodeDecision)) and isinstance(
                declaration, (NodeDecl, NodeDecision)
            ):
                continue  # a node or choice may be replaced by another structural member
            if type(prior) is not type(declaration):
                raise DefinitionError(f"{owner}: override changes declaration kind")
            old_semantics = getattr(prior, "semantics", None)
            if old_semantics is None and isinstance(prior, (Derived, View)):
                prior_function = prior.function
                if prior_function is not None:
                    old_semantics = output_semantics(prior, hints_for(prior_function, owner), owner)
            current = semantics.get(declaration)
            if (
                isinstance(old_semantics, ValueSemantics)
                and current is not None
                and not old_semantics.is_compatible_with(current)
            ):
                raise DefinitionError(f"{owner}: override changes value semantics")
    return semantics


def _collect_guards(effective: EffectiveSpace) -> Mapping[Declaration, ValueRef[bool]]:
    """Guards authored in this body: members, their candidates and fresh decisions."""
    members, space_type = effective.members, effective.space_type
    guard_candidates: list[tuple[str, Declaration]] = list(members.items())
    for name, declaration in members.items():
        children: tuple[tuple[str, NodeDecl], ...] = ()
        if isinstance(declaration, NodeDecl):
            children = ((name, declaration),)
        elif isinstance(declaration, NodeDecision):
            children = tuple(
                (f"{name}.{case}", record)
                for case, record in declaration.candidates.items()
                if record is not None
            )
            guard_candidates.extend(children)
        for placement_name, record in children:
            guard_candidates.extend(
                (f"{placement_name}.{parameter}", supplier)
                for parameter, supplier in record.bindings.items()
                if isinstance(supplier, Decision) and supplier.owner is None
            )
    guards: dict[Declaration, ValueRef[bool]] = {}
    for name, declaration in guard_candidates:
        if declaration.when is None:
            continue
        owner = f"{space_type.__qualname__}.{name} guard"
        guard = resolve_source(declaration.when, effective, owner)
        guard_semantics = source_semantics(guard, effective)
        if guard_semantics is not None and guard_semantics.type_token is not bool:
            raise DefinitionError(f"{owner}: when= requires Boolean value semantics")
        guards[declaration] = cast(ValueRef[bool], guard)
    return MappingProxyType(guards)


def _exported_view(
    effective: EffectiveSpace, key: ViewKey[object], declaration: object, label: str
) -> View[object]:
    space_type, members = effective.space_type, effective.members
    if not isinstance(declaration, Declaration) or declaration not in effective.aliases:
        raise DefinitionError(f"{space_type.__qualname__}: export {label} is not a member")
    resolved = members[effective.aliases[declaration]]
    if not isinstance(resolved, View):
        raise DefinitionError(f"{space_type.__qualname__}: export {label} has the wrong kind")
    exported_semantics = effective.semantics.get(resolved)
    if exported_semantics is None and resolved.source is not None:
        exported_semantics = source_semantics(resolved.source, effective)
    if exported_semantics is not None and not key.semantics.is_compatible_with(exported_semantics):
        raise DefinitionError(
            f"{space_type.__qualname__}: export {label} has incompatible semantics"
        )
    return resolved


def _collect_exports(
    effective: EffectiveSpace, declared_exports: object
) -> tuple[
    Mapping[ViewKey[object], Declaration],
    Mapping[ViewKey[object], tuple[tuple[str, Declaration], ...]],
]:
    """Plain exports (one view) and per-input exports (one view per reference input)."""
    space_type = effective.space_type
    exports: dict[ViewKey[object], Declaration] = {}
    input_exports: dict[ViewKey[object], tuple[tuple[str, Declaration], ...]] = {}
    if not isinstance(declared_exports, Mapping):
        raise DefinitionError(
            f"{space_type.__qualname__}: exports must map typed keys to declarations"
        )
    export_names: set[str] = set()
    for key, declaration in declared_exports.items():
        if not isinstance(key, ViewKey):
            raise DefinitionError(f"{space_type.__qualname__}: an export needs a ViewKey")
        if key.name in export_names:
            raise DefinitionError(f"{space_type.__qualname__}: duplicate export key {key.name}")
        export_names.add(key.name)
        if not isinstance(declaration, Mapping):
            exports[key] = _exported_view(effective, key, declaration, key.name)
            continue
        entries: list[tuple[str, Declaration]] = []
        for reference, view in declaration.items():
            name = effective.aliases.get(reference) if isinstance(reference, Declaration) else None
            if name is None or not is_reference_input(effective.members[name]):
                label = name or repr(reference)
                raise DefinitionError(
                    f"{space_type.__qualname__}: export {key.name}: {label} is not a "
                    "reference input"
                )
            entries.append((name, _exported_view(effective, key, view, f"{key.name} for {name}")))
        input_exports[key] = tuple(entries)
    return MappingProxyType(exports), MappingProxyType(input_exports)


def collect_space(space_type: type[Space]) -> EffectiveSpace:
    """Collect one scope and snapshot its signatures; child linking is separate."""
    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError("collect_space expects a Space subclass")
    namespace = class_namespace(space_type)
    annotation_namespace = _annotation_namespace(space_type)
    hint_cache: dict[int, dict[str, object]] = {}

    def hints_for(function: Callable[..., object], owner: str) -> dict[str, object]:
        key = id(function)
        if key not in hint_cache:
            hint_cache[key] = resolve_annotations(function, annotation_namespace, owner)
        return hint_cache[key]

    members, aliases, inherited = _collect_members(space_type)
    for declaration in members.values():
        # The annotation is the single source of a member's value type.
        if isinstance(declaration, NodeDecision):
            _check_choice_annotation(declaration)
        elif isinstance(declaration, (Param, Decision)):
            declaration.resolve()
    semantics = _collect_semantics(space_type, members, inherited, hints_for)
    preliminary = EffectiveSpace(
        space_type,
        MappingProxyType(members),
        MappingProxyType({}),
        MappingProxyType(semantics),
        MappingProxyType(aliases),
        MappingProxyType({}),
    )
    functions: dict[str, BoundFunction] = {}
    for name, declaration in members.items():
        if _function(declaration) is not None:
            functions[name] = bind_function(
                cast("Derived[object] | Constraint | View[object]", declaration),
                preliminary,
                name=name,
                annotations=hint_cache[id(_function(declaration))],
                semantics=semantics[declaration],
            )
    for declaration in members.values():
        if isinstance(declaration, View):
            inferred = source_semantics(declaration, preliminary)
            if inferred is not None:
                semantics[declaration] = inferred
    exports, input_exports = _collect_exports(preliminary, namespace.get("exports", {}))
    return replace(
        preliminary,
        functions=MappingProxyType(functions),
        guards=_collect_guards(preliminary),
        exports=exports,
        input_exports=input_exports,
    )
