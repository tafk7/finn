# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Effective-member collection and signature binding without callback execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

from ._bindings import PlacementPlans
from ._configuration import Space
from ._signatures import (
    BoundFunction,
    bind_function,
    output_semantics,
    resolve_annotations,
    resolve_source,
    source_semantics,
)
from .declarations import (
    Constraint,
    Decision,
    Declaration,
    Derived,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
    class_namespace,
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
    }
)


@dataclass(frozen=True, slots=True)
class EffectiveSpace:
    space_type: type[Space]
    members: Mapping[str, Declaration]
    functions: Mapping[str, BoundFunction]
    semantics: Mapping[Declaration, ValueSemantics[object]]
    aliases: Mapping[Declaration, str]
    exports: Mapping[ValueKey[object] | ViewKey[object], Declaration]
    placements: PlacementPlans = field(default_factory=PlacementPlans, repr=False)
    guards: Mapping[Declaration, ValueRef[bool]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _annotation_namespace(space_type: type[Space]) -> dict[str, object]:
    """Resolve local family/base annotations without adding structural members."""
    namespace: dict[str, object] = {base.__name__: base for base in reversed(space_type.__mro__)}
    namespace.update(class_namespace(space_type))
    return namespace


def _function(declaration: Declaration) -> Callable[..., object] | None:
    if isinstance(declaration, (Derived, Constraint, View)):
        return declaration.function
    return None


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
            if isinstance(value, Declaration):
                local_name(name, "declaration name")
                if name in _RESERVED or name.startswith("__"):
                    raise DefinitionError(
                        f"{base.__qualname__}.{name}: reserved configuration name"
                    )
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
                members[name] = value
            elif name in members:
                raise DefinitionError(
                    f"{base.__qualname__}.{name}: ordinary attribute hides an inherited declaration"
                )
    return members, aliases, inherited


def _collect_semantics(
    space_type: type[Space],
    members: Mapping[str, Declaration],
    inherited: Mapping[str, list[Declaration]],
    placements: PlacementPlans,
    hints_for: Callable[[Callable[..., object], str], Mapping[str, object]],
) -> dict[Declaration, ValueSemantics[object]]:
    """Check overrides and infer the local semantics of each effective member."""
    semantics: dict[Declaration, ValueSemantics[object]] = {}
    for name, declaration in members.items():
        owner = f"{space_type.__qualname__}.{name}"
        if isinstance(declaration, Subspace):
            placements.get(declaration)
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
            semantics[declaration] = output_semantics(
                function_decl, hints_for(function, owner), owner
            )
        elif isinstance(declaration, (ValueRef, View)) and declaration.semantics is not None:
            semantics[declaration] = declaration.semantics
        for prior in inherited[name][:-1]:
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
    members, placements, space_type = effective.members, effective.placements, effective.space_type
    guard_candidates: list[tuple[str, Declaration]] = [
        (name, declaration) for name, declaration in members.items()
    ]
    for name, declaration in members.items():
        children: tuple[tuple[str, Subspace[Space]], ...] = ()
        if isinstance(declaration, Subspace):
            children = ((name, declaration),)
        elif isinstance(declaration, SubspaceChoice):
            children = tuple(
                (f"{name}.{case}", placement)
                for case, placement in declaration.alternatives.items()
            )
            guard_candidates.extend(children)
        for placement_name, placement in children:
            plan = placements.get(placement)
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
        guard = resolve_source(declaration.when, effective, owner)
        guard_semantics = source_semantics(guard, effective)
        if guard_semantics is not None and guard_semantics.type_token is not bool:
            raise DefinitionError(f"{owner}: when= requires Boolean value semantics")
        guards[declaration] = cast(ValueRef[bool], guard)
    return MappingProxyType(guards)


def _collect_exports(
    effective: EffectiveSpace, declared_exports: object
) -> Mapping[
    ValueKey[object] | ViewKey[object],
    Declaration,
]:
    space_type, members = effective.space_type, effective.members
    aliases, semantics = effective.aliases, effective.semantics
    exports: dict[ValueKey[object] | ViewKey[object], Declaration] = {}
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
        resolved = members[aliases[declaration]]
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
            exported_semantics = source_semantics(resolved.source, effective)
        if exported_semantics is not None and not key.semantics.is_compatible_with(
            exported_semantics
        ):
            raise DefinitionError(
                f"{space_type.__qualname__}: export {key.name} has incompatible semantics"
            )
        exports[key] = resolved
    return MappingProxyType(exports)


def collect_space(
    space_type: type[Space], *, placements: PlacementPlans | None = None
) -> EffectiveSpace:
    """Collect one scope and snapshot its signatures; child linking is separate."""
    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError("collect_space expects a Space subclass")
    placements = placements if placements is not None else PlacementPlans()
    namespace = class_namespace(space_type)
    annotation_namespace = _annotation_namespace(space_type)
    hint_cache: dict[int, dict[str, object]] = {}

    def hints_for(function: Callable[..., object], owner: str) -> dict[str, object]:
        key = id(function)
        if key not in hint_cache:
            hint_cache[key] = resolve_annotations(function, annotation_namespace, owner)
        return hint_cache[key]

    members, aliases, inherited = _collect_members(space_type)
    semantics = _collect_semantics(space_type, members, inherited, placements, hints_for)
    preliminary = EffectiveSpace(
        space_type,
        MappingProxyType(members),
        MappingProxyType({}),
        MappingProxyType(semantics),
        MappingProxyType(aliases),
        MappingProxyType({}),
        placements=placements,
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
    return replace(
        preliminary,
        functions=MappingProxyType(functions),
        guards=_collect_guards(preliminary),
        exports=_collect_exports(preliminary, namespace.get("exports", {})),
    )
