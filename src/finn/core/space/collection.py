# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Effective-member collection and signature binding without callback execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType, UnionType
from typing import Union, cast, get_args, get_origin
from weakref import WeakKeyDictionary

from ._configuration import _Namespace
from ._nodes import NodeDecision, NodeDecl, slot_declaration
from ._signatures import (
    BoundFunction,
    bind_function,
    declared_non_values,
    output_semantics,
    resolve_annotations,
    resolve_source,
    source_semantics,
)
from .declarations import (
    MISSING,
    Constraint,
    DecisionDecl,
    Declaration,
    Derived,
    Param,
    ValueRef,
    View,
    class_namespace,
    declared_annotation,
    describe_formal,
    local_name,
)
from .errors import DefinitionError
from .groups import Group
from .results import Rejected
from .semantics import OptionalSemantics, ValueSemantics

_OWNER = "space.collection"

RESERVED_NAMES = frozenset(
    # The names a Space class reserves: no member may take one, nor a name starting
    # ``__`` (``name-reserved``). Public as ``finn.core.space.RESERVED_NAMES``.
    {
        "inspect",
        "view",
        "field",
        "query",
        "root",
        "try_with_choices",
        "with_choices",
        "_held",
        "_scope",
        "when",
        "_space_path",
    }
)


@dataclass(frozen=True, slots=True)
class EffectiveSpace:
    space_type: type[_Namespace]
    members: Mapping[str, Declaration]
    functions: Mapping[str, BoundFunction]
    semantics: Mapping[Declaration, ValueSemantics[object]]
    aliases: Mapping[Declaration, str]
    guards: Mapping[Declaration, ValueRef[bool]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _annotation_namespace(space_type: type[_Namespace]) -> dict[str, object]:
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
    """``heating: Boiler | HeatPump = Decision(domain=...)``: the union names the candidates."""
    annotation = declared_annotation(decision, "Decision")
    if annotation is MISSING:
        return  # a Decision over nodes persists its key; its annotation only types it
    options = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else ()
    options = options or (annotation,)
    label = describe_formal(decision, "Decision")
    for key, record in decision.candidates.items():
        if record is None:
            if type(None) not in options:
                raise DefinitionError.broken(
                    "choice-annotation-excludes-none",
                    _OWNER,
                    f"{label}: case {key!r} is None, so the annotation must include None",
                    declaration=label,
                )
        elif not any(
            isinstance(option, type) and issubclass(record.space_type, option) for option in options
        ):
            raise DefinitionError.broken(
                "choice-annotation-excludes-candidate",
                _OWNER,
                f"{label}: case {key!r} is a {record.space_type.__qualname__} node, which the "
                f"annotation {annotation!r} does not accept",
                declaration=label,
            )


def _collect_members(
    space_type: type[_Namespace],
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
                if name in RESERVED_NAMES or name.startswith("__"):
                    raise DefinitionError.broken(
                        "name-reserved",
                        _OWNER,
                        f"{base.__qualname__}.{name}: reserved configuration name",
                        declaration=f"{base.__qualname__}.{name}",
                    )
                if declaration.owner is not base or declaration.name != name:
                    raise DefinitionError.broken(
                        "declaration-unbound",
                        _OWNER,
                        f"{base.__qualname__}.{name}: declaration was not bound at class creation",
                        declaration=f"{base.__qualname__}.{name}",
                    )
                previous_name = aliases.get(declaration)
                if previous_name is not None and previous_name != name:
                    raise DefinitionError.broken(
                        "declaration-named-twice",
                        _OWNER,
                        f"{space_type.__qualname__}.{name}: declaration also named {previous_name}",
                        declaration=f"{space_type.__qualname__}.{name}",
                    )
                inherited.setdefault(name, []).append(declaration)
                aliases[declaration] = name
                members[name] = declaration
            elif name in members:
                raise DefinitionError.broken(
                    "attribute-hides-declaration",
                    _OWNER,
                    f"{base.__qualname__}.{name}: ordinary attribute hides an inherited "
                    "declaration",
                    declaration=f"{base.__qualname__}.{name}",
                )
    return members, aliases, inherited


def _collect_semantics(
    space_type: type[_Namespace],
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
            raise DefinitionError.broken(
                "inherited-declarations-conflict",
                _OWNER,
                f"{owner}: conflicting inherited declarations from {names}",
                declaration=owner,
            )
        function = _function(declaration)
        if function is not None:
            function_decl = cast("Derived[object] | Constraint | View[object]", declaration)
            semantics[declaration] = output_semantics(
                function_decl, hints_for(function, owner), owner
            )
        elif isinstance(declaration, (ValueRef, View)) and declaration.semantics is not None:
            # a fact's annotation, a value-authored view's source's (a function states its own)
            semantics[declaration] = declaration.semantics
        for prior in inherited[name][:-1]:
            if isinstance(prior, (NodeDecl, NodeDecision)) and isinstance(
                declaration, (NodeDecl, NodeDecision)
            ):
                has_group = (
                    isinstance(prior, NodeDecl) and issubclass(prior.space_type, Group)
                ) or (
                    isinstance(declaration, NodeDecl) and issubclass(declaration.space_type, Group)
                )
                if has_group and not (
                    isinstance(prior, NodeDecl)
                    and isinstance(declaration, NodeDecl)
                    and issubclass(declaration.space_type, prior.space_type)
                ):
                    raise DefinitionError.broken(
                        "override-changes-kind",
                        _OWNER,
                        f"{owner}: an included Group can only be replaced by its own subtype",
                        declaration=owner,
                    )
                continue  # ordinary child and selector replacements keep their existing rules
            if type(prior) is not type(declaration):
                raise DefinitionError.broken(
                    "override-changes-kind",
                    _OWNER,
                    f"{owner}: override changes declaration kind",
                    declaration=owner,
                )
            if isinstance(prior, (Derived, Constraint)):
                _check_answer_within(prior, declaration, owner)
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
                raise DefinitionError.broken(
                    "override-changes-semantics",
                    _OWNER,
                    f"{owner}: override changes value semantics",
                    declaration=owner,
                )
    return semantics


def _check_answer_within(
    prior: Derived[object] | Constraint, declaration: Declaration, owner: str
) -> None:
    """An override of a rule admits no ``Inapplicable`` and no ``Unresolved`` the answer it
    overrides does not (``override-widens-answer``): a reader of that answer relies on it,
    as the Kernel's class checks rely on a ``schedule`` declared never inapplicable, and a
    constraint reading an inapplicable value counts as true. ``Rejected`` is not compared:
    the Kernel's base declares its ``schedule`` ``Inapplicable`` and each Kernel stating one
    declares it ``Rejected``. Each answer is its own declaration's, read in the class that
    declares it (``declared_answer``)."""
    admitted, stated = declared_answer(prior, owner), declared_answer(declaration, owner)
    if admitted is None or stated is None:
        return
    wider = stated - admitted - {Rejected}
    if wider:
        names = " and ".join(sorted(marker.__name__ for marker in wider))
        raise DefinitionError.broken(
            "override-widens-answer",
            _OWNER,
            f"{owner}: its answer admits {names}, which the answer it overrides does not",
            declaration=owner,
        )


_ANSWERS: WeakKeyDictionary[Callable[..., object], dict[type, frozenset[type]]] = (
    WeakKeyDictionary()
)
"""Each rule's declared answer, by its function and the class declaring it: read once."""


def declared_answer(declaration: Declaration, owner: str) -> frozenset[type] | None:
    """The non-values ``declaration``'s return annotation admits (``declared_non_values``),
    its annotation resolved in the class that declares it, never in a subclass's namespace,
    which may give its names other values; None for a declaration with no function."""
    function = _function(declaration)
    declaring = getattr(declaration, "owner", None)
    if function is None or not isinstance(declaring, type):
        return None
    answers = _ANSWERS.setdefault(function, {})
    if declaring not in answers:
        hints = resolve_annotations(function, _annotation_namespace(declaring), owner)
        answers[declaring] = declared_non_values(hints.get("return"))
    return answers[declaring]


def check_overrides(space_type: type[_Namespace]) -> None:
    """Refuse, when ``space_type`` is made, an override of a rule its own body declares that
    widens its base's answer (``_check_answer_within``). An answer whose annotation does not
    resolve yet (a name its module binds further down) is compared when the class is
    collected, which repeats the check and reports the annotation."""
    for name, value in vars(space_type).items():
        declaration = slot_declaration(value)
        if not isinstance(declaration, (Derived, Constraint)):
            continue
        prior = next(
            (
                found
                for base in space_type.__mro__[1:]
                if (found := slot_declaration(vars(base).get(name))) is not None
            ),
            None,
        )
        if type(prior) is not type(declaration):
            continue  # a change of kind is the collection's to refuse
        owner = f"{space_type.__qualname__}.{name}"
        try:
            _check_answer_within(cast("Derived[object] | Constraint", prior), declaration, owner)
        except DefinitionError as error:
            if error.findings and error.findings[0].code == "annotation-unresolved":
                continue
            raise


def _collect_guards(effective: EffectiveSpace) -> Mapping[Declaration, ValueRef[bool]]:
    """Guards authored in this body: members and their cases."""
    members, space_type = effective.members, effective.space_type
    guard_candidates: list[tuple[str, Declaration]] = list(members.items())
    for name, declaration in members.items():
        if isinstance(declaration, NodeDecision):
            guard_candidates.extend(
                (f"{name}.{case}", record)
                for case, record in declaration.candidates.items()
                if record is not None
            )
    guards: dict[Declaration, ValueRef[bool]] = {}
    for name, declaration in guard_candidates:
        if declaration.when is None:
            continue
        owner = f"{space_type.__qualname__}.{name} guard"
        guard = resolve_source(declaration.when, effective, owner)
        guard_semantics = source_semantics(guard, effective)
        if guard_semantics is not None and guard_semantics.type_token is not bool:
            raise DefinitionError.broken(
                "guard-not-boolean",
                _OWNER,
                f"{owner}: when= requires Boolean value semantics",
                declaration=owner,
            )
        if isinstance(guard_semantics, OptionalSemantics):
            raise DefinitionError.broken(
                "guard-not-boolean",
                _OWNER,
                f"{owner}: when= requires a bool, and its value is {guard_semantics.name}, which "
                "admits None: guard on a derived value that tests `is not None` (or the bool's "
                "value) and returns a bool",
                declaration=owner,
            )
        guards[declaration] = cast(ValueRef[bool], guard)
    return MappingProxyType(guards)


def collect_space(space_type: type[_Namespace]) -> EffectiveSpace:
    """Collect one scope and snapshot its signatures; child linking is separate."""
    if not isinstance(space_type, type) or not issubclass(space_type, _Namespace):
        raise DefinitionError.broken(
            "not-a-space", _OWNER, "collect_space expects a Space subclass"
        )
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
        elif isinstance(declaration, (Param, DecisionDecl)):
            declaration.resolve()
    semantics = _collect_semantics(space_type, members, inherited, hints_for)
    preliminary = EffectiveSpace(
        space_type,
        MappingProxyType(members),
        MappingProxyType({}),
        MappingProxyType(semantics),
        MappingProxyType(aliases),
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
    )
