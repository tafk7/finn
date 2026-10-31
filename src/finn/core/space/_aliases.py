# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Node aliases: the table of the second names a Space class gives its members' nodes.

``x = inner.leaf``, ``x = compute.port`` and ``x = mm.x`` in a class body are node aliases
(``_nodes.NodeAlias``): a name, in the class that holds it, for a node a member places, at
any depth below it. The node keeps its owner; the alias owns nothing and places nothing,
so the ownership of the nodes stays a tree, and a design space's keys, cache and stored
choices are the owner's.

Each class's aliases are resolved once, by declaration, into one frozen table
(``alias_table``, also the compiled form's ``aliases``) that every lookup reads:

- a class reference (``Kind.x``, ``mm.x`` in a body) is the node by its path of
  declarations, or through a selector a ``ChoicePath`` (``alias_face``), labelled with the
  names it was read through, which a reference below it reports beside its own path;
- a read on a design space or a receiver walks the alias's ``steps`` (``alias_node``):
  ``point.x`` is the node, or ``None`` where a selection places nothing there;
- a method's read inference expands ``self.x`` into its ``steps`` (``_reads``), and a
  read below a selector reaches the selected case's node (``ChoiceMemberRef`` of a path);
- presence and a node query read the node it reaches (``_running.surface``,
  ``_running.receiver``).

Resolution is by declaration: each declaration on the path is the holding class's own or
inherited member (or of the class the one before places), read where it is placed; a
subclass overriding one of them refuses the alias it inherits (``alias-target-absent``),
never retargeting it by its name. Below a selector a path is read by the names
every case declares: a node of each case that places something (a case's own alias, or a
selector of it, read through), refused where a case lacks it (``alias-case-absent``); a case
placing nothing makes the alias absent there (``optional``), read as ``None`` and its reads
inapplicable, like a guarded member's. What it names last is a node (a child, or another
alias), never a selector or a value (``alias-target-not-node``).
"""

# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, cast

from ._nodes import (
    ChoicePath,
    NodeAlias,
    NodeDecision,
    NodeDecl,
    path_proxy,
    slot_declaration,
    space_members,
)
from .declarations import Declaration
from .errors import DefinitionError

if TYPE_CHECKING:
    from ._configuration import _Namespace

_OWNER = "space.aliases"


@dataclass(frozen=True, slots=True)
class AliasEnd:
    """A node an alias may name: its path from the class holding the alias, the owner's
    (``compute.packed.x``: its scope's name there), and its class."""

    path: str
    space_type: type[_Namespace]


@dataclass(frozen=True, slots=True)
class AliasEntry:
    """One node alias of a class, resolved: its name, its declaration, the names it is read
    through from the class (``steps``: children, a selector, then the names below it, a
    case's own alias read through in turn), every node it may name (``ends``, one per
    selection) and whether some selection places nothing there (``optional``)."""

    name: str
    declaration: NodeAlias
    steps: tuple[str, ...]
    ends: tuple[AliasEnd, ...]
    optional: bool


def alias_table(space_type: type[_Namespace]) -> Mapping[str, AliasEntry]:
    """The node aliases of a Space class, by name, resolved once and frozen (cached on the
    class, whose structure is final when it is created). A path that does not resolve is a
    ``DefinitionError`` here, and so when the class is compiled."""
    cached = space_type.__dict__.get("_space_aliases")
    if cached is not None:
        return cast("Mapping[str, AliasEntry]", cached)
    table: dict[str, AliasEntry] = {}
    for name, member in space_members(space_type).items():
        if isinstance(member, NodeAlias):
            table[name] = _resolve(space_type, name, member)
    frozen = MappingProxyType(table)
    type.__setattr__(space_type, "_space_aliases", frozen)
    return frozen


def _refuse(code: str, where: str, message: str) -> NoReturn:
    raise DefinitionError.broken(code, _OWNER, f"{where}: {message}", declaration=where)


def _placed(space_type: type[_Namespace], record: Declaration, where: str) -> Declaration:
    """``record``, a member of ``space_type`` where it is placed: an alias is resolved by
    the declaration it names, never retargeted by its name to a subclass's override."""
    name = cast(str, record.name)
    member = space_members(space_type).get(name)
    if member is None or slot_declaration(member) is not record:
        overridden = member is not None and any(
            slot_declaration(vars(base).get(name)) is record for base in space_type.__mro__
        )
        _refuse(
            "alias-target-absent",
            where,
            f"{name} of {space_type.__qualname__} is overridden: an alias names the "
            "declaration it was written against; declare the alias again beside the override"
            if overridden
            else f"{record.describe() if hasattr(record, 'describe') else name} is not a "
            f"member of {space_type.__qualname__}",
        )
    return member


def _resolve(space_type: type[_Namespace], name: str, alias: NodeAlias) -> AliasEntry:
    where = f"{space_type.__qualname__}.{name} (an alias of {alias.spelled})"
    current = space_type
    steps: list[str] = []
    path: list[str] = []
    for record in alias.target:
        member = _placed(current, record, where)
        steps.append(cast(str, member.name))
        if isinstance(member, NodeDecision):
            ends, optional = _cases(member, alias.below, tuple(path), where)
            return AliasEntry(name, alias, (*steps, *alias.below), tuple(ends), optional)
        path.append(cast(str, member.name))
        current = cast(NodeDecl, member).space_type
    end = AliasEnd(".".join(path), current)
    return AliasEntry(name, alias, tuple(steps), (end,), False)


def _cases(
    decision: NodeDecision, names: tuple[str, ...], path: tuple[str, ...], where: str
) -> tuple[list[AliasEnd], bool]:
    """The nodes ``names`` reach in each case of ``decision`` (its path ``path``)."""
    ends: list[AliasEnd] = []
    optional = False
    for case, record in decision.candidates.items():
        if record is None:
            optional = True
            continue
        found, absent = _named(
            record.space_type,
            names,
            (*path, cast(str, decision.name), case),
            f"{where}, case {case!r}",
        )
        ends.extend(found)
        optional = optional or absent
    return ends, optional


def _named(
    space_type: type[_Namespace], names: tuple[str, ...], path: tuple[str, ...], where: str
) -> tuple[list[AliasEnd], bool]:
    """The nodes the names ``names`` reach from ``space_type`` (at ``path``)."""
    name, rest = names[0], names[1:]
    member = space_members(space_type).get(name)
    if isinstance(member, NodeDecl):
        if rest:
            return _named(member.space_type, rest, (*path, name), where)
        return [AliasEnd(".".join((*path, name)), member.space_type)], False
    if isinstance(member, NodeAlias):
        entry = alias_table(space_type)[name]
        ends: list[AliasEnd] = []
        optional = entry.optional
        for end in entry.ends:
            reached = (*path, *end.path.split("."))
            if rest:
                found, absent = _named(end.space_type, rest, reached, where)
                ends.extend(found)
                optional = optional or absent
            else:
                ends.append(AliasEnd(".".join(reached), end.space_type))
        return ends, optional
    if isinstance(member, NodeDecision):
        if not rest:
            _refuse(
                "alias-target-not-node",
                where,
                f"{name} of {space_type.__qualname__} is a selector: alias a node of its "
                "cases, or the selector's own member where it is declared",
            )
        return _cases(member, rest, path, where)
    if member is None:
        _refuse(
            "alias-case-absent",
            where,
            f"{space_type.__qualname__} holds no node {name!r}: a node read through a "
            "selector is a node of every case that places something",
        )
    _refuse(
        "alias-target-not-node",
        where,
        f"{name} of {space_type.__qualname__} is a {type(member).__name__}, not a node: an "
        "alias names a node; read a value through its path",
    )


def reached(decision: NodeDecision, names: tuple[str, ...]) -> dict[str, type[_Namespace]]:
    """The classes ``names`` reach in each case of ``decision`` that places something, by
    the case (and the path below it, where the names pass another selector)."""
    found: dict[str, type[_Namespace]] = {}
    for case, record in decision.candidates.items():
        if record is not None:
            found.update(_walk(record.space_type, names, case))
    return found


def _walk(
    space_type: type[_Namespace], names: tuple[str, ...], where: str
) -> dict[str, type[_Namespace]]:
    if not names:
        return {where: space_type}
    name, rest = names[0], names[1:]
    member = space_members(space_type).get(name)
    found: dict[str, type[_Namespace]] = {}
    if isinstance(member, NodeDecl):
        found.update(_walk(member.space_type, rest, f"{where}.{name}"))
    elif isinstance(member, NodeAlias):
        for end in alias_table(space_type)[name].ends:
            found.update(_walk(end.space_type, rest, f"{where}.{end.path}"))
    elif isinstance(member, NodeDecision):
        for case, record in member.candidates.items():
            if record is not None:
                found.update(_walk(record.space_type, rest, f"{where}.{name}.{case}"))
    return found


def alias_face(
    space_type: type[_Namespace],
    name: str,
    outer: tuple[Declaration, ...],
    label: tuple[str, ...],
) -> Any:
    """The class reference of alias ``name`` of ``space_type``, below the node path
    ``outer``: the node by its declarations, or through a selector a ``ChoicePath``,
    labelled with the names it was read through (``label``, then ``name``)."""
    entry = alias_table(space_type)[name]
    current = space_type
    records: list[Declaration] = []
    for index, step in enumerate(entry.steps):
        member = space_members(current)[step]
        if isinstance(member, NodeDecision):
            return ChoicePath((*outer, *records, member), entry.steps[index + 1 :], (*label, name))
        records.append(member)
        current = cast(NodeDecl, member).space_type
    return path_proxy(current, (*outer, *records), (*label, name))


def alias_node(instance: _Namespace, name: str) -> object:
    """Alias ``name`` read on a design space or a receiver: its node, read through its steps
    (each a child, a selector's selected case, or a case's own alias). Where a selection
    places nothing there, a design space reads ``None`` (as it reads such a selector), and
    a method's read halts it as inapplicable (``MethodReader.alias_node``)."""
    from .occurrence import _reader, held

    entry = alias_table(type(instance))[name]
    reader = _reader(instance)
    if reader is not None:
        return reader.alias_node(instance, entry.steps)
    held(instance)  # a configuration, never a bare object
    current: object = instance
    for step in entry.steps:
        if current is None:
            return None
        current = getattr(current, step)
    return current


__all__ = ["AliasEnd", "AliasEntry", "alias_face", "alias_node", "alias_table", "reached"]
