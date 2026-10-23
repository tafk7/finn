# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The read inference per scope: kept on purpose, as the reference the engine's inference per
class (``finn.core.space._reads``) is checked against.

The engine resolves each ``self`` function's paths once per class, against the class's
compiled form, and maps them through a scope when an address is first asked. This module
resolves them again at every scope, over a model whose every scope is made and every member
read (``line_scopes``: an eager model of its own), as the line's inference did over a table
holding every node: each path's names against the scope's own children and members, through
each selector to its candidates' scopes, helpers followed in the scope they are reached
through, ``present()`` through a child's guard. Its answers are addresses of that model
(``ModelReads``, keyed by address): each method's possible reads, the paths that name
nothing, each guarded member's existence guards (its guard, then each guard that one is
evaluated under, to the root), and each case guard. The line oracles (``_whole_support``,
``_domains_support``, ``kernel_ops/test_stream_reads``) read it, so they do not share the
inference they check.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import MappingProxyType

from finn.core.space import Space
from finn.core.space._reads import ModelReads, Path, _Uses, _uses
from finn.core.space._scopes import Address, Argument, ScopeRecord, Scopes
from finn.core.space._template import compiled


def line_scopes(space_type: type[Space]) -> Scopes:
    """A model of ``space_type`` of its own, every scope made and every member read."""
    return Scopes(compiled(space_type), eager=True)


class LineInference:
    """The reads of every ``self`` method of one model, helpers followed in the scope they
    are reached through, resolved at each scope."""

    def __init__(self, scopes: Scopes) -> None:
        self.scopes = scopes
        self.uses: dict[int, _Uses] = {}
        # A function's reads at a scope, and the paths that resolved to nothing.
        self.found: dict[
            tuple[int, ScopeRecord], tuple[frozenset[Address] | None, tuple[str, ...]]
        ] = {}
        self.visiting: set[tuple[int, ScopeRecord]] = set()

    def _uses_of(self, function: Callable[..., object]) -> _Uses:
        key = id(function)
        found = self.uses.get(key)
        if found is None:
            found = self.uses[key] = _uses(function)
        return found

    def reads(
        self, function: Callable[..., object], scope: ScopeRecord
    ) -> tuple[frozenset[Address] | None, tuple[str, ...]]:
        """What ``function`` may read when its receiver is at ``scope``, and the paths that
        name nothing; ``None`` for a dynamic function."""
        key = (id(function), scope)
        held = self.found.get(key)
        if held is not None:
            return held
        if key in self.visiting:
            return frozenset(), ()  # a helper reached again through itself adds nothing
        self.visiting.add(key)
        try:
            result = self._reads(function, scope)
        finally:
            self.visiting.discard(key)
        self.found[key] = result
        return result

    def _reads(
        self, function: Callable[..., object], scope: ScopeRecord
    ) -> tuple[frozenset[Address] | None, tuple[str, ...]]:
        uses = self._uses_of(function)
        if uses.dynamic:
            return None, ()
        found: set[Address] = set()
        missing: list[str] = []
        helpers: list[tuple[Callable[..., object], ScopeRecord]] = []
        for path in sorted(uses.paths):
            if not self._resolve(scope, path, found, helpers):
                missing.append(".".join(path))
        for path in sorted(uses.present):
            if not self._presence(scope, path, found):
                missing.append(f"present({'.'.join(path)})")
        for prefix, name in sorted(uses.calls):
            if not self._call(scope, prefix, name, found, helpers):
                missing.append(".".join((*prefix, name)) + "()")
        for helper, at in helpers:
            more, unresolved = self.reads(helper, at)
            if more is None:
                return None, ()
            found |= more
            missing.extend(unresolved)
        return frozenset(found), tuple(missing)

    def _scopes(
        self, scope: ScopeRecord, path: Path, found: set[Address]
    ) -> list[ScopeRecord] | None:
        """The scopes ``path`` (child and selector names) reaches from ``scope``, adding the
        selectors it reads on the way; ``None`` where a name is no child or selector."""
        reached = [scope]
        for name in path:
            following: list[ScopeRecord] = []
            for at in reached:
                if name in at.named_children:
                    following.append(at.named_children[name])
                    continue
                selector = at.named_members.get(name)
                choice = None if selector is None else self.scopes.selector_choice(selector)
                if selector is None or choice is None:
                    return None
                found.add(selector)
                following.extend(case for _, case in choice.cases if case is not None)
            reached = following
        return reached

    def _resolve(
        self,
        scope: ScopeRecord,
        path: Path,
        found: set[Address],
        helpers: list[tuple[Callable[..., object], ScopeRecord]],
    ) -> bool:
        """Add the addresses ``path`` may read from ``scope``. A path's name that is a
        property or a method of the Space class there is followed as a helper. False where
        the first name names nothing (a class constant)."""
        name, rest = path[0], path[1:]
        if name in scope.named_children:
            if rest:
                self._resolve(scope.named_children[name], rest, found, helpers)
            return True
        index = scope.named_members.get(name)
        if index is None:
            return self._attribute(scope, name, helpers)
        found.add(index)
        choice = self.scopes.selector_choice(index)
        if choice is not None and rest:
            for _, case in choice.cases:
                if case is not None:
                    self._resolve(case, rest, found, helpers)
        return True

    @staticmethod
    def _attribute(
        scope: ScopeRecord, name: str, helpers: list[tuple[Callable[..., object], ScopeRecord]]
    ) -> bool:
        """A name of the Space class itself: a property or a method is a helper read at
        ``scope``; anything else reads nothing of the Space."""
        held = inspect.getattr_static(scope.space_type, name, None)
        if isinstance(held, property) and held.fget is not None:
            helpers.append((held.fget, scope))
            return True
        if inspect.isfunction(held):
            helpers.append((held, scope))
            return True
        return False

    def _call(
        self,
        scope: ScopeRecord,
        prefix: Path,
        name: str,
        found: set[Address],
        helpers: list[tuple[Callable[..., object], ScopeRecord]],
    ) -> bool:
        """``self.<prefix>.<name>(...)``: a helper method of the Space reached through
        ``prefix``, read at that scope. A static or class method reads nothing of it."""
        reached = self._scopes(scope, prefix, found)
        if reached is None:
            # A method of a value read through the path: the path is a read.
            return self._resolve(scope, prefix, found, helpers) if prefix else False
        known = False
        for at in reached:
            held = inspect.getattr_static(at.space_type, name, None)
            if isinstance(held, (staticmethod, classmethod)):
                known = True
            elif inspect.isfunction(held):
                helpers.append((held, at))
                known = True
        return known

    def _presence(self, scope: ScopeRecord, path: Path, found: set[Address]) -> bool:
        """Add what ``present()`` reads of the last name of ``path``, a child: its guard.
        Anything else is no node (a value input is known): nothing found."""
        reached = self._scopes(scope, path[:-1], found)
        if reached is None:
            return False
        name = path[-1]
        known = False
        for at in reached:
            if name in at.named_children:
                guard = at.named_children[name].guard
                if guard is not None:
                    found.add(guard)
                known = True
        return known


def line_existence(index: Address) -> tuple[Address, ...]:
    """The guards deciding whether ``index`` exists, innermost first: its own guard, and
    each guard that one is evaluated under, to the root."""
    found: list[Address] = []
    current = index.guard
    while current is not None:
        found.append(current)
        current = current.guard
    return tuple(found)


def line_cases(scopes: Scopes) -> dict[Address, tuple[Address, str]]:
    """Each guard that holds where a selector is one of its cases (the guard a case's scope
    is placed under, directly or under its ``when``: it reads the selector alone), with that
    selector and case."""
    found: dict[Address, tuple[Address, str]] = {}
    for choice in scopes.choices():
        selected = (Argument("selected", choice.selector),)
        for case, scope in choice.cases:
            guard = None if scope is None else scope.guard
            while guard is not None:
                if guard.kind == "derived" and guard.arguments == selected:
                    found[guard] = (choice.selector, case)
                    break
                guard = guard.guard
    return found


def line_infer(scopes: Scopes) -> ModelReads:
    """What every ``self`` method may read at its scope, every guarded member's existence
    guards, and which guards are a selector's cases, over every address of ``scopes``."""
    inference = LineInference(scopes)
    methods: dict[Address, frozenset[Address] | None] = {}
    unresolved: dict[Address, tuple[str, ...]] = {}
    existence: dict[Address, tuple[Address, ...]] = {}
    for index in scopes.all_addresses():
        if index.guard is not None:
            existence[index] = line_existence(index)
        function = index.function
        if index.call_style != "self" or function is None:
            continue
        reads, missing = inference.reads(function, index.scope)
        # A method reading itself keeps that read: it is a cycle, found when it runs.
        methods[index] = reads
        if missing:
            unresolved[index] = tuple(dict.fromkeys(missing))
    return ModelReads(
        MappingProxyType(methods),
        MappingProxyType(unresolved),
        MappingProxyType(existence),
        MappingProxyType(line_cases(scopes)),
    )


__all__ = [
    "LineInference",
    "line_cases",
    "line_existence",
    "line_infer",
    "line_scopes",
]
