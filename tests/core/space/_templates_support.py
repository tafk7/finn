# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The templates' tests' comparison (``test_templates``; the kernel tree's
``test_space_templates``): a model's scope records, read whole, made by demand, in the
permuted order (reversed by path) and eagerly, in a canonical form by key; and the Space
classes a set of modules defines.

Every address, scope and choice reads the same whichever order the scopes were made in:
keys, kinds, owners, edges by key, scopes by path, choices by key, the decisions in
declaration order. (That the scope records read as the structure they replaced resolved was
checked when they replaced it, as that structure had been against digests of the line's
one table of every placement.)"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Iterator

from finn.core.space import Space
from finn.core.space._scopes import Address, Scopes
from finn.core.space._template import Compiled, compiled
from finn.core.space.domains import Domain
from finn.core.space.errors import DefinitionError

# -- the canonical form ------------------------------------------------------------------------


def _name(function: object) -> str | None:
    return None if function is None else getattr(function, "__qualname__", repr(function))


def _stable(value: object) -> object:
    """A value's form independent of the objects one link makes: a domain by its parts,
    anything else by its repr without addresses."""
    if isinstance(value, Domain):
        return (
            tuple(name for name, _ in value.dependencies),
            _name(value.accepts),
            _name(value.candidates),
            None if value.value_semantics is None else value.value_semantics.name,
            _stable(value._finite_values),
            value.ordered,
        )
    return re.sub(r" at 0x[0-9a-f]+", "", repr(value))


def canonical(scopes: Scopes) -> dict[str, object]:
    """``scopes`` read whole, by key, independent of the order its scopes were made in."""

    def key(address: Address | None) -> str | None:
        return None if address is None else address.key

    found: dict[str, object] = {}
    for node in scopes.all_addresses():
        found["node " + node.key] = (
            node.kind,
            node.owner,
            node.origin,
            None if node.semantics is None else node.semantics.name,
            key(node.guard),
            tuple((argument.name, key(argument.node)) for argument in node.arguments),
            tuple((argument.name, key(argument.node)) for argument in node.domain_arguments),
            key(node.output),
            key(node.selector),
            tuple(key(target) for target in node.constraints),
            tuple((label, key(target)) for label, target in node.alternatives),
            _stable(node.value),
            _stable(node.domain),
            node.required,
            _name(node.function),
            node.call_style,
            node.scope.name,
        )
    for scope in scopes.ordered():
        found["scope " + scope.name] = (
            scope.path,
            None if scope.parent is None else scope.parent.name,
            scope.space_type,
            key(scope.guard),
            key(scope.validity),
            sorted((name, key(index)) for name, index in scope.named_members.items()),
            sorted((name, child.name) for name, child in scope.named_children.items()),
            # Keyed by declarations: the same objects in every model of the class.
            {(id(member), key(index)) for member, index in scope.members.items()},
            {(id(child), record.name) for child, record in scope.children.items()},
            {(id(choice), record.key) for choice, record in scope.choices.items()},
            id(scope.record),
        )
    for choice in scopes.choices():
        found["choice " + choice.key] = (
            choice.scope.name,
            key(choice.selector),
            key(choice.guard),
            tuple((case, None if child is None else child.name) for case, child in choice.cases),
            sorted((member, key(index)) for member, index in choice.members.items()),
        )
    found["decisions"] = [key(index) for index in scopes.decisions]
    found["parameters"] = [key(index) for index in scopes.parameters]
    found["statistics"] = scopes.statistics()
    return found


def differences(space_type: type[Space]) -> list[str]:
    """The first entries where the class's scope records, made in the permuted order or
    eagerly, differ from those made by demand, if any."""
    form = compiled(space_type)
    lazy = canonical(Scopes(form))
    found: list[str] = []
    for name, scopes in (
        ("permuted", Scopes(form, permuted=True)),
        ("eager", Scopes(form, eager=True)),
    ):
        other = canonical(scopes)
        for entry in sorted(lazy.keys() | other.keys()):
            held, got = lazy.get(entry), other.get(entry)
            if held != got:
                found.append(
                    f"{space_type.__qualname__}: {entry}\n  by demand: {held}\n  {name}: {got}"
                )
                if len(found) == 3:
                    return found
    return found


def assert_reads_alike(space_types: Iterable[type[Space]]) -> int:
    """Each class's scope records read alike by demand, permuted and eager; the number
    compared."""
    failures: list[str] = []
    compared = 0
    for space_type in space_types:
        failures.extend(differences(space_type))
        compared += 1
    assert not failures, "\n".join(failures)
    return compared


# -- the corpus ---------------------------------------------------------------------------------


def defined(names: Iterable[str]) -> Iterator[type[Space]]:
    """Every Space subclass defined at module level in the modules ``names``."""
    for name in names:
        module = importlib.import_module(name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, Space)
                and value is not Space
                and value.__module__ == name
            ):
                yield value


def links(space_type: type[Space]) -> bool:
    """Whether it compiles (a class that is a refusal's case does not)."""
    try:
        compiled(space_type)
    except DefinitionError:
        return False
    return True


def reached(form: Compiled) -> dict[type[Space], Compiled]:
    """Every compiled form reached from ``form`` through its slots, by class."""
    found: dict[type[Space], Compiled] = {}
    pending = [form]
    while pending:
        current = pending.pop()
        if current.space_type in found:
            continue
        found[current.space_type] = current
        pending.extend(slot.compiled for slot in current.slots)
    return found
