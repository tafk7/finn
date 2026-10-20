# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The templates' tests' comparison (``test_templates``; the kernel tree's
``test_space_templates``): a model's structure, fully resolved in either numbering, against
its eager form ``expand(C)``, in a canonical form by key, independent of numbering; and the
Space classes a set of modules defines.

Both are built from templates. That they are the line's one table of every placement
(``6d02612cc``'s ``link_space``) is checked against the line's digests of each class's
canonical form, kept in the design corpus (the lane's record, ``line/structure.json``),
with the probe that made them; the selections' owners apart, which a rule now fixes
(``_structure``)."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Iterator

from finn.core.space import Space
from finn.core.space._structure import Structure, expand
from finn.core.space._template import Template, link_template
from finn.core.space.domains import Domain
from finn.core.space.errors import DefinitionError
from finn.core.space.ir import Linked

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
            _name(value.order),
        )
    return re.sub(r" at 0x[0-9a-f]+", "", repr(value))


def canonical(model: Linked) -> dict[str, object]:
    """``model`` by key, independent of its numbering."""
    nodes, scopes, choices = model.nodes, model.scopes, model.choices

    def key(index: int | None) -> str | None:
        return None if index is None else nodes[index].key

    found: dict[str, object] = {}
    for node in nodes:
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
            tuple((key(alias), key(source)) for alias, source in node.via),
            _stable(node.value),
            _stable(node.domain),
            node.required,
            node.required_choice,
            _name(node.function),
            node.call_style,
            scopes[node.scope].name,
            key(model.forward[node.index]),
            node.index in model.editable_aliases,
        )
    for scope in scopes:
        found["scope " + scope.name] = (
            None if scope.parent is None else scopes[scope.parent].name,
            scope.space_type,
            key(scope.guard),
            key(scope.validity),
            sorted((name, key(index)) for name, index in scope.named_members.items()),
            sorted((name, scopes[index].name) for name, index in scope.named_children.items()),
            # Keyed by declarations: the same objects in both links.
            {(id(member), key(index)) for member, index in scope.members.items()},
            {(id(child), scopes[index].name) for child, index in scope.children.items()},
            {(id(choice), choices[index].key) for choice, index in scope.choices.items()},
            id(scope.record),
        )
    for choice in choices:
        found["choice " + choice.key] = (
            scopes[choice.scope].name,
            key(choice.selector),
            key(choice.guard),
            tuple(
                (case, None if child is None else scopes[child].name)
                for case, child in choice.cases
            ),
            sorted((member, key(index)) for member, index in choice.members.items()),
        )
    found["decisions"] = [key(index) for index in model.decisions]
    found["parameters"] = [key(index) for index in model.parameters]
    return found


def differences(space_type: type[Space]) -> list[str]:
    """The first entries where the structure, fully resolved in either numbering, differs
    from ``expand``, if any."""
    eager = canonical(expand(space_type))
    found: list[str] = []
    for name, model in (
        ("structure", Structure(link_template(space_type))),
        ("structure, permuted", Structure(link_template(space_type), permuted=True)),
    ):
        other = canonical(model)
        for entry in sorted(eager.keys() | other.keys()):
            held, got = eager.get(entry), other.get(entry)
            if held != got:
                found.append(
                    f"{space_type.__qualname__}: {entry}\n  expand: {held}\n  {name}: {got}"
                )
                if len(found) == 3:
                    return found
    return found


def assert_expands(space_types: Iterable[type[Space]]) -> int:
    """Each class's structure equals its expansion; the number compared."""
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
    """Whether its template links (a class that is a refusal's case does not)."""
    try:
        link_template(space_type)
    except DefinitionError:
        return False
    return True


def reached(template: Template) -> dict[type[Space], Template]:
    """Every template reached from ``template``, by class."""
    found: dict[type[Space], Template] = {}
    pending = [template]
    while pending:
        current = pending.pop()
        if current.space_type in found:
            continue
        found[current.space_type] = current
        pending.extend(
            link_template(scope.space_type) for scope in current.scopes[1:] if scope.parent == 0
        )
    return found
