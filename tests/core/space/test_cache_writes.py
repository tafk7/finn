# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A world's cache is only added to, and the engine's source shows no write that escapes it.

A world keeps the first entry it finds for a node, and never replaces or removes one: its
entries are shared with the worlds made from it, which take them as they are. This test
reads the engine's source for every write to a cache, so none escapes by a path the tests
never take. A cache is written only by ``setdefault``, at the sites named below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from finn.core.space import _world

SOURCE = Path(_world.__file__).parent

# Where a cache is given an entry with ``setdefault``.
SETDEFAULT_SITES = {
    ("_world.py", "_get"): "a world keeps the first entry it finds (only added to)",
    ("_world.py", "flatten"): "a world standing alone takes its ancestors' entries it lacks",
}
# A function that gives a world its cache other than its constructor's annotated assignment.
CACHE_SETTERS: set[tuple[str, str]] = set()
# Calls that change a held dict other than by adding what is not there.
WRITES = {"update", "pop", "popitem", "clear", "__setitem__", "__delitem__"}


def is_cache(node: ast.expr, aliases: set[str]) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "cache") or (
        isinstance(node, ast.Name) and node.id in aliases
    )


class Scan(ast.NodeVisitor):
    def __init__(self, name: str) -> None:
        self.name = name
        self.stack: list[str] = []
        self.scopes: list[set[str]] = [set()]
        self.setdefaults: set[tuple[str, str]] = set()
        self.setters: set[tuple[str, str]] = set()
        self.violations: list[str] = []

    @property
    def where(self) -> tuple[str, str]:
        return self.name, self.stack[-1] if self.stack else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.scopes.append(set())
        self.generic_visit(node)
        self.scopes.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def aliases(self) -> set[str]:
        return self.scopes[-1]

    def flag(self, node: ast.AST, what: str) -> None:
        self.violations.append(f"{self.name}:{node.lineno} in {self.where[1]}: {what}")  # type: ignore[attr-defined]

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.bind(target, node.value)
            self.written(target, node)
        self.generic_visit(node)

    def bind(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name) and is_cache(value, self.aliases()):
            self.aliases().add(target.id)
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            for each, given in zip(target.elts, value.elts):
                self.bind(each, given)

    def written(self, target: ast.expr, node: ast.AST) -> None:
        if isinstance(target, ast.Subscript) and is_cache(target.value, self.aliases()):
            self.flag(node, "a subscript of a cache is written")
        elif isinstance(target, ast.Attribute) and target.attr == "cache":
            self.flag(node, "a world's cache is rebound")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if is_cache(node.target, self.aliases()):
            self.flag(node, "a cache is updated in place (an augmented assignment)")
        self.written(node.target, node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript) and is_cache(target.value, self.aliases()):
                self.flag(node, "an entry of a cache is deleted")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute):
            if is_cache(function.value, self.aliases()):
                if function.attr in WRITES:
                    self.flag(node, f"a cache is written by ``{function.attr}``")
                elif function.attr == "setdefault":
                    self.setdefaults.add(self.where)
            elif (
                function.attr == "__setattr__"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "cache"
            ):
                self.setters.add(self.where)
                if self.where not in CACHE_SETTERS:
                    self.flag(node, "a world is given a cache outside its constructor")
        self.generic_visit(node)


def scan(source: str, name: str = "<snippet>") -> Scan:
    found = Scan(name)
    found.visit(ast.parse(source))
    return found


def test_no_code_writes_a_cache_but_by_setdefault() -> None:
    violations: list[str] = []
    setdefaults: set[tuple[str, str]] = set()
    setters: set[tuple[str, str]] = set()
    for path in sorted(SOURCE.rglob("*.py")):
        found = scan(path.read_text(), path.name)
        violations += found.violations
        setdefaults |= found.setdefaults
        setters |= found.setters
    assert not violations, "\n".join(violations)
    assert setdefaults == set(SETDEFAULT_SITES), (
        "a setdefault on a cache is where the rule is met by a check, and each site is named in "
        f"this test: found {sorted(setdefaults)}"
    )
    assert setters == CACHE_SETTERS


@pytest.mark.parametrize(
    "line",
    [
        "world.cache[index] = entry",
        "world.cache[index] |= entry",
        "world.cache.update(other)",
        "world.cache |= other",
        "world.cache.pop(index)",
        "world.cache.clear()",
        "del world.cache[index]",
        "cache = world.cache\n    cache[index] = entry",
        "nodes, cache = world.nodes, world.cache\n    cache.update(other)",
        "object.__setattr__(world, 'cache', {})",
    ],
)
def test_the_scan_finds_each_way_of_writing_a_cache(line: str) -> None:
    assert scan(f"def f(world, index, entry, other):\n    {line}\n").violations


def test_the_scan_lets_setdefault_and_a_local_copy_be() -> None:
    found = scan(
        "def f(world, base, index, entry):\n"
        "    kept = {}\n"
        "    kept[index] = entry\n"
        "    kept.pop(index, None)\n"
        "    cache = base.cache\n"
        "    dict.update(kept, cache)\n"
        "    return world.cache.setdefault(index, entry)\n"
    )
    assert not found.violations and found.setdefaults == {("<snippet>", "f")}
