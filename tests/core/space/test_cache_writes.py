# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A snapshot's cache is only added to, and the engine's source shows no write that escapes it.

The rule (``finn.core.space._runtime.Snapshot``) is checked where production pays nothing for
it: the tests make every cache a refusing dict (``support.checked_cache``, installed by
``tests/conftest.py``), and this test reads the engine's source for every write to a cache, so
none escapes by a path the tests never take. A cache is written only by ``setdefault``, at the
sites named below, and by the function that makes a copy (``inherit``), which gives a new
cache and writes no held one. Each setdefault site is met by a check: ``finish``'s own
(``_same``, which compares what the node read), and, in the tests, the refusing dict's, which
refuses a kept evaluation that answers differently or read more than the one dropped.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from support.checked_cache import CheckedCache
from support.space_pipeline import Pipeline
from support.space_walk import walk

from finn.core.space import _runtime, design_space

SOURCE = Path(_runtime.__file__).parent

# Where a cache is given an evaluation with ``setdefault``, and what meets a difference there.
SETDEFAULT_SITES = {
    ("_execution.py", "finish"): "a node cached twice in one run must have read the same (_same)",
    ("_runtime.py", "publish"): "a trial's evaluations, given to the successor: the checking dict",
    ("_forcing.py", "_publish"): "forcing's last copy, given to its snapshot: the checking dict",
    ("_world.py", "_get"): "the evaluation core's world: the first entry is kept (only added to)",
    ("_world.py", "flatten"): "a world standing alone takes its ancestors' entries it lacks",
}
# The function that gives a snapshot its cache.
CACHE_SETTERS = {("_runtime.py", "inherit")}
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
            self.flag(node, "a snapshot's cache is rebound")

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
                    self.flag(node, "a snapshot is given a cache outside the function that copies")
        self.generic_visit(node)


def scan(source: str, name: str = "<snippet>") -> Scan:
    found = Scan(name)
    found.visit(ast.parse(source))
    return found


def test_no_code_writes_a_cache_but_by_setdefault_and_by_the_function_that_copies() -> None:
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
        "snapshot.cache[index] = entry",
        "snapshot.cache[index] |= entry",
        "snapshot.cache.update(other)",
        "snapshot.cache |= other",
        "snapshot.cache.pop(index)",
        "snapshot.cache.clear()",
        "del snapshot.cache[index]",
        "cache = snapshot.cache\n    cache[index] = entry",
        "nodes, cache = snapshot.nodes, snapshot.cache\n    cache.update(other)",
        "object.__setattr__(snapshot, 'cache', {})",
    ],
)
def test_the_scan_finds_each_way_of_writing_a_cache(line: str) -> None:
    assert scan(f"def f(snapshot, index, entry, other):\n    {line}\n").violations


def test_the_scan_lets_setdefault_and_a_local_copy_be() -> None:
    found = scan(
        "def f(snapshot, base, index, entry):\n"
        "    kept = {}\n"
        "    kept[index] = entry\n"
        "    kept.pop(index, None)\n"
        "    cache = base.cache\n"
        "    dict.update(kept, cache)\n"
        "    return snapshot.cache.setdefault(index, entry)\n"
    )
    assert not found.violations and found.setdefaults == {("<snippet>", "f")}


def test_every_cache_the_engine_makes_is_the_refusing_class_under_the_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot, the trial, the successor's start, each copy ``inherit`` makes (the trial's, the
    successor's, forcing's extended copies and rounds), and what a walk's configured points hold."""
    callers: set[str] = set()
    made: list[_runtime.Snapshot] = []
    real = _runtime.inherit

    def spy(target: _runtime.Snapshot, *args: object, **kwargs: object) -> None:
        callers.add(sys._getframe(1).f_code.co_name)
        real(target, *args, **kwargs)  # type: ignore[arg-type]
        made.append(target)

    monkeypatch.setattr(_runtime, "inherit", spy)
    point = design_space(Pipeline(width=10))
    walk(point, seed=2, steps=40)
    assert {"__init__", "successor", "_extended", "_rounds"} <= callers
    assert made and all(type(each.cache) is CheckedCache for each in made)
    assert type(_runtime.new_cache()) is CheckedCache
    assert type(_runtime.Snapshot(None, {}).cache) is CheckedCache  # type: ignore[arg-type]
    assert type(_runtime.Snapshot(None, {}, forcing=False).cache) is CheckedCache  # type: ignore[arg-type]
