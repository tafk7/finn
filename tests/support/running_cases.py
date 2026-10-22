# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The reference harness's hand cases under the running type: a pytest plugin.

    python -m pytest -p no:xdist -p support.running_cases -q core/space/test_reference_*.py

(from ``tests``). The harness answers the engine's side by today's engine wherever a test
names no implementation (``support.space_oracle.today.TODAY``, the default of each function
in ``TARGETS``). For the session, this plugin rebinds each of those defaults to the running
type, ``support.running_type:RUNNING`` (no cache), and names it in pytest's header and
summary.

**By milestone, not by name.** The running type implements the ``T4a`` milestone's rulings
(``MILESTONE``) and neither later one. So every row of the acceptance matrix
(``reference_harness.matrix.ROWS``) whose milestone is ``T4a`` is expected to pass: its
witness's strict expected failure is removed. The witnesses of the later milestones' rows
stay expected failures.

**It fails loudly.** At configure time, each default it rebinds must be today's engine, at
the parameter it names, and ``T4a`` must be a milestone of the matrix with rows; at
collection, each ``T4a`` witness of a module collected must be collected, marked as its row
marks it. Anything else is a usage error: a rename never lets it test today's engine in
silence.

It is a stand-in: once the reference harness takes the implementation as an option of its
own, this plugin goes.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

import pytest

TARGETS: tuple[tuple[str, str, str], ...] = (
    ("core.space.reference_harness.engine", "Engine.open", "implementation"),
    ("core.space.reference_harness.engine", "Engine.make", "implementation"),
    ("core.space.reference_harness.engine", "definition", "implementation"),
    ("core.space.reference_harness.compare", "Case.start", "implementation"),
)
"""Each default rebound: the module, the function (by qualified name) and its parameter."""

TODAY = "support.space_oracle.today:TODAY"
"""Today's engine, the default each target must hold."""

RUNNING = "support.running_type:RUNNING"
"""The implementation each default is rebound to."""

MATRIX = "core.space.reference_harness.matrix"
MILESTONE = "T4a"
"""The milestone whose rows the running type implements."""


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise pytest.UsageError(f"support.running_cases: {name} not found ({error})") from error


def _load(spec: str) -> Any:
    module, _, name = spec.partition(":")
    found = getattr(_module(module), name, None)
    if found is None:
        raise pytest.UsageError(f"support.running_cases: {spec} not found")
    return found


def _function(module: str, qualname: str) -> FunctionType:
    found: Any = _module(module)
    for part in qualname.split("."):
        found = getattr(found, part, None)
        if found is None:
            raise pytest.UsageError(f"support.running_cases: {module}.{qualname} not found")
    if not isinstance(found, FunctionType):
        raise pytest.UsageError(f"support.running_cases: {module}.{qualname} is no function")
    return found


def _position(function: FunctionType, parameter: str) -> int:
    """Where ``parameter``'s default is in ``function.__defaults__``."""
    code = function.__code__
    names = code.co_varnames[: code.co_argcount]
    defaults = function.__defaults__ or ()
    if parameter not in names or names.index(parameter) < len(names) - len(defaults):
        raise pytest.UsageError(
            f"support.running_cases: {function.__module__}.{function.__qualname__} "
            f"has no default for {parameter!r}"
        )
    return names.index(parameter) - (len(names) - len(defaults))


@dataclass
class Rebinding:
    """The defaults in ``targets`` rebound from ``today`` to ``running``, and the witnesses
    of ``milestone``'s rows expected to pass (each checked first: ``check``)."""

    targets: Sequence[tuple[str, str, str]] = TARGETS
    today: str = TODAY
    running: str = RUNNING
    matrix: str = MATRIX
    milestone: str = MILESTONE
    #: each function rebound, its defaults before
    saved: list[tuple[FunctionType, tuple[Any, ...]]] = field(default_factory=list)
    #: each witness of the milestone's rows (``module::function``)
    witnesses: frozenset[str] = frozenset()
    #: the witnesses whose expected failure was removed
    flipped: set[str] = field(default_factory=set)

    def check(self) -> list[tuple[FunctionType, int]]:
        """Each target's function and the position of its default, each default today's
        engine; the milestone's witnesses. A usage error where any is not as expected."""
        today = _load(self.today)
        found = []
        for module, qualname, parameter in self.targets:
            function = _function(module, qualname)
            position = _position(function, parameter)
            if (function.__defaults__ or ())[position] is not today:
                raise pytest.UsageError(
                    f"support.running_cases: {module}.{qualname}'s {parameter} is not "
                    f"{self.today} by default"
                )
            found.append((function, position))
        matrix = _module(self.matrix)
        if self.milestone not in getattr(matrix, "MILESTONES", ()):
            raise pytest.UsageError(
                f"support.running_cases: no milestone {self.milestone!r} in {self.matrix}"
            )
        rows = [row for row in getattr(matrix, "ROWS", ()) if row.milestone == self.milestone]
        if not rows or not all("::" in row.witness for row in rows):
            raise pytest.UsageError(
                f"support.running_cases: no rows of {self.milestone!r} with witnesses in "
                f"{self.matrix}"
            )
        self.witnesses = frozenset(row.witness for row in rows)
        return found

    def apply(self) -> None:
        """Check, then rebind each default to the running type."""
        found = self.check()
        running = _load(self.running)
        for function, position in found:
            defaults = function.__defaults__ or ()
            self.saved.append((function, defaults))
            function.__defaults__ = (*defaults[:position], running, *defaults[position + 1 :])

    def restore(self) -> None:
        while self.saved:
            function, defaults = self.saved.pop()
            function.__defaults__ = defaults

    def flip(self, items: Sequence[pytest.Item]) -> None:
        """Remove each milestone witness's strict expected failure, its own marker; a
        witness of a module collected that is missing, or not marked so, is an error."""
        modules = set()
        for item in items:
            if not isinstance(item, pytest.Function):
                continue
            module = item.module.__name__.rpartition(".")[2]
            modules.add(module)
            if f"{module}::{item.originalname}" not in self.witnesses:
                continue
            marks = [mark for mark in item.own_markers if mark.name == "xfail"]
            if len(marks) != 1 or marks[0].kwargs.get("strict") is not True:
                raise pytest.UsageError(
                    f"support.running_cases: {item.nodeid} is not marked as one strict "
                    "expected failure"
                )
            item.own_markers.remove(marks[0])
            self.flipped.add(f"{module}::{item.originalname}")
        missing = sorted(
            witness
            for witness in self.witnesses - self.flipped
            if witness.partition("::")[0] in modules
        )
        if missing:
            raise pytest.UsageError(
                f"support.running_cases: {self.milestone} witnesses not collected: {missing}"
            )

    def header(self) -> str:
        names = ", ".join(qualname for _, qualname, _ in self.targets)
        return (
            f"space implementation: {_load(self.running).name} ({self.running}), "
            f"the default of {names} (support.running_cases)"
        )


_KEY = pytest.StashKey[Rebinding]()


def pytest_configure(config: pytest.Config) -> None:
    rebinding = Rebinding()
    rebinding.apply()
    config.stash[_KEY] = rebinding


def pytest_unconfigure(config: pytest.Config) -> None:
    rebinding = config.stash.get(_KEY, None)
    if rebinding is not None:
        rebinding.restore()


def pytest_report_header(config: pytest.Config) -> str:
    return config.stash[_KEY].header()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    config.stash[_KEY].flip(items)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    rebinding = config.stash[_KEY]
    terminalreporter.write_line(rebinding.header())
    flipped = ", ".join(sorted(rebinding.flipped)) or "none"
    terminalreporter.write_line(f"{rebinding.milestone} witnesses expected to pass: {flipped}")


__all__ = ["MILESTONE", "RUNNING", "TARGETS", "TODAY", "Rebinding"]
