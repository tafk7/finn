# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The implementation the harness asks wherever a caller names none, and a pytest
session's option naming it.

**The default.** ``Engine.open``, ``Engine.make``, ``definition`` and ``Case.start`` ask
``default()`` where they are given no implementation: today's engine
(``support.space_oracle.today.TODAY``) unless a session names another, and
``matrix.expected`` marks a row's witness by the milestones the default has reached
(``matrix.reached``). The walks name theirs (``walks.walk(..., implementation=)``, the
CLI's ``--implementation``) and never read this one.

**The option.** ``--space-implementation MODULE:ATTRIBUTE`` (or a registered name;
``today`` by default), registered by ``tests/core/space/conftest.py`` and loaded by
``support.space_oracle.load``, makes the implementation it names the default for the
session (``Option.start``). Named otherwise than today's engine, it counts every ``open``
and ``make`` of today's implementation for the session (its class's, whatever instance or
wrapper calls it), by the test it ran in, and the session fails unless the count is 0: a
test that names today's engine itself, or one asking it some other way, is reported, never
left to answer in silence. Its summary names the implementation, each test deselected (by
node id) and the count.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import pytest
from support.space_oracle import Implementation, load
from support.space_oracle.today import TODAY

OPTION = "--space-implementation"
"""The pytest option naming the default implementation."""

OUTSIDE = "(outside any test)"
"""Where today's opens between tests are counted."""

_DEFAULT: list[Implementation] = [TODAY]


def default() -> Implementation:
    """The implementation asked where a caller names none: today's engine, or the one
    made the default (``use``)."""
    return _DEFAULT[0]


def use(implementation: Implementation) -> Implementation:
    """Make ``implementation`` the default; the one it replaces."""
    previous, _DEFAULT[0] = _DEFAULT[0], implementation
    return previous


def chosen(implementation: Implementation | None) -> Implementation:
    """``implementation``, or the default where it is ``None``."""
    return default() if implementation is None else implementation


@dataclass
class TodaysOpens:
    """Each ``open`` and ``make`` of today's implementation (its class's), counted by the
    test it ran in while installed (``install``, ``restore``)."""

    #: today's implementation's class, its ``open`` and ``make`` counted while installed
    counted: type | None = None
    #: each ``open`` or ``make``, by the test it ran in
    opened: dict[str, int] = field(default_factory=dict)
    #: the test running, by node id (``OUTSIDE`` between tests)
    running_test: str = OUTSIDE

    def install(self) -> None:
        self.counted = type(TODAY)
        for name in ("open", "make"):
            setattr(self.counted, name, self._counting(getattr(self.counted, name)))

    def _counting(self, method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def counting(*arguments: Any, **keywords: Any) -> Any:
            test = self.running_test
            self.opened[test] = self.opened.get(test, 0) + 1
            return method(*arguments, **keywords)

        counting.__wrapped_by__ = self  # type: ignore[attr-defined]
        return counting

    def restore(self) -> None:
        if self.counted is not None:
            for name in ("open", "make"):
                method = getattr(self.counted, name)
                if getattr(method, "__wrapped_by__", None) is self:
                    setattr(self.counted, name, method.__wrapped__)
            self.counted = None

    def count(self) -> str:
        """The summary's line: how often today's engine was opened or made, and in which
        tests."""
        total = sum(self.opened.values())
        where = "; ".join(f"{test}: {number}" for test, number in sorted(self.opened.items()))
        return f"today's engine opened or made: {total}" + (f" ({where})" if where else "")


@dataclass
class Option:
    """A session's ``--space-implementation``: the implementation it names made the default
    (``start``) until the session ends (``stop``), today's engine counted while another is
    (``opens``)."""

    spec: str
    implementation: Implementation
    #: the default it replaced
    previous: Implementation
    #: today's opens, counted where another implementation is named
    opens: TodaysOpens | None = None
    #: each test deselected, by node id
    deselected: list[str] = field(default_factory=list)

    @classmethod
    def start(cls, spec: str) -> Option:
        """``spec`` loaded (``support.space_oracle.load``; a usage error where it names
        nothing) and made the default; today's engine counted unless it is today's."""
        try:
            implementation = load(spec)
        except (ImportError, AttributeError, ValueError) as error:
            raise pytest.UsageError(f"{OPTION} {spec}: {error}") from error
        option = cls(spec, implementation, use(implementation))
        if implementation is not TODAY:
            option.opens = TodaysOpens()
            option.opens.install()
        return option

    def stop(self) -> None:
        """The default restored, and today's engine no longer counted."""
        if self.opens is not None:
            self.opens.restore()
        use(self.previous)

    @contextmanager
    def running(self, test: str) -> Iterator[None]:
        """Today's opens counted as ``test``'s while it runs."""
        if self.opens is None:
            yield
            return
        self.opens.running_test = test
        try:
            yield
        finally:
            self.opens.running_test = OUTSIDE

    def failed(self) -> bool:
        """Whether today's engine was opened or made while counted."""
        return self.opens is not None and bool(self.opens.opened)

    def header(self) -> str:
        name = self.implementation.name
        return f"space implementation: {name} ({self.spec}), the harness's default ({OPTION})"

    def summary(self) -> list[str]:
        """The summary's lines where another implementation than today's is named (none
        otherwise): the implementation, each test deselected, and today's opens."""
        if self.opens is None:
            return []
        return [
            self.header(),
            *(f"deselected: {node}" for node in sorted(self.deselected) or ["none"]),
            self.opens.count(),
        ]


__all__ = ["OPTION", "OUTSIDE", "Option", "TodaysOpens", "chosen", "default", "use"]
