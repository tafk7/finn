# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The implementation the harness asks wherever a caller names none, and a pytest
session's option naming it.

**The default.** ``Engine.open``, ``Engine.make``, ``definition`` and ``Case.start`` ask
``default()`` where they are given no implementation: the public API
(``support.space_oracle.public.PUBLIC``) unless a session names another, and
``matrix.expected`` marks a row's witness by the milestones the default has reached
(``matrix.reached``). The walks name theirs (``walks.walk(..., implementation=)``, the
CLI's ``--implementation``) and never read this one.

**The option.** ``--space-implementation MODULE:ATTRIBUTE`` (or a registered name;
``public`` by default), registered by ``tests/core/space/conftest.py`` and loaded by
``support.space_oracle.load``, makes the implementation it names the default for the
session (``Option.start``), and the report's header names it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from support.space_oracle import Implementation, load
from support.space_oracle.public import PUBLIC

OPTION = "--space-implementation"
"""The pytest option naming the default implementation."""

_DEFAULT: list[Implementation] = [PUBLIC]


def default() -> Implementation:
    """The implementation asked where a caller names none: the public API, or the one made
    the default (``use``)."""
    return _DEFAULT[0]


def use(implementation: Implementation) -> Implementation:
    """Make ``implementation`` the default; the one it replaces."""
    previous, _DEFAULT[0] = _DEFAULT[0], implementation
    return previous


def chosen(implementation: Implementation | None) -> Implementation:
    """``implementation``, or the default where it is ``None``."""
    return default() if implementation is None else implementation


@dataclass
class Option:
    """A session's ``--space-implementation``: the implementation it names made the default
    (``start``) until the session ends (``stop``)."""

    spec: str
    implementation: Implementation
    #: the default it replaced
    previous: Implementation

    @classmethod
    def start(cls, spec: str) -> Option:
        """``spec`` loaded (``support.space_oracle.load``; a usage error where it names
        nothing) and made the default."""
        try:
            implementation = load(spec)
        except (ImportError, AttributeError, ValueError) as error:
            raise pytest.UsageError(f"{OPTION} {spec}: {error}") from error
        return cls(spec, implementation, use(implementation))

    def stop(self) -> None:
        """The default restored."""
        use(self.previous)

    def header(self) -> str:
        name = self.implementation.name
        return f"space implementation: {name} ({self.spec}), the harness's default ({OPTION})"


__all__ = ["OPTION", "Option", "chosen", "default", "use"]
