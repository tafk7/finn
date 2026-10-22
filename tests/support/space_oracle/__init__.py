# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle: any implementation of the engine's public questions, asked by
key, and compared with another on real Space classes.

An ``Implementation`` opens the design space today's API built (``design_space``,
``commit``, a change's instance): the common input, since every implementation reads the
same compiled classes. What it opens (``Answers``) is asked the public questions by key,
the declaration paths (``mm2.pe``), and answers them in today's public
result types (``QueryResult``, ``inspection.Validity``, the states' ``ChoiceState``s,
``inspection.Viable``, a change's ``ConfigurationResult``). No node id, handle or world
crosses this interface.

An ``Implementation`` also builds a design space from nothing (``make``): a Space class
with exactly the given facts and stored choices, by key, unchecked. A design space reached
by changes must answer as the one built from its facts and stored choices does (history
independence), and the walks hold every implementation to it through ``make``.

Today's engine (``today.TODAY``) answers through the public API alone. A prototype or
T1's evaluator plugs in as ``module:attribute`` (``load``), with no line here.

- ``forms``: the answers by key in one canonical, JSON-able form;
- ``compare``: two implementations at one design space, along one change, and a seeded
  walk of changes; disagreements by key;
- ``targets``: the design spaces compared (FINN's toys, stage 7's chain, TFC);
- ``python -m support.space_oracle``: the dump tool (``dump``, ``diff``, ``walk``).

Engine-only, but for ``targets`` and the tool: this package imports ``finn.core.space``
and nothing else of FINN.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from finn.core.space import QueryResult, Space, inspection
from finn.core.space.edits import ConfigurationResult
from finn.core.space.ir import NodeKind

NO_FACTS: Mapping[str, object] = {}


@dataclass(frozen=True)
class Changed:
    """A change's or a bind's report, in today's shape, and the successor's answers where
    it was accepted (``None`` where it was refused)."""

    report: ConfigurationResult[Any]
    after: Answers | None


class Answers(Protocol):
    """One design space, asked by key."""

    def stored(self) -> Mapping[str, object]:
        """Every stored choice by key, dormant ones included."""
        ...

    def members(self) -> Sequence[tuple[str, NodeKind]]:
        """Every member below the root, by key, with its kind (every scope made)."""
        ...

    def scopes(self) -> Sequence[str]:
        """Every scope below the root, by name (the root is ``""``), sorted."""
        ...

    def query(self, key: str) -> QueryResult[object]:
        """A member's answer, what reading it would give."""
        ...

    def validity(self, scope: str = "") -> inspection.Validity:
        """A scope's validity (the root's by default)."""
        ...

    def states(self) -> Sequence[tuple[str, inspection.ChoiceState]]:
        """Every choice's state, by key, in the flow order."""
        ...

    def viable(self, keys: Sequence[str]) -> Sequence[inspection.Viable]:
        """Each named choice's listed values, each possible or refused, in ``keys``'
        order."""
        ...

    def change(
        self,
        commits: Mapping[str, object],
        clears: Sequence[str],
        facts: Mapping[str, object] = NO_FACTS,
    ) -> Changed:
        """Other values for root facts (``facts``, by key), then the choices ``commits``
        sets and the choices ``clears`` clears, one change, in that order."""
        ...

    def bind(self, stored: Mapping[str, object]) -> Changed:
        """This design space's facts with exactly the saved choices ``stored``."""
        ...


class Implementation(Protocol):
    """An implementation of the public questions."""

    @property
    def name(self) -> str:
        """What it is, as a dump's header and a walk's report name it."""
        ...

    def open(self, configuration: Space) -> Answers:
        """The answers of the design space ``configuration`` (today's API built it)."""
        ...

    def make(
        self,
        space_class: type[Space],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        """The design space of ``space_class`` with exactly ``facts`` (by key, the root's
        params; an optional one left out is absent) and the choices ``stored`` (by key,
        dormant ones included), built from nothing and unchecked: no rule is checked, and
        each stored value must pass its type check. A design space reached by changes
        answers as the one made from its facts and its ``stored()`` (history
        independence)."""
        ...


def load(spec: str) -> Implementation:
    """A registered implementation by name (``today``), or one named ``module:attribute``."""
    if spec in IMPLEMENTATIONS:
        return IMPLEMENTATIONS[spec]
    module, colon, attribute = spec.partition(":")
    if not colon or not module or not attribute:
        known = ", ".join(sorted(IMPLEMENTATIONS))
        raise ValueError(f"{spec!r}: an implementation is one of {known}, or module:attribute")
    found: Implementation = getattr(importlib.import_module(module), attribute)
    return found


# Registered last: today's adapter reads the interface above.
from support.space_oracle.today import TODAY  # noqa: E402

IMPLEMENTATIONS: dict[str, Implementation] = {TODAY.name: TODAY}
"""Every implementation the oracle compares with today's: today's alone at T0."""

__all__ = [
    "IMPLEMENTATIONS",
    "NO_FACTS",
    "Answers",
    "Changed",
    "Implementation",
    "load",
]
