# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-class record: what a compiled model answers once for every design space of it,
made with the model (``compiler.Model``) and held by it for its life.

- ``reads``: what each ``self`` method may read, each address's existence guards, and each
  case guard (``_reads.infer``; answered address by address, as read, each under the
  model's lock);
- ``flow()``: the model's choices in the flow order, found once, on first use, from its
  compiled forms, each loop in it noted (``flow-loop``);
- ``presence_readers(source)``: the choices whose presence may read the choice or fact
  ``source``, from its compiled forms;
- ``flow_below(model, scope)``: the choices below a scope in the flow order;
  ``scopes_below(scopes, scope)``: a scope and every scope below it;
- ``session()``: the model's one ``evaluate.Session``, made on first use: every public
  question about a design space of the model runs in it, under the model's lock, and its
  node cache (``cache.NodeCache``, made in its default mode) keeps answers for the model's
  life, or until the session is dropped (``drop_session``).

Each part is made under the model's lock (``lock``).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.reporting.manifest import note

from .._reads import ModelReads, infer

if TYPE_CHECKING:
    from threading import RLock

    from .._configuration import Space
    from .._scopes import Address, ScopeRecord, Scopes
    from ..compiler import Model
    from .evaluate import Session


class PerClass:
    """One compiled model's per-class record (see the module docstring)."""

    __slots__ = ("model", "scopes", "reads", "lock", "_flow", "_session")

    def __init__(self, model: Model[Space]) -> None:
        self.model = model
        self.scopes: Scopes = model.scopes
        self.reads: ModelReads = infer(self.scopes)
        self.lock: RLock = model.lock
        self._flow: tuple[Address, ...] | None = None
        self._session: Session | None = None

    def flow(self) -> tuple[Address, ...]:
        """The model's choices in the flow order (found once per model, on first use, from
        its compiled forms: ``_whole``). Each loop in it is noted, a finding stated without
        refusing (``flow-loop``): its choices are taken in declaration order."""
        with self.lock:
            if self._flow is None:
                found = self.scopes.flow()
                for loop in found.loops:
                    keys = tuple(index.key for index in loop)
                    note(
                        Finding(
                            FindingKind.LIMITATION,
                            "flow-loop",
                            keys[0],
                            "choices whose presence, list or facts read one another: taken "
                            "in declaration order",
                            details=(("choices", keys),),
                        )
                    )
                self._flow = found.order
            return self._flow

    def presence_readers(self, source: Address) -> tuple[Address, ...]:
        """The choices whose presence may read the choice or fact ``source`` (found once
        per model, on first use, from its compiled forms: ``_whole``)."""
        with self.lock:
            return self.scopes.readers(source)

    def session(self) -> Session:
        """The model's one session (made on first use), with a node cache in its default
        mode."""
        with self.lock:
            if self._session is None:
                from .cache import NodeCache  # noqa: PLC0415 - evaluate reads the model
                from .evaluate import Session  # noqa: PLC0415

                self._session = Session(cache=NodeCache())
            return self._session

    def drop_session(self) -> None:
        """The model's session dropped, its node cache and counters with it: the next
        question makes a new one, and evaluates cold (a measurement's way to ask a question
        cold, and to release what the cache keeps before the model goes). The cache is
        taken from the dropped session too: a design space still alive keeps its
        evaluations in it (the facts layer's design space with nothing stored, which every
        membership reading evaluates, among them), and with them the session."""
        with self.lock:
            if self._session is not None:
                self._session.cache = None
            self._session = None


def scopes_below(scopes: Scopes, root: ScopeRecord) -> Collection[ScopeRecord]:
    """``root`` and every scope below it: those whose path ``root``'s prefixes (every scope
    made, no member read)."""
    return scopes.subtree(root)


def flow_below(model: Model[Space], scope: ScopeRecord) -> tuple[Address, ...]:
    """The choices below ``scope`` in the flow order: each after every choice its
    presence, its list or its Space's facts read (a selector before the choices its cases
    hold), ties in declaration order (``PerClass.flow``)."""
    flow = model.per_class.flow()
    if scope.parent is None:
        return flow
    return tuple(index for index in flow if index[0].within(scope))


__all__ = ["PerClass", "flow_below", "scopes_below"]
