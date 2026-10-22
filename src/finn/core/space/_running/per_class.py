# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-class record: what a compiled model answers once for every design space of it,
made with the model (``compiler.Model``) and held by it for its life.

- ``reads``: what each ``self`` method may read, each node's existence guards, and each
  case guard (``_reads.infer``; a structure's answered node by node, as read, each under
  the model's lock);
- ``flow()``: the model's choices in the flow order, found once, on first use, from its
  templates, each loop in it noted (``flow-loop``);
- ``presence_readers(source)``: the choices whose presence may read the choice or fact
  ``source``, from its templates;
- ``session()``: the model's one ``evaluate.Session``, made on first use: every public
  question about a design space of the model runs in it, under the model's lock, and its
  node cache (``cache.NodeCache``, made in its default mode) keeps answers for the model's
  life.

A table built by hand has no templates: it answers no question about its whole design space
(its flow order, its presence readers). Each part is made under the model's lock (``lock``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.reporting.manifest import note

from .._reads import ModelReads, infer
from ..errors import RequestError
from ..ir import Linked, NodeId, Questions

if TYPE_CHECKING:
    from threading import RLock

    from .._configuration import Space
    from ..compiler import Model
    from .evaluate import Session


class PerClass:
    """One compiled model's per-class record (see the module docstring)."""

    __slots__ = ("model", "linked", "reads", "lock", "_flow", "_session")

    def __init__(self, model: Model[Space]) -> None:
        self.model = model
        self.linked: Linked = model.linked
        self.reads: ModelReads = infer(self.linked)
        self.lock: RLock = model.lock
        self._flow: tuple[NodeId, ...] | None = None
        self._session: Session | None = None

    def _whole(self) -> Questions:
        whole = self.linked.whole
        if whole is None:
            raise RequestError(
                "a table built by hand has no templates: it answers no question about its "
                "whole design space (its flow order, its presence readers)"
            )
        return whole

    def flow(self) -> tuple[NodeId, ...]:
        """The model's choices in the flow order (found once per model, on first use, from
        its templates: ``_whole``). Each loop in it is noted, a finding stated without
        refusing (``flow-loop``): its choices are taken in declaration order."""
        with self.lock:
            if self._flow is None:
                found = self._whole().flow()
                for loop in found.loops:
                    keys = tuple(self.linked.header(index).key for index in loop)
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

    def presence_readers(self, source: NodeId) -> tuple[NodeId, ...]:
        """The choices whose presence may read the choice or fact ``source`` (found once
        per model, on first use, from its templates: ``_whole``)."""
        with self.lock:
            return self._whole().readers(source)

    def session(self) -> Session:
        """The model's one session (made on first use), with a node cache in its default
        mode."""
        with self.lock:
            if self._session is None:
                from .cache import NodeCache  # noqa: PLC0415 - evaluate reads the model
                from .evaluate import Session  # noqa: PLC0415

                self._session = Session(cache=NodeCache())
            return self._session


__all__ = ["PerClass"]
