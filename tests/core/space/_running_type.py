# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type (``finn.core.space._running``) asked directly, as an ``Implementation``
of the class-based oracle's interface (``support.space_oracle``): ``RUNNING``, the public
API with no cache (registered as ``public, no cache``; named
``core.space._running_type:RUNNING`` wherever an implementation is named), and the cache's
modes for the evidence.

It opens the design space a Space object holds (read at its root) and answers every
question from that value through ``_running`` alone, in its own session:

- a member's answer: ``evaluation(ds, session).result(node)``, the node at the member's
  key (a formal asked at its own address answers as a member of its scope);
- a scope's validity: ``Q.validity``, the scope by its index;
- the states: ``Q.states``; the possible values: ``Q.viable``, one choice at a time;
- a change: ``Q.change`` (the facts, then the commits, then the clears, the public batch
  order); a bind: ``Q.bind``; each report in the public ``ConfigurationResult``, its
  published value the design space value (the receiver's where refused).

``make`` builds a design space from the Space class, its facts and stored choices: the
class compiled by the front end (``compile_model``), the facts and choices prepared as a
change's values are (``_running.facts.prepare_values``), nothing evaluated.

What it takes from the compiled model rather than answering: the member list (each
scope's named members), the scope list (the model's scope records) and the keys. These
are the structure's, shared by every implementation of the same class.

One ``Session`` per implementation object, holding its node cache where it has one
(``Running(cache=...)``), with the purity check on where it is made so (``verify``; else
as the process's switch says, ``evaluate.Session.verifying``): one cache is shared by every
design space it opens or makes in a process, and its counters cover everything it answered
(``RUNNING.session.counters``: evaluations, calls by kind, the most evaluations in flight,
and the cache's lookups, hits and values hashed). The named implementations, each loadable
as ``core.space._running_type:NAME``:

- ``RUNNING``: no cache, the public API with no cache (``public, no cache``);
- ``RUNNING_CACHED``: the cache on, at its default width (``cache.WIDE``), unbounded;
- ``RUNNING_NARROW``: no answer kept whose read set holds more than one read;
- ``RUNNING_BOUNDED``: at most one read set kept per member;
- ``RUNNING_DROPPING``: half the entries dropped at random, at lookup and at keep (seed 0);
- ``RUNNING_VERIFIED``: the cache on, at its default width, unbounded, with the purity
  check on: each hit evaluated anyway and compared, and so is each node the depth guard
  abandoned;
- ``RUNNING_PLANTED``: a cache wrong on purpose (``LeavesOneOut``): each answer is kept
  with the last read of its read set (by node) left out of its key, so a design space
  differing from the one evaluated in that read alone takes its answer. The walks and the
  toys must catch it;
- ``RUNNING_PLANTED_WIDE``: the same fault, subtler (``LeavesOneOut(wider_than=4)``): a read
  left out only where the read set holds more than 4 reads;
- ``RUNNING_PLANTED_WIDTH``: a width bound applied to the key rather than to whether an
  answer is kept (``KeysFirstReads(wide=4)``): an answer whose read set holds more than 4
  reads is kept under its first 4 (by node), so a design space differing from the one
  evaluated in a later read alone takes its answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from support.space_oracle import NO_FACTS, Answers, Changed

from finn.core.space import QueryResult, RequestError, Space, inspection
from finn.core.space._running import questions as Q
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Ans, Evaluation, Session, evaluation
from finn.core.space._running.value import DesignSpace, FactsLayer
from finn.core.space._structure import Structure
from finn.core.space.edits import ConfigurationResult
from finn.core.space.ir import NodeId, NodeKind


class RunningAnswers:
    """One design space value, asked by key, answered by the running type."""

    def __init__(self, ds: DesignSpace, session: Session) -> None:
        self.ds = ds
        self.session = session
        self._scopes: dict[str, int] | None = None

    def _structure(self) -> Structure:
        return self.ds.layer.structure

    # -- what the model holds, by key (the compiled structure's) ---------------------

    def _scope_indices(self) -> dict[str, int]:
        if self._scopes is None:
            linked = self._structure()
            self._scopes = {linked.records[i].name: i for i in linked.subtree(0)}
        return self._scopes

    def members(self) -> Sequence[tuple[str, NodeKind]]:
        linked = self._structure()
        indices = {
            index
            for scope in self._scope_indices().values()
            for index in linked.scopes[scope].named_members.values()
        }
        return tuple(sorted((linked.header(i).key, linked.header(i).kind) for i in indices))

    def scopes(self) -> Sequence[str]:
        return tuple(sorted(self._scope_indices()))

    # -- the questions ---------------------------------------------------------------

    def stored(self) -> Mapping[str, object]:
        return {self.ds.key(index): value for index, value in self.ds.stored.items()}

    def query(self, key: str) -> QueryResult[object]:
        return evaluation(self.ds, self.session).result(self.ds.node(key))

    def validity(self, scope: str = "") -> inspection.Validity:
        index = self._scope_indices().get(scope)
        if index is None:
            raise KeyError(f"{scope!r}: no such scope")
        return Q.validity(evaluation(self.ds, self.session), index)

    def states(self) -> Sequence[tuple[str, inspection.ChoiceState]]:
        return Q.states(evaluation(self.ds, self.session))

    def viable(self, keys: Sequence[str]) -> Sequence[inspection.Viable]:
        return tuple(Q.viable(self.ds, self.ds.node(key), self.session) for key in keys)

    def change(
        self,
        commits: Mapping[str, object],
        clears: Sequence[str],
        facts: Mapping[str, object] = NO_FACTS,
    ) -> Changed:
        ds = self.ds
        given = {ds.node(key): value for key, value in facts.items()}
        requests: list[tuple[NodeId, object]] = [
            *((ds.node(key), value) for key, value in commits.items()),
            *((ds.node(key), Q.REMOVE) for key in clears),
        ]
        return self._changed(Q.change(ds, given, requests, self.session))

    def bind(self, stored: Mapping[str, object]) -> Changed:
        ds = self.ds
        saved = [(ds.node(key), value) for key, value in stored.items()]
        return self._changed(Q.bind(ds, saved, self.session))

    def _changed(self, report: Q.Report) -> Changed:
        """The report in the public API's shape: the published value, or the receiver where
        refused."""
        published = report.published
        shaped = ConfigurationResult(
            cast(Any, self.ds if published is None else published),
            report.accepted,
            report.outcomes,
            report.broken,
            report.dormant,
            report.in_play,
            report.opened,
        )
        after = None if published is None else RunningAnswers(published, self.session)
        return Changed(shaped, after)


@dataclass(frozen=True, eq=False)
class Running:
    """The running type, through ``_running``: its one ``session``, holding ``cache``
    (none by default), with the purity check on where ``verify`` is, named ``label``."""

    cache: NodeCache | None = None
    label: str = "public, no cache"
    verify: bool = False
    session: Session = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", Session(cache=self.cache, verify=self.verify))

    @property
    def name(self) -> str:
        return self.label

    def open(self, configuration: Space) -> Answers:
        """The design space value a root Space object holds (``occurrence.held``)."""
        from finn.core.space.occurrence import held  # noqa: PLC0415

        if configuration._scope != 0:
            raise ValueError("the running type opens a root configuration only")
        return RunningAnswers(held(configuration).ds, self.session)

    def make(
        self,
        space_class: type[Space],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        """The design space of ``space_class`` with ``facts`` and exactly ``stored``: the
        class compiled by the front end, its root's facts and the stored choices prepared
        as a change's values are; nothing evaluated or checked."""
        from finn.core.space._running.facts import prepare_values  # noqa: PLC0415
        from finn.core.space.compiler import (  # noqa: PLC0415
            compile_model,
            root_parameters,
            root_record,
        )

        declared = root_record(space_class(**cast(dict[str, Any], facts)))
        model = compile_model(space_class)
        linked = cast(Structure, model.linked)
        given = root_parameters(cast(Any, model), declared)
        missing = [
            linked.header(index).key
            for index in linked.parameters
            if linked.raw_nodes[index].required and index not in given
        ]
        if missing:
            raise RequestError(f"missing required parameters: {', '.join(missing)}")
        unknown = sorted(key for key in stored if linked.keys.get(key) not in linked.decisions)
        if unknown:
            raise KeyError(f"not choices of {space_class.__qualname__}: {unknown}")
        layer = FactsLayer.of(cast(Any, model), prepare_values(linked, given, "parameter"))
        values = prepare_values(
            linked, {linked.keys[key]: value for key, value in stored.items()}, "choice"
        )
        return RunningAnswers(DesignSpace(layer, values), self.session)


class LeavesOneOut(NodeCache):
    """A node cache wrong on purpose: each answer whose read set holds more than
    ``wider_than`` reads is kept with the last of them (by node) left out, so its key misses
    that read (``RUNNING_PLANTED``: every answer that read anything; ``RUNNING_PLANTED_WIDE``:
    only where more than 4 were read)."""

    __slots__ = ("wider_than",)

    def __init__(self, *, wider_than: int = 0) -> None:
        super().__init__()
        self.wider_than = wider_than

    def keep(self, ev: Evaluation, index: NodeId, found: Ans) -> None:
        if len(found.reads) > self.wider_than:
            found = replace(found, reads=found.reads - {max(found.reads)})
        super().keep(ev, index, found)


class KeysFirstReads(NodeCache):
    """A node cache wrong on purpose: its width bound applied to the key rather than to
    whether an answer is kept. An answer whose read set holds more than ``wide`` reads is
    kept under the first ``wide`` of them (by node), so its key misses the rest
    (``RUNNING_PLANTED_WIDTH``)."""

    __slots__ = ()

    def keep(self, ev: Evaluation, index: NodeId, found: Ans) -> None:
        if self.wide is not None and len(found.reads) > self.wide:
            found = replace(found, reads=frozenset(sorted(found.reads)[: self.wide]))
        super().keep(ev, index, found)


RUNNING: Running = Running()
RUNNING_CACHED = Running(NodeCache(), "running, cached")
RUNNING_NARROW = Running(NodeCache(wide=1), "running, cached, width 1")
RUNNING_BOUNDED = Running(NodeCache(bound=1), "running, cached, bounded 1")
RUNNING_DROPPING = Running(NodeCache(drop=0.5, seed=0), "running, cached, dropping 0.5")
RUNNING_VERIFIED = Running(NodeCache(), "running, cached, verified", verify=True)
RUNNING_PLANTED = Running(LeavesOneOut(), "running, cached, planted: a read left out")
RUNNING_PLANTED_WIDE = Running(
    LeavesOneOut(wider_than=4), "running, cached, planted: a read left out of sets over 4"
)
RUNNING_PLANTED_WIDTH = Running(
    KeysFirstReads(wide=4), "running, cached, planted: a width bound applied to the key"
)

__all__ = [
    "RUNNING",
    "RUNNING_BOUNDED",
    "RUNNING_CACHED",
    "RUNNING_DROPPING",
    "RUNNING_NARROW",
    "RUNNING_PLANTED",
    "RUNNING_PLANTED_WIDE",
    "RUNNING_PLANTED_WIDTH",
    "RUNNING_VERIFIED",
    "KeysFirstReads",
    "LeavesOneOut",
    "Running",
    "RunningAnswers",
]
