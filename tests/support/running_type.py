# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type (``finn.core.space._running``) as an ``Implementation`` of the
class-based oracle's interface (``support.space_oracle``): ``RUNNING``, named
``support.running_type:RUNNING`` wherever an implementation is named.

It opens today's configuration as the design space value it is (its model, facts and
stored choices, read once) and answers every question from that value through
``_running`` alone:

- a member's answer: ``evaluation(ds, session).result(node)``, the node at the member's
  key (a formal asked at its own address answers as a member of its scope);
- a scope's validity: ``Q.validity``, the scope by its index;
- the states: ``Q.states``; the possible values: ``Q.viable``, one choice at a time;
- a change: ``Q.change`` (the facts, then the commits, then the clears, today's batch
  order); a bind: ``Q.bind``; each report in today's ``ConfigurationResult``, its
  published value the design space value (the receiver's where refused).

``make`` builds a design space from the Space class, its facts and stored choices: the
class compiled by today's front end (``compile_model``), the facts and choices prepared as
a change's values are, and nothing of today's engine evaluated.

What it takes from the compiled model rather than answering: the member list (each
scope's named members), the scope list (the model's scope records) and the keys. These
are the structure's, shared by every implementation of the same class.

One ``Session`` per implementation object: its counters cover everything it answered
(``RUNNING.session.counters``: evaluations, calls by kind, the most evaluations in flight).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from finn.core.space import QueryResult, RequestError, Space, inspection
from finn.core.space._running import questions as Q
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.value import DesignSpace, FactsLayer
from finn.core.space._structure import Structure
from finn.core.space.edits import ConfigurationResult
from finn.core.space.ir import NodeId, NodeKind
from support.space_oracle import NO_FACTS, Answers, Changed


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
        """The report in today's shape: the published value, or the receiver where
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
    """The running type, through ``_running``, with no cache: its one ``session``."""

    session: Session = field(default_factory=Session)

    @property
    def name(self) -> str:
        return "running"

    def open(self, configuration: Space) -> Answers:
        """The design space value of today's root configuration: its model, facts and
        stored choices (dormant ones included), read once."""
        from finn.core.space.occurrence import state  # noqa: PLC0415 - today's, read once

        if configuration._scope != 0:
            raise ValueError("the running type opens a root configuration only")
        current = state(configuration)
        layer = FactsLayer.of(current.model, current.world.facts)
        return RunningAnswers(DesignSpace(layer, current.world.stored), self.session)

    def make(
        self,
        space_class: type[Space],
        facts: Mapping[str, object],
        stored: Mapping[str, object],
    ) -> Answers:
        """The design space of ``space_class`` with ``facts`` and exactly ``stored``: the
        class compiled by today's front end, its root's facts and the stored choices
        prepared as a change's values are; nothing evaluated or checked."""
        from finn.core.space import occurrence  # noqa: PLC0415
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
        layer = FactsLayer.of(
            cast(Any, model), occurrence.prepare_values(linked, given, "parameter")
        )
        values = occurrence.prepare_values(
            linked, {linked.keys[key]: value for key, value in stored.items()}, "choice"
        )
        return RunningAnswers(DesignSpace(layer, values), self.session)


RUNNING = Running()

__all__ = ["RUNNING", "Running", "RunningAnswers"]
