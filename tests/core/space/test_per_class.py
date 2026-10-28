# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-class record (``_running.per_class.PerClass``), made by every compiled model
(``compiler.Model.per_class``):

- the inferred reads: each ``self`` method's reads, each address's existence guards and
  each case guard;
- the flow order, found once and noted once per loop (``flow-loop``), and every choice's
  and fact's presence readers;
- the model's one session (``session()``): made once, with a node cache; a question
  that fails in it leaves nothing the next question on the model reads (each failure is its
  run's: asked again, the function runs again).

The Space classes asked: FINN's toys, and a model whose choices read one another (a loop).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import pytest
from support.space_oracle import toys

from finn.core.reporting.manifest import FINDING
from finn.core.space import (
    Available,
    Decision,
    Param,
    Space,
    bind,
    derived,
    design_space,
    domain,
    inspection,
)
from finn.core.space._failure import NativeEvaluationError
from finn.core.space._reads import ModelReads
from finn.core.space._running import questions
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.facts import open as open_design_space
from finn.core.space._running.per_class import PerClass
from finn.core.space._template import compiled
from finn.core.space.compiler import Model, compile_model, model_of
from finn.core.space.occurrence import held


class Fed(Space):
    u: int = Param()
    b: int = Decision(domain=(1, 2))


class Looped(Space):
    """``a``'s presence reads ``fed.b``, whose Space's facts read ``a``: a loop."""

    @derived
    def through_a(self) -> int:
        return self.a

    fed = Fed(u=through_a)

    @derived
    def wide(self) -> bool:
        return self.fed.b > 1

    a: int = Decision(domain=(1, 2), when=wide)


CLASSES: dict[str, type[Space]] = {
    **{name: type(toy.make()) for name, toy in sorted(toys.TOYS.items())},
    "looped": Looped,
}


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_the_per_class_record_is_the_models_and_finds_its_flow_once(name: str) -> None:
    model = cast(Model[Space], compile_model(CLASSES[name]))
    assert isinstance(model.per_class, PerClass) and model.per_class.model is model
    assert model.per_class.lock is model.lock
    assert isinstance(model.per_class.reads, ModelReads)
    flow = model.per_class.flow()
    assert flow is model.per_class.flow()  # found once
    scopes = model.scopes
    assert set(flow) == set(scopes.decisions)
    for source in (*scopes.parameters, *scopes.decisions):
        assert set(model.per_class.presence_readers(source)) <= set(flow)


def test_a_loop_in_the_flow_order_is_noted_once(caplog: pytest.LogCaptureFixture) -> None:
    def noted(ask: Callable[[], object]) -> list[object]:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="finn.core.reporting.manifest"):
            ask()
        return [
            getattr(item, FINDING)
            for item in caplog.records
            if getattr(item, FINDING).code == "flow-loop"
        ]

    model = model_of(compiled(Looped))  # a model of its own: nothing noted yet
    mine = noted(model.per_class.flow)
    assert len(mine) == 1
    assert noted(model.per_class.flow) == []  # found once


# -- the model's one session --------------------------------------------------------------------


def test_the_model_holds_one_session_with_a_node_cache_in_its_default_mode() -> None:
    model = cast(Model[Space], compile_model(type(toys.TOYS["house"].make())))
    session = model.per_class.session()
    assert session is model.per_class.session()
    assert type(session.cache) is NodeCache and session.verify is Session().verify


CALLS: list[int] = []


class Brittle(Space):
    size: int = Decision(domain=(1, 2, 3))

    @derived
    def half(self) -> int:
        CALLS.append(self.size)
        if self.size == 3:
            raise ValueError("three has no half")
        return self.size // 2

    @derived
    def double(self) -> int:
        return self.size * 2


def test_a_failing_question_leaves_nothing_the_next_on_the_model_reads() -> None:
    CALLS.clear()
    point = bind(design_space(Brittle()), {Brittle.size: 3}).instance
    current = held(point).ds
    model = current.model
    session = model.per_class.session()
    ds = open_design_space(model, dict(current.facts))
    keys = model.scopes.keys
    report = questions.bind(ds, list(current.stored.items()), session)
    assert report.published is not None
    stored = report.published
    for _ in range(2):  # each failure is its run's: asked again, the function runs again
        with pytest.raises(NativeEvaluationError, match="three has no half"):
            evaluation(stored, session).result(keys["half"])
        assert (session.context, session.depth) == (None, 0)
        assert not (session.pending or session.failures or session.abandoned or session.stack)
        assert evaluation(stored, session).result(keys["double"]) == Available(6)
    assert CALLS == [3, 3]


def test_every_public_question_runs_in_the_models_session_under_its_lock() -> None:
    """A read, a validity and a change of a design space each run in its model's one
    session, holding the model's lock while they run; a public question that fails leaves
    nothing in the session the next reads."""
    CALLS.clear()
    point = design_space(Brittle()).with_choices(size=3)
    model = inspection.model(point)
    session = model.per_class.session()
    held: list[bool] = []
    locks: list[object] = []

    def holding(*, candidate: int) -> bool:
        # an authored function the change calls: the model's lock is held while it runs
        held.append(cast(Any, locks[0])._is_owned())
        return True

    class Locked(Space):
        size: int = Decision(domain=domain(accepts=holding))

    locks.append(inspection.model(Locked).lock)
    design_space(Locked()).with_choices(size=1)
    assert held and all(held)
    for _ in range(2):  # each failure is its run's: asked again, the function runs again
        before = session.counters.evaluated
        with pytest.raises(NativeEvaluationError, match="three has no half"):
            point.half  # noqa: B018
        assert session.counters.evaluated > before
        assert (session.context, session.depth) == (None, 0)
        assert not (session.pending or session.failures or session.abandoned or session.stack)
        assert point.double == 6
        assert inspection.validity(point).status == "valid"
    assert CALLS == [3, 3]
