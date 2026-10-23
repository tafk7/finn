# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Controlled native reads, purity, context propagation, and dynamic cycles."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    NativeEvaluationError,
    Param,
    Space,
    Unresolved,
    ValueSemantics,
    ValueUnavailableError,
    constraint,
    derived,
    design_space,
    domain,
    inspection,
    view,
)
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Ans, Session, evaluation
from finn.core.space._scopes import Address
from finn.core.space.occurrence import ask, held


@pytest.mark.parametrize("field", ("caught", "finally_return", "blocked"))
def test_caught_failure_and_nonvalue_cannot_publish_fallback(field: str) -> None:
    class Example(Space):
        choice: int = Decision(values=(1, 2))

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def caught(self) -> int:
            try:
                return self.broken
            except EvaluationError:
                return 99

        @derived
        def finally_return(self) -> int:
            try:
                return self.broken
            finally:
                return 99

        @derived
        def blocked(self) -> int:
            try:
                return self.choice
            except ValueUnavailableError:
                return 99

    point = design_space(Example())
    if field == "blocked":
        assert isinstance(point.query(Example.blocked), Unresolved)
        assert point.with_choices(choice=1).blocked == 1
    else:
        with pytest.raises(NativeEvaluationError):
            getattr(point, field)
        # A failure is never kept: the design space's evaluation holds no answer for it.
        ds = held(point).ds
        index = ds.node(field)
        assert index not in ask(ds.model, lambda session: evaluation(ds, session).memo)


@pytest.mark.parametrize(
    "operation",
    (
        "query",
        "inspect",
        "state",
        "candidates",
        "change",
        "clear",
        "update",
        "explain",
        "assign",
        "delete",
        "foreign",
        "configure",
    ),
)
def test_driver_operations_remain_sticky_when_caught(operation: str) -> None:
    class Example(Space):
        fact: int = Param()
        choice: int = Decision(values=(1, 2))

        @view
        def output(self) -> int:
            return self.fact

        @derived
        def invalid(self) -> int:
            try:
                actions[operation](self)
            except (EvaluationError, AttributeError):
                return 99
            return 0

    foreign = design_space(Example(fact=8))
    actions: dict[str, Callable[[Example], object]] = {
        "query": lambda point: point.query(Example.choice),
        "inspect": lambda point: point.inspect(Example.output),
        "state": lambda point: point.field(Example.choice).state,
        "candidates": lambda point: point.field(Example.choice).candidates(),
        "change": lambda point: point.field(Example.choice).change(1),
        "clear": lambda point: point.field(Example.choice).clear(),
        "update": lambda point: point.with_choices(choice=1),
        "explain": lambda point: inspection.explain(point, Example.choice),
        "assign": lambda point: setattr(point, "fact", 99),
        "delete": lambda point: delattr(point, "fact"),
        "foreign": lambda point: foreign.fact,
        # design_space() is the compile step: a configuration is never constructed.
        "configure": lambda point: design_space(Example(fact=1)),
    }
    point = design_space(Example(fact=7))
    with pytest.raises(NativeEvaluationError, match="cross-configuration|driver-only"):
        _ = point.invalid
    assert point.fact == 7 and isinstance(point.query(Example.choice), Unresolved)


def test_context_variables_survive_suspension_without_leaking() -> None:
    marker: ContextVar[str] = ContextVar("test_native_marker", default="driver")
    observed: list[str] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def inner(self) -> int:
            observed.append(marker.get())
            marker.set("inner")
            return self.fact

        @derived
        def outer(self) -> int:
            token = marker.set("outer")
            try:
                value = self.inner
                observed.append(marker.get())
                return value
            finally:
                marker.reset(token)

    point = design_space(Example(fact=7))
    assert point.outer == 7 and observed == ["driver", "outer"]
    assert marker.get() == "driver"


def test_contextual_domain_error_is_one_primary_failure() -> None:
    def accepts(*, candidate: int) -> bool:
        raise LookupError("membership failure")

    class Example(Space):
        choice: int = Decision(domain=domain(accepts=accepts))

    with pytest.raises(NativeEvaluationError) as caught:
        design_space(Example()).with_choices(choice=1)
    assert caught.value.owner == "choice" and caught.value.role == "domain membership"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()


def test_public_snapshot_failure_cannot_be_caught_into_success() -> None:
    armed = False

    def snapshot(value: list[int]) -> list[int]:
        if armed:
            raise LookupError("snapshot failure")
        return list(value)

    semantics = ValueSemantics(
        list, "list", lambda value: type(value) is list, lambda left, right: left == right, snapshot
    )

    class Example(Space):
        fact: list[int] = Param(semantics=semantics)

        @derived
        def output(self) -> int:
            try:
                return len(self.fact)
            except EvaluationError:
                return 99

    point = design_space(Example(fact=[1]))
    armed = True
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.output
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.primary.owner == "fact"
    assert caught.value.cleanup_failures == ()


@pytest.mark.parametrize("hook", ("recognition", "snapshot", "equality"))
def test_semantic_transformations_cannot_read_configuration_even_when_caught(hook: str) -> None:
    armed = False

    class Facts(Space):
        value: int = Param()

    facts = design_space(Facts(value=7))

    def probe(role: str) -> None:
        if armed and hook == role:
            try:
                _ = facts.value
            except EvaluationError:
                pass

    def recognizes(value: object) -> bool:
        probe("recognition")
        return type(value) is list

    def snapshot(value: list[int]) -> list[int]:
        probe("snapshot")
        return list(value)

    def equal(left: list[int], right: list[int]) -> bool:
        probe("equality")
        return left == right

    semantics = ValueSemantics(list, "list", recognizes, equal, snapshot)

    class Example(Space):
        choice: list[int] = Decision(
            domain=domain(accepts=lambda candidate: True), semantics=semantics
        )

    point = design_space(Example()).with_choices(choice=[1])
    armed = True
    with pytest.raises(EvaluationError, match="pure value transformation"):
        point.with_choices(choice=[1])
    armed = False
    assert point.choice == [1] and facts.value == 7


def test_blocked_self_constraint_keeps_its_inspectable_assessment() -> None:
    class Example(Space):
        choice: int = Decision(values=(1, 2))

        @constraint
        def positive(self) -> bool:
            return self.choice > 0

    assessment = design_space(Example()).inspect(Example.positive)
    assert isinstance(assessment.result, Unresolved)
    assert isinstance(assessment.results["positive"], Unresolved)


def test_membership_only_domain_enumeration_keeps_applicability_and_blockers() -> None:
    class Example(Space):
        enabled: bool = Param()
        prerequisite: int | None = Param(default=None)
        choice: int = Decision(
            domain=domain(accepts=lambda candidate, value: candidate == value, value=prerequisite),
            when=enabled,
        )

    assert isinstance(
        design_space(Example(enabled=False)).field(Example.choice).candidates(), Inapplicable
    )
    point = design_space(Example(enabled=True))
    assert point.field(Example.choice).candidates() is None  # known by membership only
    assert point.with_facts({Example.prerequisite: 1}).field(Example.choice).candidates() is None


def test_self_cycles_are_reached_through_scopes_admission_and_guards() -> None:
    class Child(Space):
        source: int = Param()

        @derived
        def value(self) -> int:
            return self.source

    class Parent(Space):
        @derived
        def output(self) -> int:
            return self.child.value

        child = Child(source=output)

    class Admission(Space):
        @derived
        def extent(self) -> int:
            return self.choice

        choice: int = Decision(
            domain=domain(accepts=lambda candidate, value: candidate == value, value=extent)
        )

    class Guard(Space):
        @derived
        def enabled(self) -> bool:
            return self.choice > 0

        choice: int = Decision(values=(1,), when=enabled)

    point = design_space(Parent())
    with pytest.raises(EvaluationError, match="dependency cycle") as scoped:
        _ = point.output
    assert "child.value" in str(scoped.value)
    # A domain reading its own choice reads a choice: refused when the model is made.
    with refused("domain-reads-choice", error=DefinitionError):
        design_space(Admission())
    with pytest.raises(EvaluationError, match="dependency cycle"):
        design_space(Guard()).with_choices(choice=1)


def test_an_answer_a_query_keeps_has_read_what_a_fresh_evaluation_reads() -> None:
    """A guard a query evaluates on its way is kept with the reads (its trace, and the
    choices and facts it read) an evaluation from nothing records for it."""

    class Example(Space):
        want: bool = Decision(values=(True, False))
        choice: int = Decision(values=(1, 2), when=want)

    def read(found: Ans) -> tuple[Address, ...]:
        return tuple(dict.fromkeys(node for node, _ in found.trace))

    point = design_space(Example())
    assert isinstance(point.query(Example.choice), Unresolved)
    ds = held(point).ds
    index = ds.node("choice.$guard")
    kept = ask(ds.model, lambda session: evaluation(ds, session).memo[index])
    with ds.model.lock:
        alone = evaluation(ds, Session(cache=NodeCache())).get(index)
    assert (read(kept), kept.reads) == (read(alone), alone.reads)
    assert kept.reads == {ds.node("want")}
