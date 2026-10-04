# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Controlled native reads, purity, context propagation, and dynamic cycles."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

import pytest

from finn.core.space import (
    Decision,
    EvaluationError,
    Inapplicable,
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
    selections,
    view,
)
from finn.core.space._execution import NativeEvaluationError
from finn.core.space.occurrence import state


@pytest.mark.parametrize("field", ("caught", "finally_return", "blocked"))
def test_caught_failure_and_nonvalue_cannot_publish_fallback(field: str) -> None:
    class Family(Space):
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

    point = design_space(Family())
    if field == "blocked":
        assert isinstance(point.query(Family.blocked), Unresolved)
        assert point.with_choices(choice=1).blocked == 1
    else:
        with pytest.raises(NativeEvaluationError):
            getattr(point, field)
        index = state(point).model.linked.keys[field]
        assert index not in state(point).cache


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
        "capture",
        "restore",
        "explain",
        "assign",
        "delete",
        "foreign",
        "configure",
    ),
)
def test_driver_operations_remain_sticky_when_caught(operation: str) -> None:
    class Family(Space):
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

    foreign = design_space(Family(fact=8))
    saved = selections.capture(foreign)
    actions: dict[str, Callable[[Family], object]] = {
        "query": lambda point: point.query(Family.choice),
        "inspect": lambda point: point.inspect(Family.output),
        "state": lambda point: point.field(Family.choice).state,
        "candidates": lambda point: point.field(Family.choice).candidates(),
        "change": lambda point: point.field(Family.choice).change(1),
        "clear": lambda point: point.field(Family.choice).clear(),
        "update": lambda point: point.with_choices(choice=1),
        "capture": selections.capture,
        "restore": lambda point: selections.restore(point, saved),
        "explain": lambda point: inspection.explain(point, Family.choice),
        "assign": lambda point: setattr(point, "fact", 99),
        "delete": lambda point: delattr(point, "fact"),
        "foreign": lambda point: foreign.fact,
        # design_space() is the compile step that replaced constructing a configuration.
        "configure": lambda point: design_space(Family(fact=1)),
    }
    point = design_space(Family(fact=7))
    with pytest.raises(NativeEvaluationError, match="cross-snapshot|driver-only"):
        _ = point.invalid
    assert point.fact == 7 and isinstance(point.query(Family.choice), Unresolved)


def test_context_variables_survive_suspension_without_leaking() -> None:
    marker: ContextVar[str] = ContextVar("test_native_marker", default="driver")
    observed: list[str] = []

    class Family(Space):
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

    point = design_space(Family(fact=7))
    assert point.outer == 7 and observed == ["driver", "outer"]
    assert marker.get() == "driver"


def test_contextual_domain_error_is_one_primary_failure() -> None:
    def accepts(*, candidate: int) -> bool:
        raise LookupError("membership failure")

    class Family(Space):
        choice: int = Decision(domain=domain(accepts=accepts))

    with pytest.raises(NativeEvaluationError) as caught:
        design_space(Family()).with_choices(choice=1)
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

    class Family(Space):
        fact: list[int] = Param(semantics=semantics)

        @derived
        def output(self) -> int:
            try:
                return len(self.fact)
            except EvaluationError:
                return 99

    point = design_space(Family(fact=[1]))
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

    class Family(Space):
        choice: list[int] = Decision(
            domain=domain(accepts=lambda candidate: True), semantics=semantics
        )

    point = design_space(Family()).with_choices(choice=[1])
    armed = True
    with pytest.raises(EvaluationError, match="pure value transformation"):
        point.with_choices(choice=[1])
    armed = False
    assert point.choice == [1] and facts.value == 7


def test_blocked_self_constraint_keeps_its_inspectable_assessment() -> None:
    class Family(Space):
        choice: int = Decision(values=(1, 2))

        @constraint
        def positive(self) -> bool:
            return self.choice > 0

    assessment = design_space(Family()).inspect(Family.positive)
    assert isinstance(assessment.result, Unresolved)
    assert isinstance(assessment.results["positive"], Unresolved)


def test_membership_only_domain_enumeration_keeps_applicability_and_blockers() -> None:
    class Family(Space):
        enabled: bool = Param()
        prerequisite: int = Decision(values=(1, 2))
        choice: int = Decision(
            domain=domain(accepts=lambda candidate, value: candidate == value, value=prerequisite),
            when=enabled,
        )

    assert isinstance(
        design_space(Family(enabled=False)).field(Family.choice).candidates(), Inapplicable
    )
    point = design_space(Family(enabled=True))
    assert isinstance(point.field(Family.choice).candidates(), Unresolved)
    assert point.with_choices(prerequisite=1).field(Family.choice).candidates() is None


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
    for family in (Admission, Guard):
        with pytest.raises(EvaluationError, match="dependency cycle"):
            design_space(family()).with_choices(choice=1)
