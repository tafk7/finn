# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Sparse ownership, detached captures, and atomic replay on empty roots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.core.space import (
    Available,
    Decision,
    Param,
    Selection,
    Space,
    Unresolved,
    ValueSemantics,
    derived,
    design_space,
    divisors_of,
    domain,
    inspection,
    selections,
)
from finn.core.space.errors import RequestError


def test_capture_child_keeps_each_committed_root_owner_once_and_includes_first_value() -> None:
    class Child(Space):
        supplied: int = Param()
        local: str = Decision(values=("auto", "block"))

    class Example(Space):
        factor: int = Decision(values=(1, 2))
        left = Child(supplied=factor)
        right = Child(supplied=factor)

        @derived
        def doubled(*, factor: int) -> int:
            return factor * 2

    base = design_space(Example())
    first = base.with_choices(factor=1)
    point = first.with_choices({Example.left.local: "auto"})
    assert point.doubled == 2
    captured = selections.capture(point.left)
    assert captured.keys == ("factor", "left.local")
    assert [entry.value for entry in captured.entries] == [1, "auto"]
    assert selections.capture(base).keys == ()
    replay = selections.restore(base, captured)
    assert replay.accepted
    assert selections.capture(replay.instance) == captured
    assert isinstance(replay.instance.right.query(Child.local), Unresolved)


def test_configuration_edits_can_be_captured_without_changing_earlier_points() -> None:
    class Example(Space):
        factor: int = Decision(values=(1, 2))
        style: str = Decision(values=("small", "fast"))

    base = design_space(Example())
    point = base.with_choices(factor=1, style="small")
    original = selections.capture(point)
    revised = point.with_choices(point.field(Example.style).clear(), factor=2)
    edited = selections.capture(revised)
    assert original.keys == ("factor", "style")
    assert edited.keys == ("factor",)
    assert original.value(Example.factor) == 1
    assert edited.value(Example.factor) == 2
    report = selections.restore(base, edited)
    assert report.accepted and report.instance.factor == 2
    assert point.factor == 1 and point.style == "small"
    for saved in (original, edited, selections.capture(base)):
        with pytest.raises(RequestError, match="no committed choices"):
            selections.restore(point, saved)
    with pytest.raises(KeyError, match="style"):
        edited.value(Example.style)


def test_rebound_inputs_can_refuse_an_earlier_selection_without_partial_publication() -> None:
    class Example(Space):
        extent: int = Param()
        factor: int = Decision(domain=divisors_of(extent))
        style: str = Decision(values=("auto", "block"))

    first = design_space(Example(extent=12)).with_choices(factor=4).with_choices(style="block")
    changed = design_space(Example(extent=10))
    report = selections.restore(changed, selections.capture(first))
    assert not report.accepted and report.instance is changed
    assert selections.capture(changed).keys == ()
    assert first.factor == 4 and first.extent == 12


def test_selector_change_requires_explicit_case_clearing_before_capture() -> None:
    class Child(Space):
        lanes: int = Decision(values=(1, 2))

    class Example(Space):
        left = Child()  # a class attribute naming a candidate: a typed handle
        implementation: Child = Decision({"left": left, "right": Child()})

    base = design_space(Example())
    selector = inspection.choices(base)[0].selector
    point = base.with_choices({Example.implementation: "left", Example.left.lanes: 1})
    captured = selections.capture(point)
    assert captured.keys == ("implementation", "implementation.left.lanes")
    refused = point.try_with_choices({selector: "right"})
    assert not refused.accepted and refused.instance is point
    assert selections.capture(point) == captured
    revised = point.with_choices(point.left.field(Child.lanes).clear(), {selector: "right"})
    cleaned = selections.capture(revised)
    accepted = selections.restore(base, cleaned)
    assert accepted.accepted and cleaned.keys == ("implementation",)


def test_singleton_choices_persist_their_selector_like_any_decision() -> None:
    # Replaces "singleton choices do not create persisted selectors": a singleton
    # Decision over nodes is an ordinary Decision, committed and captured like any other.
    # Uncommitted, its one case is forced: read, never captured.
    class Child(Space):
        lanes: int = Decision(values=(1,))

    class Example(Space):
        implementation: Child = Decision({"only": Child()})

    base = design_space(Example())
    assert base.query(Example.implementation.lanes) == Available(1)
    assert selections.capture(base).keys == ()
    point = base.with_choices({Example.implementation: "only", Example.implementation.lanes: 1})
    captured = selections.capture(point)
    assert captured.keys == ("implementation", "implementation.only.lanes")
    assert inspection.decision_info(point, Example.implementation).selector
    assert not inspection.decision_info(point, Example.implementation.lanes).selector


@dataclass
class Payload:
    values: list[int]


PAYLOAD = ValueSemantics(
    Payload,
    "payload",
    lambda value: type(value) is Payload,
    lambda left, right: sorted(left.values) == sorted(right.values),
    lambda value: Payload(list(value.values)),
)


def test_capture_and_public_entries_detach_mutable_payloads() -> None:
    class Example(Space):
        payload: Payload = Decision(values=(Payload([1, 2]), Payload([3])), semantics=PAYLOAD)

    base = design_space(Example())
    source = Payload([2, 1])
    point = base.with_choices(payload=source)
    captured = selections.capture(point)
    source.values.append(99)
    public = captured.entries[0].value
    assert isinstance(public, Payload)
    public.values.append(99)
    captured.value(Example.payload).values.append(99)
    assert captured.value(Example.payload).values == [2, 1]
    replacement = Payload([3])
    edited = selections.capture(point.with_choices(payload=replacement))
    replacement.values.append(99)
    assert edited.value(Example.payload).values == [3]
    assert selections.restore(base, edited).instance.payload.values == [3]
    assert point.payload.values == [2, 1]
    equal = selections.capture(point.with_choices(payload=Payload([1, 2])))
    assert equal == captured


def test_foreign_models_invalid_selections_and_child_restore_are_rejected() -> None:
    class Child(Space):
        value: int = Decision(values=(1, 2))

    class Example(Space):
        first = Child()
        second = Child()

    class OtherExample(Space):
        first = Child()
        second = Child()

    point = design_space(Example()).with_choices({Example.first.value: 1})
    saved = selections.capture(point)
    other = design_space(OtherExample())
    with pytest.raises(RequestError, match="different compiled model"):
        selections.restore(other, saved)
    with pytest.raises(RequestError, match="root configuration"):
        selections.restore(point.first, saved)
    with pytest.raises(RequestError):
        saved.value(Child.value)
    with pytest.raises(RequestError, match="different compiled model"):
        foreign = inspection.decision_handle(other, OtherExample.first.value)
        saved.value(foreign)
    with pytest.raises(RequestError, match="Selection"):
        selections.restore(other, cast(Selection, object()))


def test_capture_does_not_evaluate_unrelated_uncommitted_guard_callbacks() -> None:
    class Example(Space):
        committed: int = Decision(values=(1,))

        @derived
        def explosive() -> bool:
            raise AssertionError("unrelated capture must not query this guard")

        unrelated: int = Decision(values=(1,), when=explosive)

    point = design_space(Example()).with_choices(committed=1)
    assert selections.capture(point).keys == ("committed",)


def test_restore_preconditions_precede_value_adapters_and_admission() -> None:
    events: list[str] = []

    def snapshot(value: int) -> int:
        events.append("snapshot")
        return value

    def accepts(*, candidate: int) -> bool:
        events.append("admission")
        return candidate > 0

    semantics = ValueSemantics(
        int, "integer", lambda value: type(value) is int, int.__eq__, snapshot
    )

    class Example(Space):
        value: int = Decision(domain=domain(accepts=accepts), semantics=semantics)

    base = design_space(Example())
    configured = base.with_choices(value=1)
    same = selections.capture(configured)
    different = selections.capture(base.with_choices(value=2))
    empty = selections.capture(base)
    events.clear()
    for saved in (same, different, empty):
        with pytest.raises(RequestError, match="no committed choices"):
            selections.restore(configured, saved)
    assert events == []
    # Querying the base does not make it ineligible, and empty replay is a no-op.
    assert isinstance(base.query(Example.value), Unresolved)
    assert selections.restore(base, empty).instance is base
    assert events == []
    assert selections.restore(base, same).instance.value == 1
    cleared = configured.with_choices(configured.field(Example.value).clear())
    assert selections.restore(cleared, different).instance.value == 2


def test_singleton_structural_selection_is_committed_and_replays_on_an_empty_root() -> None:
    # Replaces "selecting the only case is a no-op": a singleton Decision over nodes
    # needs a commitment now; the committed selector replays like any other choice.
    class Child(Space):
        value: int = Decision(values=(1,))

    class Example(Space):
        choice: Child = Decision({"only": Child()})

    base = design_space(Example())
    committed = base.with_choices(choice="only")
    assert committed is not base and selections.capture(committed).keys == ("choice",)
    configured = committed.with_choices({Example.choice.value: 1})
    result = selections.restore(base, selections.capture(configured))
    only = result.instance.choice
    assert result.accepted and isinstance(only, Child) and only.field(Child.value).get() == 1
