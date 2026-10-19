# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from finn.core.space import (
    Decision,
    Derived,
    Param,
    Space,
    ValueSemantics,
    composite,
    constraint,
    design_space,
    domain,
)
from finn.core.space.edits import ChangeRequest
from finn.core.space.errors import (
    DefinitionError,
    EvaluationError,
    RequestError,
)
from finn.core.space.inspection import candidate, choices, decision_handle
from finn.core.space.results import Available, Inapplicable, Unresolved


def test_malformed_batch_structure_and_types_precede_every_snapshot_callback() -> None:
    events: list[str] = []

    def snapshot(value: int) -> int:
        events.append("snapshot")
        return value

    def membership(*, candidate: int) -> bool:
        events.append("membership")
        return True

    semantics = ValueSemantics(
        int, "integer", lambda value: type(value) is int, lambda a, b: a == b, snapshot
    )

    class Trial(Space):
        first: int = Decision(domain=domain(accepts=membership), semantics=semantics)
        second: int = Decision(values=(1, 2))

    base, other = design_space(Trial()), design_space(Trial())
    first = base.field(Trial.first).change(1)
    bad_edits: tuple[ChangeRequest, ...] = (
        first,
        replace(first, scope=True),
        replace(first, scope=-1),
        replace(first, node=True),
        replace(first, node=10**6),
        other.field(Trial.second).change(1),
        base.field(Trial.second).change(cast(int, "wrong")),
        cast(ChangeRequest, object()),
    )
    for bad in bad_edits:
        with pytest.raises(RequestError):
            base.try_with_choices(first, bad)
        assert events == []
    assert isinstance(base.query(Trial.first), Unresolved)


def test_foreign_model_handles_are_rejected_before_evaluators() -> None:
    calls: list[int] = []

    def membership(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Trial(Space):
        value: int = Decision(domain=domain(accepts=membership))

    class Other(Trial):
        pass

    point = design_space(Other())
    handle = decision_handle(Trial, Trial.value)
    with pytest.raises(RequestError):
        point.field(handle)
    with pytest.raises(RequestError):
        point.try_with_choices({handle: 1})
    assert calls == []


def test_snapshot_and_recognition_programmer_failures_are_contextual() -> None:
    memberships: list[int] = []

    def membership(*, candidate: int) -> bool:
        memberships.append(candidate)
        return True

    def broken(value: int) -> int:
        raise RuntimeError("adapter failed")

    semantics = ValueSemantics(
        int, "integer", lambda value: type(value) is int, lambda a, b: a == b, broken
    )

    class Inputs(Space):
        value: int = Param(semantics=semantics)

    class Decisions(Space):
        earlier: int = Decision(domain=domain(accepts=membership))
        value: int = Decision(domain=domain(accepts=lambda *, candidate: True), semantics=semantics)

    # An unrecognized literal is a bad binding at the node call.
    with pytest.raises(DefinitionError):
        Inputs(value=cast(int, "wrong"))
    with pytest.raises(EvaluationError) as input_error:
        design_space(Inputs(value=1))
    assert input_error.value.owner == "value"
    assert input_error.value.role == "parameter snapshot"
    assert isinstance(input_error.value.__cause__, RuntimeError)
    base = design_space(Decisions())
    with pytest.raises(EvaluationError) as candidate_error:
        base.try_with_choices(
            base.field(Decisions.earlier).change(1), base.field(Decisions.value).change(1)
        )
    assert candidate_error.value.role == "choice snapshot"
    assert isinstance(candidate_error.value.__cause__, RuntimeError)
    assert memberships == []

    def unrecognizable(value: object) -> bool:
        raise LookupError("recognition failed")

    recognition: ValueSemantics[int] = ValueSemantics(
        int, "integer", unrecognizable, lambda a, b: a == b, lambda a: a
    )

    class Unrecognizable(Space):
        value: int = Param(semantics=recognition)

    with pytest.raises(EvaluationError) as recognition_error:
        design_space(Unrecognizable(value=1))
    assert recognition_error.value.role == "parameter recognition"
    assert isinstance(recognition_error.value.__cause__, LookupError)


def test_replacement_equality_is_contextual_and_cannot_mutate_stored_values() -> None:
    fail = False

    def equal(left: list[int], right: list[int]) -> bool:
        if fail:
            raise RuntimeError("equality failed")
        result = left == right
        left.append(99)
        right.append(98)
        return result

    semantics: ValueSemantics[list[int]] = ValueSemantics(
        list, "integers", lambda value: type(value) is list, equal, list
    )

    class Trial(Space):
        value: list[int] = Decision(
            domain=domain(accepts=lambda *, candidate: True), semantics=semantics
        )

    base = design_space(Trial())
    chosen = base.try_with_choices(base.field(Trial.value).change([1])).instance
    assert chosen.try_with_choices(chosen.field(Trial.value).change([1])).instance is chosen
    assert chosen.value == [1]
    conflict = chosen.try_with_choices(chosen.field(Trial.value).change([2]))
    assert conflict.accepted and conflict.instance.value == [2]
    assert chosen.value == [1]
    fail = True
    with pytest.raises(EvaluationError) as raised:
        chosen.try_with_choices(chosen.field(Trial.value).change([1]))
    assert raised.value.owner == "value"
    assert raised.value.role == "configuration equality"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert chosen.value == [1]


def test_independent_batch_reuses_trial_dependencies_and_publishes_only_once() -> None:
    limits: list[int] = []
    admitted: list[int] = []

    def limit_value(*, seed: int) -> int:
        limits.append(seed)
        return seed

    def membership(*, candidate: int, limit: int) -> bool:
        admitted.append(candidate)
        return 0 < candidate <= limit

    class Seeded(Space):
        seed: int = Param()
        limit = Derived(limit_value)

    limit = Seeded.limit
    members: list[int] = [
        Decision(domain=domain(accepts=membership, limit=limit)) for _ in range(64)
    ]
    space_type = composite(
        "IndependentBatch",
        {f"choice_{index}": member for index, member in enumerate(members)},
        annotations={f"choice_{index}": int for index in range(len(members))},
        base=Seeded,
    )
    base = design_space(space_type(seed=100))
    report = base.try_with_choices(
        *(
            base.field(member).change(index + 1)
            for index, member in reversed(list(enumerate(members)))
        )
    )
    assert report.accepted
    assert limits == [100]
    assert len(admitted) == len(members)
    assert all(outcome.status == "changed" for outcome in report.outcomes)
    assert report.instance.query(limit) == Available(100)
    assert limits == [100]  # the successor keeps the change's evaluation: it read no choice

    limits.clear()
    admitted.clear()
    edits = [base.field(member).change(index + 1) for index, member in enumerate(members)]
    edits[-1] = base.field(members[-1]).change(-1)
    failed = base.try_with_choices(*edits)
    assert not failed.accepted
    assert failed.instance is base
    # The model's memo answers the limit, a function of the same fact, as before, and the
    # membership of every value the first batch checked: equal values, by their semantics.
    assert limits == []
    assert admitted == [-1]
    assert [outcome.status for outcome in failed.outcomes].count("checked") == len(members) - 1
    assert all(isinstance(base.query(member), Unresolved) for member in members)


def test_a_chain_of_related_choices_is_order_independent_as_its_constraints() -> None:
    """Each step one more than the last: a relation of a choice to another, so a
    constraint of the Space holding both (a domain reads facts only). Set in one batch,
    in any order, the chain is accepted; one step off breaks the rule beside it."""
    calls: list[int] = []
    steps: tuple[int, ...] = tuple(range(1, 34))
    members = [Decision(values=steps) for _ in range(32)]
    rules: dict[str, object] = {}

    def follows(at: int) -> Callable[..., bool]:
        def rule(*, step: int, previous: int) -> bool:
            calls.append(at)
            return step == previous + 1

        return rule

    for index in range(1, len(members)):
        rules[f"follows_{index}"] = constraint(step=members[index], previous=members[index - 1])(
            follows(index)
        )
    space_type = composite(
        "DependentBatch",
        {**{f"step_{i}": member for i, member in enumerate(members)}, **rules},
        annotations={f"step_{i}": int for i in range(len(members))},
    )
    base = design_space(space_type())
    edits = [base.field(member).change(index + 1) for index, member in enumerate(members)]
    report = base.try_with_choices(*reversed(edits))
    assert report.accepted
    assert sorted(calls) == list(range(1, len(members)))  # each rule once
    assert report.instance.query(members[-1]) == Available(len(members))
    edits[0] = base.field(members[0]).change(2)
    failed = base.try_with_choices(*reversed(edits))
    assert not failed.accepted and failed.instance is base
    assert all(outcome.status == "checked" for outcome in failed.outcomes)
    assert [(rule.rule, rule.standing) for rule in failed.broken] == [("follows_1", ())]


def test_selector_and_nested_edit_share_atomic_order_and_inactive_edits_are_dormant() -> None:
    class Child(Space):
        value: int = Decision(values=(1, 2))

    class Root(Space):
        implementation: Child = Decision({"a": Child(), "b": Child()})

    base = design_space(Root())
    selector = choices(base)[0].selector
    child = candidate(base, Root.implementation, "a")
    assert child is not None
    report = base.try_with_choices(child.field(Child.value).change(2), {selector: "a"})
    assert report.accepted
    selected = candidate(report.instance, Root.implementation, "a")
    assert selected is not None and selected.query(Child.value) == Available(2)
    inactive = candidate(report.instance, Root.implementation, "b")
    assert inactive is not None
    # Set under the case not selected, the value is checked there and kept, dormant.
    dormant = report.instance.try_with_choices(inactive.field(Child.value).change(1))
    assert dormant.accepted
    other = candidate(dormant.instance, Root.implementation, "b")
    assert other is not None and isinstance(other.query(Child.value), Inapplicable)
    # Switching to it puts a's value out of play and brings b's in.
    switched = dormant.instance.with_choices({selector: "b"})
    assert candidate(switched, Root.implementation, "b").query(Child.value) == Available(1)  # type: ignore[union-attr]
    assert isinstance(
        candidate(switched, Root.implementation, "a").query(Child.value),  # type: ignore[union-attr]
        Inapplicable,
    )


def test_enumerated_candidates_still_pass_membership_before_commitment() -> None:
    seen: list[int] = []

    def membership(*, candidate: int) -> bool:
        seen.append(candidate)
        return candidate > 0

    class Trial(Space):
        value: int = Decision(domain=domain(accepts=membership, candidates=lambda: (-1, 1)))

    base = design_space(Trial())
    assert base.field(Trial.value).candidates() == Available((-1, 1))
    assert seen == []
    report = base.try_with_choices(base.field(Trial.value).change(-1))
    assert not report.accepted
    assert seen == [-1]
    # Nothing is filled in, though one listed value alone passes membership.
    assert isinstance(base.query(Trial.value), Unresolved)


def test_programmer_failure_after_provisional_admission_never_publishes_a_successor() -> None:
    def broken(*, candidate: int) -> bool:
        raise RuntimeError("membership failed")

    class Trial(Space):
        first: int = Decision(values=(1, 2))
        second: int = Decision(domain=domain(accepts=broken))

    base = design_space(Trial())
    with pytest.raises(EvaluationError):
        base.try_with_choices(base.field(Trial.first).change(1), base.field(Trial.second).change(1))
    assert isinstance(base.query(Trial.first), Unresolved)
    assert isinstance(base.query(Trial.second), Unresolved)
    assert base.try_with_choices().instance is base


def test_selector_report_uses_authored_owner() -> None:
    class Root(Space):
        implementation: Space = Decision({"a": Space(), "b": Space()})

    point = design_space(Root())
    selector = choices(point)[0].selector
    report = point.try_with_choices({selector: "a"})
    assert report.accepted
    assert report.outcomes[0].owner == "implementation"
