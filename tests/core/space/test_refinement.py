# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from finn.core.space import (
    Decision,
    Derived,
    Param,
    Space,
    Subspace,
    SubspaceChoice,
    ValueSemantics,
    compile_space,
    domain,
    refinement,
)
from finn.core.space.edits import ChangeRequest
from finn.core.space.errors import EvaluationError, ConfigurationError, RequestError
from finn.core.space.inspection import choices, decision_handle
from finn.core.space.results import Available, Inapplicable, Rejected, Unresolved


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
        first = Decision(semantics, domain=domain(accepts=membership))
        second = Decision(int, values=(1, 2))

    model = compile_space(Trial)
    base, other = model.bind(), model.bind()
    first = refinement.change(base, Trial.first, 1)
    bad_edits: tuple[ChangeRequest, ...] = (
        first,
        replace(first, scope=True),
        replace(first, scope=-1),
        replace(first, node=True),
        replace(first, node=10**6),
        refinement.change(other, Trial.second, 1),
        refinement.change(base, Trial.second, cast(int, "wrong")),
        cast(ChangeRequest, object()),
    )
    for bad in bad_edits:
        with pytest.raises(RequestError):
            refinement.commit(base, first, bad)
        assert events == []
    assert isinstance(base.query(Trial.first), Unresolved)


def test_foreign_model_handles_are_rejected_before_evaluators() -> None:
    calls: list[int] = []

    def membership(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Trial(Space):
        value = Decision(int, domain=domain(accepts=membership))

    class Other(Trial):
        pass

    left, right = compile_space(Trial), compile_space(Other)
    point = right.bind()
    handle = decision_handle(left, Trial.value)
    with pytest.raises(RequestError):
        refinement.change(point, handle, 1)
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
        value = Param(semantics)

    class Decisions(Space):
        earlier = Decision(int, domain=domain(accepts=membership))
        value = Decision(semantics, domain=domain(accepts=lambda *, candidate: True))

    with pytest.raises(RequestError):
        Inputs({Inputs.value: "wrong"})
    with pytest.raises(EvaluationError) as input_error:
        Inputs({Inputs.value: 1})
    assert input_error.value.owner == "value"
    assert input_error.value.role == "parameter snapshot"
    assert isinstance(input_error.value.__cause__, RuntimeError)
    base = Decisions()
    with pytest.raises(EvaluationError) as candidate_error:
        refinement.commit(
            base,
            refinement.change(base, Decisions.earlier, 1),
            refinement.change(base, Decisions.value, 1),
        )
    assert candidate_error.value.role == "candidate snapshot"
    assert isinstance(candidate_error.value.__cause__, RuntimeError)
    assert memberships == []

    def unrecognizable(value: object) -> bool:
        raise LookupError("recognition failed")

    recognition: ValueSemantics[int] = ValueSemantics(
        int, "integer", unrecognizable, lambda a, b: a == b, lambda a: a
    )

    class Unrecognizable(Space):
        value = Param(recognition)

    with pytest.raises(EvaluationError) as recognition_error:
        Unrecognizable({Unrecognizable.value: 1})
    assert recognition_error.value.role == "parameter recognition"
    assert isinstance(recognition_error.value.__cause__, LookupError)


def test_recommit_equality_is_contextual_and_cannot_mutate_stored_values() -> None:
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
        value = Decision(semantics, domain=domain(accepts=lambda *, candidate: True))

    base = Trial()
    chosen = refinement.commit(base, refinement.change(base, Trial.value, [1])).instance
    assert refinement.commit(chosen, refinement.change(chosen, Trial.value, [1])).instance is chosen
    assert chosen.value == [1]
    conflict = refinement.commit(chosen, refinement.change(chosen, Trial.value, [2]))
    assert not conflict.accepted and conflict.instance is chosen
    assert chosen.value == [1]
    fail = True
    with pytest.raises(EvaluationError) as raised:
        refinement.commit(chosen, refinement.change(chosen, Trial.value, [1]))
    assert raised.value.owner == "value"
    assert raised.value.role == "commitment equality"
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

    seed = Param(int)
    limit = Derived(limit_value)
    members = [Decision(int, domain=domain(accepts=membership, limit=limit)) for _ in range(64)]
    namespace: dict[str, object] = {"seed": seed, "limit": limit}
    namespace.update({f"choice_{index}": member for index, member in enumerate(members)})
    Family = cast(type[Space], type("IndependentBatch", (Space,), namespace))
    model = compile_space(Family)
    base = model.bind({seed: 100})
    report = refinement.commit(
        base,
        *(
            refinement.change(base, member, index + 1)
            for index, member in reversed(list(enumerate(members)))
        ),
    )
    assert report.accepted
    assert limits == [100]
    assert len(admitted) == len(members)
    assert all(outcome.status == "committed" for outcome in report.outcomes)
    assert report.instance.query(limit) == Available(100)
    assert limits == [100, 100]  # publication starts a separate cache

    limits.clear()
    admitted.clear()
    edits = [refinement.change(base, member, index + 1) for index, member in enumerate(members)]
    edits[-1] = refinement.change(base, members[-1], -1)
    failed = refinement.commit(base, *edits)
    assert not failed.accepted
    assert failed.instance is base
    assert limits == [100]
    assert len(admitted) == len(members)
    assert [outcome.status for outcome in failed.outcomes].count("admissible") == len(members) - 1
    assert all(isinstance(base.query(member), Unresolved) for member in members)


def test_dependent_batch_is_order_independent_without_precommitting_candidates() -> None:
    calls: list[tuple[int, int]] = []

    def membership(*, candidate: int, previous: int) -> bool:
        calls.append((candidate, previous))
        return candidate == previous + 1

    members = [Decision(int, values=(1,))]
    for _ in range(31):
        members.append(Decision(int, domain=domain(accepts=membership, previous=members[-1])))
    Family = cast(
        type[Space],
        type("DependentBatch", (Space,), {f"step_{i}": member for i, member in enumerate(members)}),
    )
    base = Family()
    edits = [refinement.change(base, member, index + 1) for index, member in enumerate(members)]
    report = refinement.commit(base, *reversed(edits))
    assert report.accepted
    assert calls == [(index + 1, index) for index in range(1, len(members))]
    assert report.instance.query(members[-1]) == Available(len(members))
    calls.clear()
    edits[0] = refinement.change(base, members[0], 2)
    failed = refinement.commit(base, *reversed(edits))
    assert not failed.accepted
    assert failed.instance is base
    assert calls == []  # the refused first candidate never became a dependency value
    primary = failed.outcomes[-1].result
    assert isinstance(primary, Rejected)
    assert primary.findings and all(finding.owner == "step_0" for finding in primary.findings)
    # Dependents preserve the prerequisite cause; they have not run membership.
    for item in failed.outcomes[:-1]:
        assert isinstance(item.result, Rejected)
        assert item.result.findings == primary.findings


def test_selector_and_nested_edit_share_atomic_order_and_inactive_edits_refuse() -> None:
    class Child(Space):
        value = Decision(int, values=(1, 2))

    class Root(Space):
        implementation = SubspaceChoice({"a": Subspace(Child), "b": Subspace(Child)})

    base = Root()
    selector = choices(base)[0].selector
    assert selector is not None
    child = base.implementation.alternative("a")
    report = refinement.commit(
        base,
        refinement.change(child, Child.value, 2),
        refinement.change(base, selector, "a"),
    )
    assert report.accepted
    assert report.instance.implementation.alternative("a").query(Child.value) == Available(2)
    inactive = report.instance.implementation.alternative("b")
    failed = refinement.commit(report.instance, refinement.change(inactive, Child.value, 1))
    assert not failed.accepted
    assert isinstance(failed.outcomes[0].result, Inapplicable)
    assert failed.instance is report.instance
    with pytest.raises(ConfigurationError):
        report.instance.with_choices(report.instance.field(selector).change("b"))


def test_enumerated_candidates_still_pass_membership_before_commitment() -> None:
    seen: list[int] = []

    def membership(*, candidate: int) -> bool:
        seen.append(candidate)
        return candidate > 0

    class Trial(Space):
        value = Decision(int, domain=domain(accepts=membership, candidates=lambda: (-1, 1)))

    base = Trial()
    assert base.field(Trial.value).candidates() == Available((-1, 1))
    assert seen == []
    report = refinement.commit(base, refinement.change(base, Trial.value, -1))
    assert not report.accepted
    assert seen == [-1]
    assert isinstance(base.query(Trial.value), Unresolved)


def test_programmer_failure_after_provisional_admission_never_publishes_a_successor() -> None:
    def broken(*, candidate: int) -> bool:
        raise RuntimeError("membership failed")

    class Trial(Space):
        first = Decision(int, values=(1,))
        second = Decision(int, domain=domain(accepts=broken))

    base = Trial()
    with pytest.raises(EvaluationError):
        refinement.commit(
            base,
            refinement.change(base, Trial.first, 1),
            refinement.change(base, Trial.second, 1),
        )
    assert isinstance(base.query(Trial.first), Unresolved)
    assert isinstance(base.query(Trial.second), Unresolved)
    assert refinement.commit(base).instance is base


def test_selector_report_uses_authored_owner() -> None:
    class Root(Space):
        implementation = SubspaceChoice({"a": Subspace(Space), "b": Subspace(Space)})

    point = Root()
    selector = choices(point)[0].selector
    assert selector is not None
    report = refinement.commit(point, refinement.change(point, selector, "a"))
    assert report.accepted
    assert report.outcomes[0].owner == "implementation"
