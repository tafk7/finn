# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The optional harness falsifies sampled violations without generating choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.kernels.space import (
    QueryResult,
    Const,
    Available,
    Decision,
    Param,
    Readiness,
    Space,
    Subspace,
    Unresolved,
    ValueSemantics,
    View,
    constraint,
    derived,
    divisors_of,
    full_result,
    refinement,
)
from finn.kernels.space.conformance import MonotonicityHarness, Sample
from finn.kernels.space.errors import EvaluationError, RequestError


def test_conforming_values_states_views_constraints_and_candidates() -> None:
    class Family(Space):
        source = Param(int, required=False)
        extent = Decision(int, values=(4, 8))
        factor = Decision(int, domain=divisors_of(extent))
        fixed = Const([1, 2])

        @constraint
        def supported() -> bool:
            return True

        ready = Readiness(factor)
        physical = View(fixed, constraints=(supported,), requires=(factor,))

    base = Family()
    report = MonotonicityHarness().verify(
        base,
        [
            refinement.change(base, Family.extent, 4),
            (
                refinement.change(base, Family.factor, 2),
                refinement.change(base, Family.extent, 8),
            ),
        ],
    )
    assert report.conformant is True
    assert report.checked_successors == 2
    assert report.violations == ()
    assert [outcome.status for outcome in report.outcomes] == ["checked", "checked"]
    assert isinstance(base.query(Family.extent), Unresolved)


def test_full_result_callback_which_changes_a_settled_value_is_reported() -> None:
    class Family(Space):
        choice = Decision(int, values=(1, 2))

        @derived(observed=full_result(choice))
        def optimistic(*, observed: QueryResult[int]) -> int:
            return observed.value if isinstance(observed, Available) else 0

        @constraint(observed=full_result(choice))
        def unstable_admission(*, observed: QueryResult[int]) -> bool:
            return isinstance(observed, Unresolved)

        physical = View(optimistic, constraints=(unstable_admission,))

    base = Family()
    result = MonotonicityHarness().verify(base, [refinement.change(base, Family.choice, 1)])
    assert result.conformant is False
    assert {(item.key, item.category) for item in result.violations} == {
        ("optimistic", "value"),
        ("unstable_admission", "constraint"),
        ("physical", "view"),
        ("physical", "view_output"),
    }
    assert result.violations == tuple(
        sorted(result.violations, key=lambda item: (item.sample, item.key, item.category))
    )


@dataclass
class Bag:
    values: list[int]


BAG = ValueSemantics(
    Bag,
    "bag",
    lambda value: type(value) is Bag,
    lambda left, right: sorted(left.values) == sorted(right.values),
    lambda value: Bag(list(value.values)),
)


def test_unhashable_values_and_candidates_use_declared_equality() -> None:
    calls = 0

    class Family(Space):
        candidate = Decision(BAG, values=(Bag([1, 2]), Bag([3])))

        @derived(semantics=BAG)
        def reordered() -> Bag:
            nonlocal calls
            calls += 1
            return Bag([1, 2] if calls % 2 else [2, 1])

        physical = View(reordered)

    base = Family()
    result = MonotonicityHarness().verify(
        base, [refinement.change(base, Family.candidate, Bag([2, 1]))]
    )
    assert result.conformant is True
    assert result.violations == ()
    assert calls == 2


def test_skipped_and_noop_samples_are_distinct_and_never_claim_success() -> None:
    class Family(Space):
        choice = Decision(int, values=(1, 2))

    point = Family().with_choices(choice=1)
    different_base = Family()
    samples: list[Sample] = [
        refinement.change(point, Family.choice, 1),
        (),
        refinement.change(point, Family.choice, 2),
        refinement.change(different_base, Family.choice, 1),
        cast(Sample, object()),
    ]
    result = MonotonicityHarness().verify(point, samples)
    assert result.checked_successors == 0 and result.conformant is None
    assert [outcome.status for outcome in result.outcomes] == [
        "noop",
        "noop",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert len(result.noops) == 2 and len(result.skipped) == 3
    assert MonotonicityHarness().verify(point, ()).conformant is None


def test_candidate_refusals_are_skipped_without_publication() -> None:
    class Family(Space):
        a = Decision(int, values=(1,))
        b = Decision(int, values=(2,))

    base = Family()
    report = MonotonicityHarness().verify(
        base,
        [
            (
                refinement.change(base, Family.a, 1),
                refinement.change(base, Family.b, 3),
            ),
        ],
    )
    assert report.conformant is None and report.skipped[0].reason == "refinement refused"
    assert isinstance(base.query(Family.a), Unresolved)


def test_programmer_failures_remain_contextual_exceptions() -> None:
    class Family(Space):
        choice = Decision(int, values=(1,))

        @derived(observed=full_result(choice))
        def failure(*, observed: QueryResult[int]) -> int:
            if isinstance(observed, Available):
                raise RuntimeError("callback failed")
            return 0

    base = Family()
    with pytest.raises(EvaluationError) as raised:
        MonotonicityHarness().verify(base, [refinement.change(base, Family.choice, 1)])
    assert raised.value.owner == "failure"
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_child_base_is_rejected_without_guessing_root_sample_scope() -> None:
    class Child(Space):
        choice = Decision(int, values=(1,))

    class Family(Space):
        child = Subspace(Child)

    point = Family().child
    with pytest.raises(RequestError, match="root configuration"):
        MonotonicityHarness().verify(point, [refinement.change(point, Child.choice, 1)])
