# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic request normalization and public domain behavior."""

from typing import cast

import pytest

from finn.core.space import Decision, Space, ValueSemantics
from finn.core.space.domains import Domain
from finn.core.space.errors import RequestError
from finn.core.space.results import Available, Unresolved


def test_all_candidate_snapshots_precede_membership_callbacks() -> None:
    payload = [1]
    seen: list[tuple[int, ...]] = []

    def first_membership(*, candidate: int) -> bool:
        # Deliberately perturb the caller's second payload to probe the
        # boundary: the second candidate was already snapshotted at this point.
        payload.append(2)
        return candidate > 0

    def second_membership(*, candidate: list[int]) -> bool:
        seen.append(tuple(candidate))
        candidate.append(99)
        return True

    class Trial(Space):
        first = Decision(int, domain=Domain((), first_membership))
        second: Decision[list[int]] = Decision(list, domain=Domain((), second_membership))

    base = Trial()
    result = base.try_with_choices(
        base.field(Trial.first).change(1), base.field(Trial.second).change(payload)
    )
    assert result.accepted
    assert seen == [(1,)]
    assert result.instance.second == [1]
    assert payload == [1, 2]
    assert isinstance(base.query(Trial.second), Unresolved)


def test_malformed_last_candidate_prevents_first_membership_callback() -> None:
    calls: list[int] = []

    def membership(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Trial(Space):
        first = Decision(int, domain=Domain((), membership))
        second = Decision(int, values=(1, 2))

    base = Trial()
    with pytest.raises(RequestError):
        base.try_with_choices(
            base.field(Trial.first).change(1), base.field(Trial.second).change(cast(int, "bad"))
        )
    assert calls == []
    assert isinstance(base.query(Trial.first), Unresolved)


def test_unhashable_finite_domain_uses_declared_equality_and_detaches_reads() -> None:
    semantics: ValueSemantics[list[int]] = ValueSemantics(
        type_token=list,
        name="integer bag",
        recognizes=lambda value: isinstance(value, list) and all(type(x) is int for x in value),
        equal=lambda left, right: sorted(left) == sorted(right),
        snapshot=lambda value: list(value),
    )
    allowed = [1, 2]

    class Bags(Space):
        bag = Decision(semantics, values=(allowed,))

    allowed.append(3)
    base = Bags()
    candidates = base.field(Bags.bag).candidates()
    assert candidates == Available(([1, 2],))
    assert isinstance(candidates, Available)
    candidates.value[0].append(4)
    assert base.field(Bags.bag).candidates() == Available(([1, 2],))
    chosen = base.with_choices(bag=[2, 1])
    assert chosen.bag == [2, 1]
    assert chosen.with_choices(bag=[1, 2]) is chosen
    failed = base.try_with_choices(base.field(Bags.bag).change([1, 2, 3]))
    assert not failed.accepted
    assert failed.instance is base
