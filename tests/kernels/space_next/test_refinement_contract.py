# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic request normalization and public domain behavior."""

from typing import cast

import pytest

from finn.kernels.space._next import Decision, Space, ValueSemantics
from finn.kernels.space._next.domains import Domain
from finn.kernels.space._next.errors import RequestError
from finn.kernels.space._next.results import Decided, Unresolved


def test_all_candidate_snapshots_precede_membership_callbacks():
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
        second = Decision(list, domain=Domain((), second_membership))

    base = Trial.start()
    result = base.refine(base.edit(Trial.first, 1), base.edit(Trial.second, payload))
    assert result.accepted
    assert seen == [(1,)]
    assert result.point.second == [1]
    assert payload == [1, 2]
    assert isinstance(base.answer(Trial.second), Unresolved)


def test_malformed_last_candidate_prevents_first_membership_callback():
    calls: list[int] = []

    def membership(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Trial(Space):
        first = Decision(int, domain=Domain((), membership))
        second = Decision(int, values=(1, 2))

    base = Trial.start()
    with pytest.raises(RequestError):
        base.refine(base.edit(Trial.first, 1), base.edit(Trial.second, cast(int, "bad")))
    assert calls == []
    assert isinstance(base.answer(Trial.first), Unresolved)


def test_unhashable_finite_domain_uses_declared_equality_and_detaches_reads():
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
    base = Bags.start()
    candidates = base.candidates(Bags.bag)
    assert candidates == Decided(([1, 2],))
    assert isinstance(candidates, Decided)
    candidates.value[0].append(4)
    assert base.candidates(Bags.bag) == Decided(([1, 2],))
    chosen = base.assign(Bags.bag, [2, 1])
    assert chosen.bag == [2, 1]
    assert chosen.assign(Bags.bag, [1, 2]) is chosen
    failed = base.refine(base.edit(Bags.bag, [1, 2, 3]))
    assert not failed.accepted
    assert failed.point is base
