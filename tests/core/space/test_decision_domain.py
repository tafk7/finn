# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One domain spelling, with unchanged finite and fact-dependent choice semantics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    RequestError,
    Space,
    design_space,
    inspection,
)
from finn.core.space.domains import divisors_of, domain, finite
from finn.core.space.results import Available, Unresolved


@pytest.mark.parametrize("values", [(3, 1, 2), [3, 1, 2], range(3, 0, -1), frozenset((3, 1, 2))])
@pytest.mark.parametrize("ordered", [False, True])
def test_finite_domain_normalizes_without_changing_cases_or_order(
    values: Iterable[int], ordered: bool
) -> None:
    expected = tuple(values)

    class Choice(Space):
        number: int = Decision(domain=values, ordered=ordered)

    point = design_space(Choice())
    (info,) = inspection.decisions(point)
    assert info.key == "number" and info.ordered is ordered
    (viable,) = inspection.viable(point, [Choice.number])
    assert viable.cases == expected
    assert isinstance(point.query(Choice.number), Unresolved)
    for number in expected:
        assert point.with_choices(number=number).query(Choice.number) == Available(number)
    assert not point.try_with_choices(number=9).accepted
    with pytest.raises(RequestError, match="nominal type int"):
        point.try_with_choices(number=True)


def test_finite_iterable_is_snapshotted_and_empty_means_no_choices() -> None:
    values = [3, 1]

    class Choice(Space):
        number: int = Decision(domain=iter(values))

    values.append(2)
    point = design_space(Choice())
    (viable,) = inspection.viable(point, [Choice.number])
    assert viable.cases == (3, 1)
    assert not point.try_with_choices(number=2).accepted

    class Empty(Space):
        number: int = Decision(domain=())

    empty = design_space(Empty())
    (viable,) = inspection.viable(empty, [Empty.number])
    assert viable.cases == ()
    assert not empty.try_with_choices(number=1).accepted


def test_tuple_choice_is_one_member_of_the_outer_value_space() -> None:
    class Choice(Space):
        pair: tuple[int, int] = Decision(domain=((1, 2), (3, 4)), ordered=True)

    point = design_space(Choice())
    (viable,) = inspection.viable(point, [Choice.pair])
    assert viable.cases == ((1, 2), (3, 4))
    assert point.with_choices(pair=(3, 4)).pair == (3, 4)
    assert not point.try_with_choices(pair=(1, 4)).accepted


def test_domain_object_keeps_its_own_order_and_lazy_callbacks() -> None:
    calls: list[str] = []

    def accepts(*, candidate: int) -> bool:
        calls.append("membership")
        return candidate > 0

    allowed = domain(accepts=accepts, ordered=True)

    class Choice(Space):
        number: int = Decision(domain=allowed)

    assert calls == []
    assert cast(Decision[int], Choice.number).domain is allowed
    point = design_space(Choice())
    assert calls == []
    (info,) = inspection.decisions(point)
    assert info.ordered
    assert point.with_choices(number=7).number == 7
    assert calls and set(calls) == {"membership"}


def test_fact_dependent_domain_keeps_keys_and_recomputes_allowed_values() -> None:
    class Choice(Space):
        extent: int = Param()
        factor: int = Decision(domain=divisors_of(extent))

    point = design_space(Choice(extent=12))
    (viable,) = inspection.viable(point, [Choice.factor])
    assert viable.cases == (1, 2, 3, 4, 6, 12)
    changed = point.with_facts({Choice.extent: 10})
    (viable,) = inspection.viable(changed, [Choice.factor])
    assert viable.cases == (1, 2, 5, 10)
    assert changed.with_choices(factor=5).factor == 5
    assert {item.key for item in inspection.decisions(changed)} == {"factor"}


def test_invalid_domains_and_order_overrides_are_rejected() -> None:
    invalid: Any = 3
    with refused("domain-argument-invalid", error=DefinitionError, message="finite iterable"):
        Decision(domain=invalid)
    for ordered in (False, True):
        with refused(
            "ordered-invalid", error=DefinitionError, message="Domain states its own order"
        ):
            Decision(domain=finite((1, 2), ordered=ordered), ordered=ordered)  # type: ignore[call-overload]
