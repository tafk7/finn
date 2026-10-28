# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Scope-presence diagnostics use declaration positions, never allocation order."""

import pytest

from finn.core.space import Available, Decision, Param, Space, Unresolved, inspection
from finn.core.space.compiler import compiled, model_of
from finn.core.space.occurrence import opened


class Leaf(Space):
    value: int = Param(default=8)


class OptionalChildren(Space):
    enabled: bool = Decision(domain=(False, True))
    first = Leaf(when=enabled)
    second = Leaf(when=enabled)
    third = Leaf(when=enabled)
    fourth = Leaf(when=enabled)


@pytest.mark.parametrize("permuted,eager", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize("reverse_history", (False, True))
def test_presence_order_is_stable_across_allocation_and_query_history(
    permuted: bool, eager: bool, reverse_history: bool
) -> None:
    model = model_of(compiled(OptionalChildren), permuted=permuted, eager=eager)
    point = opened(model, {})
    references = (OptionalChildren.first.value, OptionalChildren.fourth.value)
    for reference in reversed(references) if reverse_history else references:
        assert isinstance(point.query(reference), Unresolved)
    found = inspection.presence(point)
    assert [name for name, _ in found] == ["", "first", "second", "third", "fourth"]
    assert found[0][1] == Available(True)
    assert all(isinstance(result, Unresolved) for _, result in found[1:])
    assert inspection.presence(point) == found
    assert [name for name, result in found if isinstance(result, Unresolved)] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
