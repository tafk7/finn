# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a policy reads of an open Decision beyond its viable cases: whether its domain
orders them, the Space class that declares it, and the Decisions known by membership
only (``inspection.open``), whose values a policy proposes."""

from __future__ import annotations

import pytest

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Space,
    design_space,
    divisors_of,
    domain,
    finite,
    inspection,
)


class Buffer(Space):
    depth: int = Decision(domain=domain(accepts=lambda *, candidate: candidate >= 2, ordered=True))
    style: str = Decision(values=("auto", "block"))


class Core(Space):
    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))
    stages: int = Decision(values=(0, 1, 2), ordered=True)
    buffer = Buffer()


def test_a_domain_states_whether_its_cases_are_ordered() -> None:
    assert divisors_of(6).ordered
    assert not finite((1, 2, 3)).ordered and finite((1, 2, 3), ordered=True).ordered
    infos = {item.key: item for item in inspection.decisions(Core)}
    assert infos["pe"].ordered and infos["stages"].ordered and infos["buffer.depth"].ordered
    assert not infos["buffer.style"].ordered


def test_a_decision_names_the_space_class_that_declares_it() -> None:
    infos = {item.key: item for item in inspection.decisions(Core)}
    assert infos["pe"].space_type is Core and infos["buffer.depth"].space_type is Buffer


def test_ordered_applies_to_listed_values_only() -> None:
    with pytest.raises(DefinitionError, match="ordered= is a bool, for listed values="):
        Decision(domain=divisors_of(4), ordered=True)  # type: ignore[call-overload]


def test_a_decision_known_by_membership_only_is_open_until_committed() -> None:
    point = design_space(Core(width=4))
    assert [item.key for item in inspection.open(point)] == ["buffer.depth"]
    assert inspection.open(point)[0].ordered
    # It is never among the enumerable choices.
    assert "buffer.depth" not in {item.key for item in inspection.viable(point)}
    committed = point.with_choices({Core.buffer.depth: 16})
    assert inspection.open(committed) == ()
