# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a policy reads of an open Decision beyond its possible values: whether its domain
orders them, the Space class that declares it, and the Decisions known by membership
only (their values listed as ``None``), whose values a policy proposes."""

from __future__ import annotations

from support.refusals import refused

from core.space._possible_support import listed, open_now
from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Space,
    design_space,
    divisors_of,
    domain,
    inspection,
)
from finn.core.space.domains import finite


class Buffer(Space):
    depth: int = Decision(domain=domain(accepts=lambda *, candidate: candidate >= 2, ordered=True))
    style: str = Decision(domain=("auto", "block"))


class Core(Space):
    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))
    stages: int = Decision(domain=(0, 1, 2), ordered=True)
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
    with refused(
        "ordered-invalid", error=DefinitionError, message="ordered= is a bool for finite iterables"
    ):
        Decision(domain=divisors_of(4), ordered=True)  # type: ignore[call-overload]


def test_a_decision_known_by_membership_only_is_open_until_committed() -> None:
    point = design_space(Core(width=4))
    assert "buffer.depth" in open_now(point)
    info = {item.key: item for item in inspection.decisions(point)}["buffer.depth"]
    assert info.ordered
    # It lists no values: a policy proposes them.
    assert listed(point, Core.buffer.depth) is None
    committed = point.with_choices({Core.buffer.depth: 16})
    assert "buffer.depth" not in open_now(committed)
