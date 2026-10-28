# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic request normalization and public domain behavior."""

from typing import cast

import pytest

from finn.core.space import Decision, Space, design_space
from finn.core.space.domains import Domain
from finn.core.space.errors import RequestError
from finn.core.space.results import Unresolved


def test_malformed_last_candidate_prevents_first_membership_callback() -> None:
    calls: list[int] = []

    def membership(*, candidate: int) -> bool:
        calls.append(candidate)
        return True

    class Trial(Space):
        first: int = Decision(domain=Domain((), membership))
        second: int = Decision(domain=(1, 2))

    base = design_space(Trial())
    with pytest.raises(RequestError):
        base.try_with_choices(
            base.field(Trial.first).change(1), base.field(Trial.second).change(cast(int, "bad"))
        )
    assert calls == []
    assert isinstance(base.query(Trial.first), Unresolved)
