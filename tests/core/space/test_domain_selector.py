# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A domain that reads a selection, whose cases read facts only, reads its selector: the
choice that picks the case. The domain rule (a domain reads facts only) refuses it,
``domain-reads-choice``, naming the selector; the composed rule as it compiles the class
(``_compose``), and the line's rule over every address (``_domains_support``) alike, through
a class-body selection and through a method reading one.

A hand case: it runs alike before native addressing (the line's rule then over ``expand``'s
table) and after it."""

from __future__ import annotations

import pytest
from support.refusals import refused

from core.space._domains_support import line_refusals
from core.space._line_reads import line_scopes
from finn.core.space import Decision, Param, Select, Space, derived, divisors_of
from finn.core.space.compiler import compile_model
from finn.core.space.errors import DefinitionError


class Wide(Space):
    w: int = Param()


class Narrow(Space):
    w: int = Param()


class Selection(Space):
    """``limit`` is the selection of the selected case's ``w``; each case's is the fact
    ``w``."""

    w: int = Param()
    pick: Wide | Narrow = Select(wide=Wide(w=w), narrow=Narrow(w=w))
    limit = pick.w
    p: int = Decision(domain=divisors_of(limit))


class ByMethod(Space):
    """``limit`` reads the selected case's ``w`` in a method: it reads the selector and each
    case's ``w``, the fact ``w``."""

    w: int = Param()
    pick: Wide | Narrow = Select(wide=Wide(w=w), narrow=Narrow(w=w))

    @derived
    def limit(self) -> int:
        return self.pick.w

    p: int = Decision(domain=divisors_of(limit))


@pytest.mark.parametrize("space_type", [Selection, ByMethod], ids=lambda c: c.__name__)
def test_a_domain_reading_a_selection_reads_its_selector(space_type: type[Space]) -> None:
    with refused(
        "domain-reads-choice",
        error=DefinitionError,
        owner="p",
        message=r"reads the choice pick \(",
    ):
        compile_model(space_type)
    line = line_refusals(line_scopes(space_type))
    assert set(line) == {"p"}
    assert "reads the choice pick (" in str(line["p"])
