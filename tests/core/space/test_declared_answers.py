# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A rule's declared answer: the non-values its return annotation admits.

A derived value or a constraint answers its value type, and each of ``Rejected``,
``Inapplicable`` and ``Unresolved`` its annotation names (``T | Rejected``; a plain ``T``
names none). An override admitting ``Inapplicable`` or ``Unresolved`` where the answer it
overrides does not is refused when its class is made (``override-widens-answer``);
``Rejected`` is not compared.
"""

from __future__ import annotations

from support.refusals import refused

from finn.core.space import (
    Available,
    Inapplicable,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    reject,
)
from finn.core.space.collection import collect_space
from finn.core.space.errors import DefinitionError


class Base(Space):
    n: int = Param()

    @derived
    def plain(self) -> int:
        return self.n

    @derived
    def refusing(self) -> int | Rejected:
        return reject("odd", "odd") if self.n % 2 else self.n

    @derived
    def any_answer(self) -> QueryResult[int]:
        return Inapplicable() if self.n % 2 else Available(self.n)

    @constraint
    def even(self) -> bool | Rejected:
        return self.n % 2 == 0


def test_an_override_within_its_base_s_answer_is_made() -> None:
    class Narrower(Base):
        @derived
        def refusing(self) -> int:
            return self.n

        @derived
        def any_answer(self) -> int | Rejected:
            return self.n

        @constraint
        def even(self) -> bool:
            return True

    assert design_space(Narrower(n=3)).refusing == 3


def test_a_derived_override_admitting_inapplicable_or_unresolved_its_base_does_not_is_refused() -> (
    None
):
    class Absent(Base):
        @derived
        def refusing(self) -> int | Inapplicable | Rejected:
            return Inapplicable()

    class Waiting(Base):
        @derived
        def plain(self) -> int | Unresolved:
            return self.n

    for wider, member, widened in (
        (Absent, "refusing", "Inapplicable"),
        (Waiting, "plain", "Unresolved"),
    ):
        with refused(
            "override-widens-answer",
            error=DefinitionError,
            message=rf"admits {widened}, which the answer it overrides does not",
            declaration=f"{wider.__qualname__}.{member}",
        ):
            collect_space(wider)


def test_an_override_admitting_a_refusal_its_base_does_not_is_made() -> None:
    """The Kernel's ``schedule``: its base's is ``Schedule | Inapplicable``, each Kernel
    stating one declares ``Schedule | Rejected``."""

    class Refusing(Base):
        @derived
        def plain(self) -> int | Rejected:
            return reject("odd", "odd") if self.n % 2 else self.n

    assert isinstance(design_space(Refusing(n=3)).query(Refusing.plain), Rejected)


def test_a_constraint_override_admitting_inapplicable_is_refused() -> None:
    class Wider(Base):
        @constraint
        def even(self) -> bool | Inapplicable | Rejected:
            return Inapplicable()

    with refused("override-widens-answer", error=DefinitionError):
        collect_space(Wider)
