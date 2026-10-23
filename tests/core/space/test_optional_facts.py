# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""An optional fact, ``x: T | None = Param(default=None)``: the annotation's union with
``None`` gives ``T``'s check made optional (``semantics.optional``), which sees ``None``
first, by identity. Left out and given ``None`` are one fact, so one interned facts layer
and one design space (``K(x=None) == K()``); a value given is a ``T``, compared by type and
``==``, and keyed as ``T``'s values are (``None`` as the plain ``None``)."""

from __future__ import annotations

from typing import Any, Optional, cast

import pytest

from finn.core.space import Available, Param, Rejected, Space, derived, design_space
from finn.core.space._running.cache import key as cache_key
from finn.core.space.errors import DefinitionError, RequestError
from finn.core.space.occurrence import held
from finn.core.space.semantics import OptionalSemantics, ValueSemantics


def _semantics(space_type: type[Space], name: str) -> ValueSemantics[Any]:
    semantics = cast("Param[object]", vars(space_type)[name]).semantics
    assert semantics is not None
    return semantics


class Lanes(Space):
    width: int | None = Param(default=None)

    @derived
    def lanes(self) -> int:
        return 2 if self.width is None else self.width // 4


def test_an_optional_fact_left_out_given_none_and_given_a_value() -> None:
    left_out = design_space(Lanes())
    given_none = design_space(Lanes(width=None))
    given = design_space(Lanes(width=12))
    assert left_out.width is None and left_out.lanes == 2
    assert given_none.width is None and given_none.lanes == 2
    assert given.width == 12 and given.lanes == 3


def test_left_out_and_given_none_are_one_design_space() -> None:
    left_out, given_none = held(design_space(Lanes())), held(design_space(Lanes(width=None)))
    assert left_out.ds.layer is given_none.ds.layer
    assert left_out.ds == given_none.ds
    given = held(design_space(Lanes(width=12)))
    assert given.ds.layer is not left_out.ds.layer
    assert given.ds != left_out.ds


def test_the_optional_check_sees_none_first() -> None:
    semantics = _semantics(Lanes, "width")
    assert isinstance(semantics, OptionalSemantics)
    assert semantics.inner.type_token is int
    assert semantics.type_token is int and semantics.name == "int | None"
    assert semantics.accepts(None) and semantics.accepts(3) and not semantics.accepts("3")
    assert not semantics.accepts(True)


def test_optional_spelled_with_typing_is_the_same() -> None:
    class Spelled(Space):
        width: Optional[int] = Param(default=None)  # noqa: UP045 - the spelling under test

    semantics = _semantics(Spelled, "width")
    assert isinstance(semantics, OptionalSemantics) and semantics.type_token is int
    assert design_space(Spelled()).width is None


def test_a_value_outside_the_optional_type_is_refused() -> None:
    with pytest.raises((DefinitionError, RequestError), match=r"int \| None"):
        design_space(Lanes(width="wide"))  # type: ignore[arg-type]


def test_a_union_of_several_classes_takes_a_value_of_any_of_them() -> None:
    """Each member a value type: the union's check takes a value any member's takes, and
    ``None`` where it is a member; anything else is refused at the boundary."""

    class Several(Space):
        width: int | str | None = Param(default=None)

        @derived
        def kind(self) -> str:
            return type(self.width).__name__

    assert design_space(Several()).kind == "NoneType"
    assert design_space(Several(width=3)).kind == "int"
    assert design_space(Several(width="wide")).kind == "str"
    with pytest.raises(DefinitionError, match=r"int \| str \| None"):
        Several(width=1.5)  # type: ignore[arg-type]
    with pytest.raises(DefinitionError, match="no value type"):

        class Untyped(Space):
            width: Any = Param()


def test_the_cache_keys_an_optional_value_as_its_type_keys_it() -> None:
    assert cache_key(None) == (type(None), None)
    assert cache_key(3) == (int, 3)
    assert cache_key(None) != cache_key(0)


# devi's probe (E2's review): an optional fact left out, bound to a child's required formal
# of the same type, with no guard. The binding is a definition (the two types are
# compatible); the formal recognizes the value it is bound to by its own type, so ``None``
# never flows in: the formal is refused at itself (``formal-value``), and a derived value
# reading it is refused through it, where before both raised.


class Needs(Space):
    rate: int = Param()

    @derived
    def doubled(self) -> int:
        return self.rate * 2


class Office(Space):
    rate: int | None = Param(default=None)
    child = Needs(rate=rate)  # type: ignore[arg-type]  # the engine's check, not mypy's


def _codes(result: object) -> set[tuple[str, str]]:
    assert isinstance(result, Rejected)
    return {(finding.code, finding.owner) for finding in result.findings}


def test_an_optional_fact_left_out_is_refused_at_the_required_formal_it_is_bound_to() -> None:
    left_out = design_space(Office())
    rate = left_out.child.query(Needs.rate)
    assert _codes(rate) == {("formal-value", "child.rate")}
    assert isinstance(rate, Rejected) and rate.findings[0].details == (("value type", "NoneType"),)
    assert _codes(left_out.child.query(Needs.doubled)) == {("formal-value", "child.rate")}


def test_an_optional_fact_given_reads_through_the_required_formal_it_is_bound_to() -> None:
    given = design_space(Office(rate=5))
    assert given.child.query(Needs.rate) == Available(5)
    assert given.child.query(Needs.doubled) == Available(10)
