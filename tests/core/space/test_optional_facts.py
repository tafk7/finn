# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""An optional fact, ``x: T | None = Param(default=None)``: the annotation's union with
``None`` gives ``T``'s semantics made optional (``semantics.optional``), which see ``None``
first, by identity, wherever a value meets them: recognition, equality, the snapshot, the
facts layer's interning and the cache's key. Left out and given ``None`` are one fact, so
one interned facts layer and one design space (``K(x=None) == K()``); a value given is read
as ``T``'s semantics read it, an explicit ``semantics=`` included."""

from __future__ import annotations

from typing import Any, Optional, cast

import pytest

from finn.core.space import Available, Param, Rejected, Space, derived, design_space
from finn.core.space._running.cache import key as cache_key
from finn.core.space._running.cache import trusted
from finn.core.space.errors import DefinitionError, RequestError
from finn.core.space.occurrence import held
from finn.core.space.semantics import OptionalSemantics, ValueSemantics, default_semantics


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


def test_the_optional_semantics_see_none_first() -> None:
    semantics = _semantics(Lanes, "width")
    assert isinstance(semantics, OptionalSemantics)
    assert semantics.inner.type_token is int
    assert semantics.type_token is int and semantics.name == "int | None"
    assert semantics.accepts(None) and semantics.accepts(3) and not semantics.accepts("3")
    assert semantics.values_equal(None, None)
    assert not semantics.values_equal(None, 0) and not semantics.values_equal(0, None)
    assert semantics.values_equal(3, 3) and not semantics.values_equal(3, 4)
    assert semantics.freeze(None) is None


def test_optional_spelled_with_typing_is_the_same() -> None:
    class Spelled(Space):
        width: Optional[int] = Param(default=None)  # noqa: UP045 - the spelling under test

    semantics = _semantics(Spelled, "width")
    assert isinstance(semantics, OptionalSemantics) and semantics.type_token is int
    assert design_space(Spelled()).width is None


def test_a_value_outside_the_optional_type_is_refused() -> None:
    with pytest.raises((DefinitionError, RequestError), match=r"int \| None"):
        design_space(Lanes(width="wide"))  # type: ignore[arg-type]


def test_a_union_of_several_classes_still_states_its_semantics() -> None:
    with pytest.raises(DefinitionError, match="semantics="):

        class Several(Space):
            width: int | str | None = Param(default=None)


def test_the_cache_keys_an_optional_value_as_its_type_keys_it() -> None:
    semantics = _semantics(Lanes, "width")
    assert trusted(semantics)
    assert cache_key(None, semantics) == cache_key(None, default_semantics(type(None)))
    assert cache_key(3, semantics) == cache_key(3, default_semantics(int))
    assert cache_key(None, semantics) != cache_key(0, semantics)


# -- an optional fact with explicit semantics: T's, made optional ------------------------------
#
# ``EVEN`` holds two ints equal by parity and snapshots each to its parity: semantics of the
# fact's own (the cache trusts none of its values), which never see None.

SEEN: list[object] = []


def _parity(value: int) -> int:
    SEEN.append(value)
    return value % 2


EVEN: ValueSemantics[int] = ValueSemantics(
    int,
    "parity",
    lambda value: type(value) is int,
    lambda left, right: _parity(left) == _parity(right),
    _parity,
)


class Parity(Space):
    bit: int | None = Param(default=None, semantics=EVEN)

    @derived
    def odd(self) -> bool:
        return self.bit is not None and self.bit == 1


def test_an_optional_fact_with_explicit_semantics_left_out_given_none_and_given() -> None:
    SEEN.clear()
    left_out, given_none = design_space(Parity()), design_space(Parity(bit=None))
    assert held(left_out).ds.layer is held(given_none).ds.layer
    assert left_out.bit is None and given_none.bit is None
    assert not left_out.odd and not given_none.odd
    assert None not in SEEN  # the fact's semantics never saw None
    given = design_space(Parity(bit=7))
    assert given.bit == 1 and given.odd  # the explicit snapshot
    assert held(given).ds.layer is not held(left_out).ds.layer
    assert held(design_space(Parity(bit=9))).ds.layer is held(given).ds.layer
    assert None not in SEEN


def test_explicit_optional_semantics_are_trusted_as_their_own_are() -> None:
    semantics = _semantics(Parity, "bit")
    assert isinstance(semantics, OptionalSemantics) and semantics.inner is EVEN
    assert not trusted(semantics) and cache_key(None, semantics) is None


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
