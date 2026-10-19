# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reads of a configuration's state on its world: answers, presence of a value input,
validity and its pieces, and the evidence of a query from the entries' traces."""

from __future__ import annotations

from typing import cast

import pytest

from core.space.test_assign import change, codes, start
from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueUnavailableError,
    constraint,
    derived,
    divisors_of,
    reject,
    supplied,
)
from finn.core.space._state import State, answer, source_applies, value
from finn.core.space.inspection import _evidence, _validity_of


class Leaf(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))

    @constraint
    def even(self) -> bool:
        return self.lanes % 2 == 0


class Wide(Space):
    pass


class Root(Space):
    extent: int = Param()
    mode: Leaf | Wide = Decision({"leaf": Leaf(extent=extent), "wide": Wide})
    width: int = Decision(values=(8, 16))

    @constraint
    def fits(self) -> bool:
        return self.width <= self.extent


def test_validity_is_valid_refused_or_open_naming_its_pieces() -> None:
    base = start(Root(extent=8))
    validity = _validity_of(base, 0)
    # ``fits`` waits on ``width``; the leaf's ``even`` waits under the open ``mode``.
    assert validity.status == "open"
    assert [piece.owner for piece in validity.pieces] == ["fits", "mode.leaf.even"]
    point = cast(State, change(base, {"width": 16, "mode": "leaf", "mode.leaf.lanes": 1}).state)
    refused = _validity_of(point, 0)
    assert refused.status == "refused"
    assert [piece.owner for piece in refused.pieces] == ["fits", "mode.leaf.even"]
    valid = cast(State, change(point, {"width": 8, "mode": "wide"}).state)
    assert _validity_of(valid, 0).status == "valid"


class Empty(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))


def test_a_present_choice_whose_domain_lists_no_value_is_a_false_piece() -> None:
    validity = _validity_of(start(Empty(extent=0)), 0)
    assert validity.status == "refused"
    (piece,) = validity.pieces
    assert piece.owner == "lanes" and codes(piece.result) == [("domain-empty", "lanes")]


class Refusing(Space):
    w: int = Param()

    @derived
    def odd(self) -> bool | Rejected:
        return reject("bad-width", "an odd width") if self.w % 2 else False

    child: Leaf = Leaf(extent=w, when=odd)


def test_a_guard_refused_through_its_input_is_one_guard_piece() -> None:
    validity = _validity_of(start(Refusing(w=7)), 0)
    assert validity.status == "refused"
    (piece,) = validity.pieces
    assert piece.owner == "child"
    assert codes(piece.result) == [("guard-refused", "child"), ("bad-width", "odd")]


class Fed(Space):
    size: int = Param(required=False)
    depth: int = Param()

    @derived
    def doubled(self) -> int:
        return 2 * self.depth

    has_size = supplied(size)
    gated: int = Decision(values=(1, 2), when=has_size)


def test_answers_values_and_whether_an_input_is_supplied() -> None:
    point = start(Fed(depth=3))
    keys = point.linked.keys
    assert answer(point, keys["doubled"]) == Available(6)
    assert value(point, keys["doubled"]) == 6
    with pytest.raises(ValueUnavailableError):
        value(point, keys["gated"])
    assert source_applies(point, keys["depth"]) is True
    assert isinstance(answer(point, keys["gated"]), Inapplicable)
    open_ = start(Fed(depth=3, size=4))
    assert isinstance(answer(open_, keys["gated"]), Unresolved)


def test_the_evidence_of_a_query_is_what_its_entries_read() -> None:
    point = start(Fed(depth=3))
    keys = point.linked.keys
    evidence = _evidence(point, keys["doubled"])
    assert evidence.result == Available(6)
    by_key = {item.declaration.key: item for item in evidence.nodes}
    assert set(by_key) == {"doubled", "depth"}
    assert [handle._resolve(point.linked) for handle in by_key["doubled"].dependencies] == [
        keys["depth"]
    ]
    assert by_key["depth"].input_presence == "supplied"
