# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a design space answers, read by the running type: answers, presence of a value
input, validity and its pieces, and the evidence of a query from the answers' traces."""

from __future__ import annotations

import pytest

from core.space._design_support import built, change, codes, published, start, validity
from finn.core.space import (
    Available,
    Decision,
    Domain,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    ValueUnavailableError,
    constraint,
    derived,
    divisors_of,
    domain,
    reject,
)
from finn.core.space._running import surface
from finn.core.space._running.evaluate import Session
from finn.core.space._running.surface import At
from finn.core.space._running.value import DesignSpace
from finn.core.space._scopes import Address
from finn.core.space.references import ValueHandle
from finn.core.space.results import QueryResult


def _handle(ds: DesignSpace, index: Address) -> ValueHandle[object]:
    return ValueHandle[object](ds.model, index)


def answer(ds: DesignSpace, index: Address) -> QueryResult[object]:
    """``index``'s public answer, detached."""
    return surface.query(At(ds), _handle(ds, index), Session())


def value(ds: DesignSpace, index: Address) -> object:
    return surface.value(At(ds), _handle(ds, index), Session())


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
    mode: Leaf | Wide = Select(leaf=Leaf(extent=extent), wide=Wide)
    width: int = Decision(domain=(8, 16))

    @constraint
    def fits(self) -> bool:
        return self.width <= self.extent


def test_validity_is_valid_refused_or_open_naming_its_pieces() -> None:
    base = start(Root(extent=8))
    found = validity(base)
    # ``fits`` waits on ``width``; the leaf's ``even`` waits under the open ``mode``.
    assert found.status == "open"
    assert [piece.owner for piece in found.pieces] == ["fits", "mode.leaf.even"]
    choices = {"width": 16, "mode": "leaf", "mode.leaf.lanes": 1}
    report = change(base, choices)  # breaks both rules on the values it names
    assert [(rule.rule, rule.standing) for rule in report.broken] == [
        ("fits", ()),
        ("mode.leaf.even", ()),
    ]
    # Built directly, the same choices read refused, naming both.
    point = built(Root(extent=8), choices)
    refused = validity(point)
    assert refused.status == "refused"
    assert [piece.owner for piece in refused.pieces] == ["fits", "mode.leaf.even"]
    valid = published(change(point, {"width": 8, "mode": "wide"}))
    assert validity(valid).status == "valid"


class Empty(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))


def test_a_present_choice_whose_domain_lists_no_value_is_a_false_piece() -> None:
    found = validity(start(Empty(extent=0)))
    assert found.status == "refused"
    (piece,) = found.pieces
    assert piece.owner == "lanes" and codes(piece.result) == [("domain-empty", "lanes")]


def at_most(bound: object) -> Domain[int]:
    """A domain known by membership alone: nothing is listed."""
    return domain(accepts=lambda *, candidate, bound: 0 <= candidate <= bound, bound=bound)


class Bounded(Space):
    f: int = Param()
    c: int = Decision(domain=at_most(f))


class Bare(Space):
    pass


class UnderOpen(Space):
    f: int = Param()
    s: Bounded | Bare = Select(a=Bounded(f=f), b=Bare())


class RefusingInput(Space):
    f: int = Param()

    @derived
    def d(self) -> int | Rejected:
        return reject("too-big", "over two") if self.f > 2 else self.f

    c: int = Decision(domain=at_most(d))


class AbsentInput(Space):
    wide: bool = Param()
    f: int = Param()

    @derived
    def d(self) -> int | Inapplicable:
        if not self.wide:
            return Inapplicable()
        return self.f

    c: int = Decision(domain=at_most(d))


@pytest.mark.parametrize(
    "declaration",
    [Bounded(f=1), UnderOpen(f=1), RefusingInput(f=5), AbsentInput(wide=False, f=1)],
    ids=["input-known", "under-an-open-selector", "input-refusing", "input-absent"],
)
def test_a_domain_that_lists_no_values_is_never_a_piece(declaration: Space) -> None:
    """Membership alone lists nothing: whatever its inputs answer (a value, a refusal on
    the facts, an input that does not apply), the choice adds no piece, and an open choice
    with no rule reading it leaves the configuration valid."""
    found = validity(start(declaration))
    assert (found.status, found.pieces) == ("valid", ())


class Refusing(Space):
    w: int = Param()

    @derived
    def odd(self) -> bool | Rejected:
        return reject("bad-width", "an odd width") if self.w % 2 else False

    child: Leaf = Leaf(extent=w, when=odd)


def test_a_guard_refused_through_its_input_is_one_guard_piece() -> None:
    found = validity(start(Refusing(w=7)))
    assert found.status == "refused"
    (piece,) = found.pieces
    assert piece.owner == "child"
    assert codes(piece.result) == [("guard-refused", "child"), ("bad-width", "odd")]


class Fed(Space):
    size: int | None = Param(default=None)
    depth: int = Param()

    @derived
    def doubled(self) -> int:
        return 2 * self.depth

    @derived
    def has_size(self) -> bool:
        return self.size is not None

    gated: int = Decision(domain=(1, 2), when=has_size)


def test_answers_and_values() -> None:
    point = start(Fed(depth=3))
    keys = point.model.scopes.keys
    assert answer(point, keys["doubled"]) == Available(6)
    assert value(point, keys["doubled"]) == 6
    assert answer(point, keys["size"]) == Available(None)
    with pytest.raises(ValueUnavailableError):
        value(point, keys["gated"])
    assert isinstance(answer(point, keys["gated"]), Inapplicable)
    open_ = start(Fed(depth=3, size=4))
    assert isinstance(answer(open_, keys["gated"]), Unresolved)


def test_the_evidence_of_a_query_is_what_its_answers_read() -> None:
    point = start(Fed(depth=3))
    keys = point.model.scopes.keys
    evidence = surface.explain(At(point), _handle(point, keys["doubled"]), Session())
    assert evidence.result == Available(6)
    by_key = {item.declaration.key: item for item in evidence.nodes}
    assert set(by_key) == {"doubled", "depth"}
    scopes = point.model.scopes
    assert [handle._resolve(scopes) for handle in by_key["doubled"].dependencies] == [keys["depth"]]
