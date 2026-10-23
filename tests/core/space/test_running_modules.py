# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type's modules by role, each on its own claims:

- ``depth``: the depth guard's limit and exception (a ``BaseException``: no authored
  ``except Exception`` catches it);
- ``purity``: the purity check's switch, read at each ask;
- ``facts``: a design space opened on its facts is its facts layer's with nothing stored,
  interned with the layer; opening checks the facts given (each exposed, every required
  one given, each of its semantics);
- ``requests``: a ``Change`` holds the design space it was made on, and applies to any
  design space equal to it; one made at a child's scope applies at the root.
"""

from __future__ import annotations

from typing import cast

import pytest
from support.space_oracle import toys
from support.space_oracle.public import PublicAnswers

from finn.core.space import RequestError, Space, design_space, inspection
from finn.core.space._running import depth, facts, purity, requests
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.surface import At
from finn.core.space._running.value import FactsLayer
from finn.core.space.occurrence import held

TOYS = toys.TOYS


def _changed(name: str) -> Space:
    """The toy's design space after its change."""
    toy = TOYS[name]
    report = PublicAnswers(toy.make()).change(toy.change, ()).report
    assert report.accepted
    return cast(Space, report.instance)


# -- the depth guard and the purity check --------------------------------------------------------


def test_the_depth_guard_unwinds_past_any_authored_handler() -> None:
    assert depth.DEPTH == 32
    found = depth.Deeper(7)
    assert isinstance(found, BaseException) and not isinstance(found, Exception)
    assert found.node == 7 and found.args == (7,)


def test_the_purity_switch_is_read_at_each_ask() -> None:
    # The tests' switch turns it on for every test here (``support.purity``).
    assert purity._PURITY is True
    first = purity.check_purity(True)
    try:
        for on in (False, True, False):
            purity.check_purity(on)
            assert purity._PURITY is on
            assert evaluation(held(TOYS["house"].make()).ds, Session()).session.verifying is on
    finally:
        purity.check_purity(first)


# -- the facts and a design space opened on them -------------------------------------------------


@pytest.mark.parametrize("name", sorted(TOYS))
def test_a_design_space_opened_on_its_facts_is_the_layers_with_nothing_stored(name: str) -> None:
    current = held(TOYS[name].make()).ds
    model = current.model
    ds = facts.open(model, dict(current.facts))
    assert ds == FactsLayer.of(model, current.facts).bare and not ds.stored
    assert ds.facts == current.facts
    assert ds is facts.open(model, dict(current.facts))  # interned with its layer


def test_opening_a_design_space_checks_its_facts() -> None:
    model = held(design_space(toys.Boundary(width=3))).ds.model
    width, inner = model.linked.keys["width"], model.linked.keys["outside.width"]
    assert dict(facts.open(model, {width: 4}).facts) == {width: 4}
    for given, refused in (
        ({}, "missing required parameters: width"),
        ({width: 3, inner: 3}, "outside.width is not an exposed parameter"),
        ({width: "wide"}, "width"),
    ):
        with pytest.raises(RequestError, match=refused):
            facts.open(model, given)


# -- requests --------------------------------------------------------------------------------


def test_a_change_applies_to_any_design_space_equal_to_its_own() -> None:
    """A change made on a design space applies to any design space equal to it (here, the
    same toy changed again)."""
    point, copy = _changed("house"), _changed("house")
    first = inspection.decisions(point)[0].reference
    made = requests.change(held(point), first, True)
    again = held(copy)
    assert again.ds == made.base and again.ds is not made.base
    assert list(requests.normalize(again, [made])) == [made.node]
    other = held(design_space(type(point)(budget=400)))  # type: ignore[call-arg]
    with pytest.raises(RequestError):
        requests.normalize(other, [made])


def test_a_change_made_at_a_childs_scope_applies_at_the_root() -> None:
    point = TOYS["house"].make()
    kitchen = point.kitchen  # type: ignore[attr-defined]
    reference = type(kitchen).finish
    made = requests.change(held(kitchen), reference, 2)
    root = held(point)
    assert made.scope == kitchen._scope and made.base == root.ds
    (normalized,) = requests.normalize(At(root.ds, 0), [made]).values()
    assert (normalized.node, normalized.value) == (made.node, 2)
