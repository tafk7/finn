# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A method's ``present()`` of a value input bound through a chain of aliases, the middle
one under a guard of its own that the first one's guard implies.

Collapse takes the first alias straight to the chain's source, past the middle one (its
guard holds whenever the first one's does). A method's reader walks the chain as resolved,
each alias in turn, and reads each one's guard to know whether a source applies. What the
method may read (``_reads``) is found on the same chain as resolved, so it names the middle
alias's guard as well, and the read check passes: the public API and the public API with no
cache answer alike. (Found on the collapsed chain, it did not: the running type failed
here, ``read-undeclared`` on ``mid.$guard``.)

- ``mid`` is placed under ``Top.placed`` (``mid.$guard``); in it, ``inner`` under
  ``entered`` (``mid.inner.$guard``, under ``mid.$guard``) and ``leaf`` under ``lit``
  (``mid.leaf.$guard``, under ``mid.$guard``).
- ``inner.x`` is bound to ``mid.b``, a member of ``mid`` (under ``mid.$guard``) that is
  ``leaf.v``: the chain ``inner.x``, ``mid.b``, ``mid.leaf.v``.
- ``inner.has`` asks ``present(Inner.x)``: collapsed, ``inner.x`` goes to ``mid.leaf.v``
  past ``mid.b``, so ``mid.$guard`` is no guard on that way; as resolved, it is ``mid.b``'s.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from support.space_oracle import Implementation, compare, forms, load
from support.space_oracle.public import PUBLIC

from finn.core.space import Decision, Param, Space, bind, derived, design_space
from finn.core.space._reads import infer
from finn.core.space._structure import Structure, expand
from finn.core.space.occurrence import held

RUNNING: Implementation = load("core.space._running_type:RUNNING")


class Inner(Space):
    x: int = Param(required=False)

    @derived
    def has(self) -> bool:
        return self.present(Inner.x)


class Leaf(Space):
    v: int = Param()


class Mid(Space):
    gate: int = Decision(values=(0, 1))
    pick: int = Decision(values=(0, 1))

    @derived
    def entered(self) -> bool:
        return self.gate == 1

    @derived
    def lit(self) -> bool:
        return self.pick == 1

    leaf = Leaf(v=3, when=lit)
    b = leaf.v
    inner = Inner(x=b, when=entered)


class Top(Space):
    on: int = Decision(values=(0, 1))

    @derived
    def placed(self) -> bool:
        return self.on == 1

    mid = Mid(when=placed)


def _point(stored: dict[object, object]) -> Space:
    return cast(Space, bind(design_space(Top()), stored).instance)


def _asked(implementation: Implementation, point: Space) -> forms.Form:
    """``inner.has``'s answer at ``point``, or what asking it raised, as a form."""
    try:
        return forms.answer(implementation.open(point).query("mid.inner.has"))
    except Exception as error:  # noqa: BLE001 - what was raised is the answer compared
        return forms.raised(error)


def test_the_chain_is_collapsed_past_the_middle_alias_under_its_own_implied_guard() -> None:
    linked = cast(Structure, held(design_space(Top())).ds.model.linked)
    keys = linked.keys
    first, middle, end = keys["mid.inner.x"], keys["mid.b"], keys["mid.leaf.v"]
    raw = linked.raw_nodes
    assert (raw[first].output, raw[middle].output) == (middle, end)
    assert raw[middle].guard == keys["mid.$guard"]
    guard = raw[first].guard
    assert guard is not None and raw[guard].guard == keys["mid.$guard"]  # implied by it
    assert linked.nodes[first].output == end  # collapse goes past the middle alias


def test_what_present_may_read_names_every_guard_on_the_chain_as_resolved() -> None:
    model = held(design_space(Top())).ds.model
    linked = model.linked
    keys = linked.keys
    allowed = model.per_class.reads.methods[keys["mid.inner.has"]]
    assert allowed is not None
    found = sorted(linked.header(index).key for index in allowed)
    assert found == ["mid.$guard", "mid.inner.$guard", "mid.leaf.$guard"]


def test_a_table_names_the_same_guards_through_the_aliases_collapse_went_past() -> None:
    table = expand(Top)
    (method,) = [node.index for node in table.nodes if node.key == "mid.inner.has"]
    allowed = infer(table).methods[method]
    assert allowed is not None
    found = sorted(table.nodes[index].key for index in allowed)
    assert found == ["mid.$guard", "mid.inner.$guard", "mid.leaf.$guard"]


@pytest.mark.parametrize(
    ("pick", "expected"),
    [(1, {"kind": "value", "value": True}), (0, {"kind": "value", "value": False})],
)
def test_present_through_the_chain_answers_alike_with_and_without_the_cache(
    pick: int, expected: dict[str, Any]
) -> None:
    point = _point({Top.on: 1, Top.mid.gate: 1, Top.mid.pick: pick})
    public, uncached = _asked(PUBLIC, point), _asked(RUNNING, point)
    assert (public, uncached) == (expected, expected), f"public: {public}; no cache: {uncached}"
    assert compare.everything(PUBLIC.open(point), RUNNING.open(point)).lines() == []


@pytest.mark.parametrize("seed", range(4))
def test_a_walk_over_the_chain_answers_alike_with_and_without_the_cache(seed: int) -> None:
    found = compare.walk(lambda: design_space(Top()), {}, seed, 20, PUBLIC, RUNNING, forms.ALL)
    assert found.lines() == []
