# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The root-occurrence factory, and the capability boundary it is there to keep.

A caller that owns something the design space has no declaration for -- an
operation holding a frozen ONNX node -- has to get that context onto its root
occurrence somehow.  The rejected way was a payload field on
``OccurrenceContext``, which every authored class in the tree can read.  These
tests pin the accepted way: a factory consulted for the root and for successor
roots, and for nothing else.
"""

from __future__ import annotations

import pytest

from finn.dataflow.model import (
    AuthoringError,
    Decision,
    OccurrenceContext,
    Problem,
    Space,
    Subspace,
    Variant,
    compile_space_model,
    finite,
)


class Leaf(Space):
    """A child that records every context its own allocator was handed."""

    seen: list[OccurrenceContext] = []
    scale = Decision(int, domain=finite((1, 2, 4)))

    @classmethod
    def _new_occurrence(cls, context: OccurrenceContext) -> Space:
        cls.seen.append(context)
        return super()._new_occurrence(context)


class OtherLeaf(Leaf):
    """A second child class, so a Variant has something to choose between."""

    seen: list[OccurrenceContext] = []


class Bound(Space):
    """A root whose instances carry a payload the design space cannot see."""

    width = Problem(int)
    child = Subspace(Leaf)
    pick = Variant({"a": Subspace(Leaf), "b": Subspace(OtherLeaf)})

    payload: object = None

    @classmethod
    def _new_occurrence(cls, context: OccurrenceContext) -> Space:
        raise AssertionError("a factory-started lineage must not fall back to _new_occurrence")


def _model():
    return compile_space_model(Bound, "root", problem_namespace="problem.root")


def _factory(payload: object):
    def make(context: OccurrenceContext) -> Bound:
        instance = object.__new__(Bound)
        instance.payload = payload
        return instance

    return make


def test_the_factory_allocates_the_root():
    payload = object()
    root = _model().start({Bound.width: 8}, root_factory=_factory(payload))

    assert isinstance(root, Bound)
    assert root.payload is payload


def test_a_successor_root_is_built_by_the_same_factory():
    payload = object()
    root = _model().start({Bound.width: 8}, root_factory=_factory(payload))

    successor = root.child.assign(Leaf.scale, 2).root

    assert isinstance(successor, Bound)
    assert successor.payload is payload
    assert successor is not root


def test_a_child_is_never_built_by_the_factory():
    """The capability boundary, asserted rather than described.

    ``Bound._new_occurrence`` raises, so the factory is provably the only thing
    that built the root; ``Leaf._new_occurrence`` records, so the children are
    provably built by their own authored class and were handed a context whose
    ``root`` is the operation -- not the operation's payload.
    """

    Leaf.seen.clear()
    OtherLeaf.seen.clear()
    root = _model().start({Bound.width: 8}, root_factory=_factory(object()))

    child = root.child
    alternative = root.pick.alternative("b")

    assert [context.space_type for context in Leaf.seen] == [Leaf]
    assert [context.space_type for context in OtherLeaf.seen] == [OtherLeaf]
    for context in (*Leaf.seen, *OtherLeaf.seen):
        assert context.root is root
        assert not context.is_root
    assert child is not root
    assert alternative is not root


def test_an_ordinary_lineage_still_uses_the_authored_hook():
    Leaf.seen.clear()
    root = compile_space_model(Leaf, "root", problem_namespace="problem.root").start({})

    assert isinstance(root, Leaf)
    assert [context.is_root for context in Leaf.seen] == [True]


def test_reconstruct_keeps_the_factory():
    """A reconstruction drops assignments; it must not drop the binding.

    Silently returning a root of the right class with none of the caller's
    context is the failure this pins: it type-checks, it answers most queries,
    and it is not the occurrence the caller asked to rebuild.
    """

    payload = object()
    root = _model().start({Bound.width: 8}, root_factory=_factory(payload))

    fresh = root.reconstruct()

    assert isinstance(fresh, Bound)
    assert fresh.payload is payload
    assert fresh is not root


def test_a_factory_returning_the_wrong_class_is_refused():
    def wrong(context: OccurrenceContext) -> Bound:
        return object.__new__(Leaf)  # type: ignore[return-value]

    with pytest.raises(AuthoringError, match="lineage root factory"):
        _model().start({Bound.width: 8}, root_factory=wrong)


def test_a_factory_returning_an_attached_occurrence_is_refused():
    started = _model().start({Bound.width: 8}, root_factory=_factory(object()))

    with pytest.raises(AuthoringError, match="already attached"):
        _model().start({Bound.width: 8}, root_factory=lambda context: started)
