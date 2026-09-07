# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``name=`` is the stable identity a persistence layer keys a choice on.

This package persists nothing and imports no ONNX.  What it owes a persistence
layer is narrower and testable here: every committable choice beneath one root,
each with an identity that is stable across the two things that legitimately
change around it -- the root namespace it is started under, and the Python
member name it happens to be spelled with.

``name=`` is the only override.  There is no ``persist_as``: a second naming
mechanism would mean two answers to "what is this choice called" and one of them
silently winning.

**The dotted strings below are identities, not serialized spellings.**  Nothing
here fixes what a native ONNX attribute is finally called, and no assertion in
this file should be read as locking ``dataflow_design.case`` as the thing a user
sees in Netron.  S2-A owns that mapping -- it may well render a nested selector
as ``dataflow_design`` and a named ``PE`` as ``PE`` -- and owns the collision
rule that makes such a rendering safe.  What the model guarantees, and all it
guarantees, is that the identity S2-A maps *from* is stable and unique.
"""

from __future__ import annotations


import pytest

from finn.dataflow.model.declarations import (
    AuthoringError,
    Decision,
    Input,
    Problem,
    Space,
    Subspace,
    SubspaceChoice,
    derived,
)
from finn.dataflow.model.occurrence import occurrence_persistable


class Leaf(Space):
    size = Input(int)
    tile = Decision(int, values=(1, 2))

    @derived(int, size=size, tile=tile)
    def result(*, size: int, tile: int) -> int:
        return size * tile

    exports = (result,)


class Renamed(Space):
    """A child whose Decision is persisted under a name of the author's choosing."""

    size = Input(int)
    tile = Decision(int, values=(1, 2), name="tile_size")

    @derived(int, size=size, tile=tile)
    def result(*, size: int, tile: int) -> int:
        return size * tile

    exports = (result,)


class Root(Space):
    size = Problem(int)
    held = Subspace(Leaf, size=size)
    #: The same class, placed twice; the second placement is renamed.
    second = Subspace(Leaf, size=size, name="spare")
    implementation = SubspaceChoice(
        {"plain": Subspace(Leaf, size=size), "renamed": Subspace(Renamed, size=size)},
        outputs=("result",),
        name="dataflow_design",
    )

    @derived(int, selected=implementation.result)
    def observed(*, selected: int) -> int:
        return selected


def _paths(root: Space) -> tuple[str, ...]:
    return tuple(item.path for item in occurrence_persistable(root))


def test_every_committable_choice_is_named_relative_to_its_root() -> None:
    """One flat identity per choice, including the ones nobody enumerated.

    Asserted exactly because it is the input to S2-A's attribute mapping, not
    because these strings are what any attribute is called.
    """

    assert _paths(Root.start({Root.size: 3})) == (
        "dataflow_design.case",
        "dataflow_design.plain.tile",
        "dataflow_design.renamed.tile_size",
        "held.tile",
        "spare.tile",
    )


def test_the_same_root_under_two_namespaces_produces_the_same_names() -> None:
    """A persisted document must reload under either, so the prefix is removed."""

    default = _paths(Root.start({Root.size: 3}))
    elsewhere = _paths(Root.start({Root.size: 3}, namespace="node_17"))
    assert default == elsewhere


def test_a_name_overrides_the_member_it_is_spelled_with() -> None:
    """The member name is a Python fact; ``name=`` is the one that travels."""

    names = _paths(Root.start({Root.size: 3}))
    assert "dataflow_design.case" in names
    assert "implementation.case" not in names
    assert "spare.tile" in names
    assert "second.tile" not in names


def test_a_valid_name_is_stable_across_a_python_member_rename() -> None:
    class Before(Space):
        old_spelling = Decision(int, values=(1, 2), name="stable_identity")

    class After(Space):
        new_spelling = Decision(int, values=(1, 2), name="stable_identity")

    assert _paths(Before.start({})) == ("stable_identity",)
    assert _paths(After.start({})) == ("stable_identity",)


def test_nested_identity_comes_from_real_structural_nesting() -> None:
    class Middle(Space):
        size = Input(int)
        leaf = Subspace(Leaf, size=size)

    class NestedRoot(Space):
        size = Problem(int)
        middle = Subspace(Middle, size=size)

    assert _paths(NestedRoot.start({NestedRoot.size: 3})) == ("middle.leaf.tile",)


def test_two_placements_of_one_class_do_not_collide() -> None:
    names = _paths(Root.start({Root.size: 3}))
    assert names.count("held.tile") == 1
    assert names.count("spare.tile") == 1


def test_selectors_come_before_the_decisions_whose_existence_they_decide() -> None:
    """Replay order, not presentation order: which case is live decides what exists."""

    items = occurrence_persistable(Root.start({Root.size: 3}))
    selectors = tuple(index for index, item in enumerate(items) if item.selector)
    decisions = tuple(index for index, item in enumerate(items) if not item.selector)
    assert selectors and decisions
    assert max(selectors) < min(decisions)


def test_a_selector_is_reported_as_one_and_carries_no_codec() -> None:
    """Its values are alternative ids the branch owns, not a declared encoding."""

    selector = next(item for item in occurrence_persistable(Root.start({Root.size: 3})))
    assert selector.selector is True
    assert selector.codec is None
    assert (selector.owner, selector.member) == ("Root", "implementation")


def test_a_singleton_choice_persists_nothing_and_gains_a_selector_later() -> None:
    """Adding a second candidate must not rename what the first one already wrote."""

    class One(Space):
        size = Problem(int)
        implementation = SubspaceChoice({"plain": Subspace(Leaf, size=size)})

    class Two(Space):
        size = Problem(int)
        implementation = SubspaceChoice(
            {"plain": Subspace(Leaf, size=size), "renamed": Subspace(Renamed, size=size)}
        )

    assert _paths(One.start({One.size: 3})) == ("implementation.plain.tile",)
    assert _paths(Two.start({Two.size: 3})) == (
        "implementation.case",
        "implementation.plain.tile",
        "implementation.renamed.tile_size",
    )


def test_a_name_is_the_only_naming_override_a_declaration_takes() -> None:
    """No ``persist_as``: one mechanism, so there is one answer."""

    with pytest.raises(TypeError):
        Decision(int, values=(1, 2), persist_as="tile")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SubspaceChoice(  # type: ignore[call-arg]
            {"plain": Subspace(Leaf, size=Input(int))}, persist_as="design"
        )


def test_an_empty_name_is_refused_rather_than_falling_back_to_the_member() -> None:
    """An override that silently reverted would persist under the wrong name."""

    with pytest.raises(AuthoringError):
        Subspace(Leaf, name="", size=Input(int))
    with pytest.raises(AuthoringError):
        Decision(int, values=(1, 2), name="")
    with pytest.raises(AuthoringError):
        SubspaceChoice({"plain": Subspace(Leaf, size=Input(int))}, name="")


def test_a_local_name_cannot_inject_structural_path_segments() -> None:
    with pytest.raises(AuthoringError, match="one path segment"):
        Decision(int, values=(1, 2), name="outer.inner")
