# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three-method seam a layer specializes a ``SubspaceChoice`` through.

The point of the seam is what it *does not* let a specialization do.  A layer
says which ids its candidates already carry, which ones it admits, and which
outputs it implies; it does not get to create a selector, gate a case, name a
namespace, or build a view, because there is one of each and the generic
compiler owns them.  Every test here therefore checks a specialization against
the generic machinery rather than against a second copy of it.

The synthetic ``Tagged`` specialization below stands in for the real one
(``KernelChoice``, and the physical-strategy choice U6 will add) so this package
stays layer-neutral: nothing in ``finn.dataflow.model`` may know what a Kernel
is, including its tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, cast

import pytest

from finn.dataflow._engine import Decided, QualifiedPath, Unresolved
from finn.dataflow.space.compiler import compile_space, compile_space_model
from finn.dataflow import space
from finn.dataflow.space.declarations import (
    AuthoringError,
    Decision,
    Input,
    Problem,
    Space,
    Subspace,
    SubspaceChoice,
    ValueSource,
    derived,
)


class Wide(Space):
    """A candidate that carries its own stable id."""

    id: ClassVar[str] = "wide"

    size = Input(int)

    @derived(int, size=size)
    def width(*, size: int) -> int:
        return size * 2

    exports = (width,)


class Narrow(Space):
    id: ClassVar[str] = "narrow"

    size = Input(int)

    @derived(int, size=size)
    def width(*, size: int) -> int:
        return size

    exports = (width,)


class Anonymous(Space):
    """A candidate with no id of its own."""

    size = Input(int)

    @derived(int, size=size)
    def width(*, size: int) -> int:
        return size

    exports = (width,)


class Tagged(SubspaceChoice):
    """A specialization that uses all three seam methods and adds nothing else."""

    #: The generic selector, renamed -- not a second selector.
    selector_name: ClassVar[str] = "tag"

    __slots__ = ()

    def __init__(
        self,
        *candidates: Subspace[Space],
        outputs: Sequence[str] | None = None,
        when: ValueSource[bool] | None = None,
        role: str | None = None,
    ) -> None:
        self._from_candidates(candidates, outputs=outputs, when=when, name=role)

    def candidate_id(self, subspace: Subspace[Space]) -> str:
        if subspace.stable_name is not None:
            return subspace.stable_name
        candidate = str(getattr(subspace.space_type, "id", "") or "")
        if not candidate:
            raise AuthoringError(f"{subspace.space_type.__name__} declares no stable id")
        return candidate

    def validate_candidate(self, owner: type[Space], subspace: Subspace[Space]) -> None:
        del owner
        if not hasattr(subspace.space_type, "id"):
            raise AuthoringError(f"{subspace.space_type.__name__} is not tagged")

    def default_outputs(self) -> tuple[str, ...]:
        return ("width",)


# -- candidate_id -------------------------------------------------------------


def test_a_candidate_names_itself_and_is_never_named_twice() -> None:
    """The whole reason the seam exists: self-naming candidates."""

    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Wide, size=size), Subspace(Narrow, size=size))

    assert tuple(item for item, _ in Root.implementation.alternatives) == ("wide", "narrow")


def test_an_alias_overrides_a_candidates_own_id() -> None:
    """One class in two slots, which is exactly what an id-carrying candidate cannot do."""

    class Root(Space):
        size = Problem(int)
        implementation = Tagged(
            Subspace(Wide, size=size, name="first"),
            Subspace(Wide, size=size, name="second"),
        )

    assert tuple(item for item, _ in Root.implementation.alternatives) == ("first", "second")


def test_a_candidate_without_an_id_is_refused_where_it_is_written() -> None:
    with pytest.raises(AuthoringError, match="Anonymous declares no stable id"):
        Tagged(Subspace(Anonymous, size=Input(int)))


def test_a_generic_choice_names_alternatives_with_mapping_keys() -> None:
    """The base ``candidate_id`` is a refusal, not a guess at a convention."""

    generic = SubspaceChoice({"wide": Subspace(Wide, size=Input(int))})
    with pytest.raises(AuthoringError, match="names its alternatives with mapping keys"):
        generic.candidate_id(Subspace(Narrow, size=Input(int)))


def test_the_compiler_still_owns_the_duplicate_id_rule() -> None:
    """A self-named duplicate is refused by the one rule, not by a second copy."""

    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Wide, size=size), Subspace(Wide, size=size))

    with pytest.raises(AuthoringError, match="declares alternative id 'wide' twice"):
        compile_space(Root, "root", problem_namespace="problem.root")


# -- validate_candidate -------------------------------------------------------


def test_a_refused_candidate_names_the_member_that_declared_it() -> None:
    """The specialization says what is wrong; the caller says where it is written."""

    class Untagged(Space):
        size = Input(int)

        @derived(int, size=size)
        def width(*, size: int) -> int:
            return size

        exports = (width,)

    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Untagged, size=size, name="untagged"))

    with pytest.raises(AuthoringError, match=r"Root\.implementation: Untagged is not tagged"):
        compile_space(Root, "root", problem_namespace="problem.root")


def test_admission_can_be_run_early_without_restating_the_rule() -> None:
    """A layer consuming a candidate before compilation reuses the one wrapper.

    Private, and deliberately not a free function on the public model surface:
    there is one way to invoke the rule and one place the member name is
    attached, whether the caller is the compiler or a layer running early.
    """

    class Untagged(Space):
        size = Input(int)

    choice = Tagged(Subspace(Wide, size=Input(int)))
    with pytest.raises(AuthoringError, match=r"Wide\.segment: Untagged is not tagged"):
        choice._admit_candidate(Wide, "segment", Subspace(Untagged, size=Input(int)))


def test_the_admission_wrapper_is_not_public_space_vocabulary() -> None:
    """Compiler plumbing, not something an ordinary Space author calls."""

    assert "admit_candidate" not in space.__all__
    assert not hasattr(space, "admit_candidate")


# -- one structural rule, whichever spelling produced the alternative ---------


def test_a_specialization_cannot_widen_what_an_alternative_may_be() -> None:
    """``candidate_id`` supplies the id, never permission to skip the checks."""

    class Empty(Tagged):
        __slots__ = ()

        def candidate_id(self, subspace: Subspace[Space]) -> str:
            return ""

    class NotAString(Tagged):
        __slots__ = ()

        def candidate_id(self, subspace: Subspace[Space]) -> str:
            return cast(str, None)

    with pytest.raises(AuthoringError, match="non-empty string"):
        Empty(Subspace(Wide, size=Input(int)))
    with pytest.raises(AuthoringError, match="non-empty string"):
        NotAString(Subspace(Wide, size=Input(int)))


def test_a_non_subspace_candidate_is_refused_in_both_spellings() -> None:
    with pytest.raises(AuthoringError, match="not a Subspace"):
        Tagged(cast("Subspace[Space]", "wide"))
    with pytest.raises(AuthoringError, match="not a Subspace"):
        SubspaceChoice({"wide": cast("Subspace[Space]", "wide")})


def test_a_generic_choice_admits_every_subspace() -> None:
    class Root(Space):
        size = Problem(int)
        implementation = SubspaceChoice({"any": Subspace(Anonymous, size=size)})

    compile_space(Root, "root", problem_namespace="problem.root")


# -- default_outputs ----------------------------------------------------------


def test_a_specialization_implies_its_selected_outputs() -> None:
    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Wide, size=size), Subspace(Narrow, size=size))

        @derived(int, selected=implementation.width)
        def observed(*, selected: int) -> int:
            return selected

    root = Root.start({Root.size: 3})
    selected = root.implementation.select("narrow")
    assert cast("Space", selected.root).answer(Root.observed) == Decided(3)


def test_an_author_may_still_name_the_outputs_explicitly() -> None:
    choice = Tagged(Subspace(Wide, size=Input(int)), outputs=())
    assert choice.outputs == ()


def test_a_generic_choice_implies_nothing() -> None:
    assert SubspaceChoice({"any": Subspace(Anonymous, size=Input(int))}).outputs == ()


# -- one selector, one view, one catalog --------------------------------------


def test_a_specialization_reaches_the_generic_selector_and_view() -> None:
    """No layer path: the same catalog, the same selector, the same ``ChoiceView``."""

    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Wide, size=size), Subspace(Narrow, size=size), role="body")

    model = compile_space_model(Root, "root", problem_namespace="problem.root")
    branch = model.branches.branch("root.body")
    assert branch.selector == QualifiedPath("root.body.tag")
    assert tuple(case.id for case in branch.cases) == ("wide", "narrow")

    root = Root.start({Root.size: 3})
    view = root.implementation
    assert view.alternatives == ("wide", "narrow")
    assert isinstance(view.selected(), Unresolved)
    assert view.select("narrow").selected() == Decided("narrow")
    assert type(view.alternative("narrow")) is Narrow


def test_a_singleton_specialization_adds_no_selector() -> None:
    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Wide, size=size))

    model = compile_space_model(Root, "root", problem_namespace="problem.root")
    assert model.branches.branch("root.implementation").selector is None


def test_a_specialization_does_not_change_how_a_nested_decision_compiles() -> None:
    class Deciding(Space):
        id: ClassVar[str] = "deciding"

        size = Input(int)
        tile = Decision(int, values=(1, 2))

        @derived(int, size=size, tile=tile)
        def width(*, size: int, tile: int) -> int:
            return size * tile

        exports = (width,)

    class Root(Space):
        size = Problem(int)
        implementation = Tagged(Subspace(Deciding, size=size), Subspace(Wide, size=size))

    model = compile_space_model(Root, "root", problem_namespace="problem.root")
    case = model.branches.branch("root.implementation").case("deciding")
    assert case.decision_paths == (QualifiedPath("root.implementation.deciding.tile"),)
