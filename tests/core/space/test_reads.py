# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What each method may read, inferred from its source, and each address's existence guards.

The evaluation core checks every read a ``self`` method makes against these reads
(``finn.core.space._reads``), so each way a method reads its Space is pinned here: a
member, a child's member, a candidate's member through its selector, ``getattr`` with a
fixed name, a helper method or property (in the scope it is reached through),
``present()``, a local bound once to a path, and what makes a method dynamic.
Function metadata cannot change those inferred reads.
"""

from __future__ import annotations

from typing import cast

from finn.core.space import Decision, Param, Select, Space, constraint, derived, view
from finn.core.space._reads import ModelReads, infer, summary
from finn.core.space._scopes import Address, Scopes, existence
from finn.core.space.compiler import compile_model


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(domain=(1, 2))

    @derived
    def bits(self) -> int:
        return self.width * self.lanes

    @property
    def doubled(self) -> int:
        return 2 * self.width

    def scaled(self, factor: int) -> int:
        return factor * self.lanes


class Wide(Space):
    width: int = Param()
    span: int = Decision(domain=(2, 4))


NAME = "width"


class Outer(Space):
    width: int = Param()
    enabled: bool = Decision(domain=(False, True))
    leaf = Leaf(width=width)
    guarded = Leaf(width=width, when=enabled)
    mode: Leaf | Wide = Select(leaf=Leaf(width=width), wide=Wide(width=width))

    @derived
    def direct(self) -> int:
        return self.width + 1

    @derived
    def through_child(self) -> int:
        return self.leaf.bits

    @derived
    def through_choice(self) -> int:
        return self.mode.width

    @derived
    def by_getattr(self) -> int:
        return cast(int, getattr(self, "width")) + cast(int, getattr(self, NAME))

    @derived
    def by_helper(self) -> int:
        return self.leaf.scaled(3)

    @derived
    def by_property(self) -> int:
        return self.leaf.doubled

    @derived
    def by_presence(self) -> bool:
        return self.present(Outer.guarded)

    @constraint
    def branching(self) -> bool:
        return self.enabled or self.direct > 2

    @view
    def explicit(*, width: int) -> int:
        return width


def _model() -> tuple[Scopes, ModelReads]:
    linked = compile_model(Outer).scopes
    return linked, infer(linked)


def _keys(nodes: frozenset[Address] | None) -> set[str]:
    assert nodes is not None
    return {index.key for index in nodes}


def _reads(linked: Scopes, reads: ModelReads, key: str) -> set[str]:
    return _keys(reads.methods[linked.keys[key]])


def test_member_child_and_candidate_paths() -> None:
    linked, reads = _model()
    assert _reads(linked, reads, "direct") == {"width"}
    assert _reads(linked, reads, "through_child") == {"leaf.bits"}
    # A member read through a Decision over nodes reads the selector and the member of
    # every candidate that has it.
    assert _reads(linked, reads, "through_choice") == {"mode", "mode.leaf.width", "mode.wide.width"}


def test_getattr_with_a_name_the_source_or_its_module_fixes() -> None:
    linked, reads = _model()
    # The literal is fixed; NAME is a module global, not a closure: it is not followed,
    # and the read through it makes the method dynamic.
    assert reads.methods[linked.keys["by_getattr"]] is None

    def make(name: str) -> type[Space]:
        def read(self: Space) -> int:
            return cast(int, getattr(self, name))

        members = {"__annotations__": {"width": int}, "width": Param(), "value": derived(read)}
        return cast(type[Space], type("Closure", (Space,), members))

    linked = compile_model(make("width")).scopes
    assert _reads(linked, infer(linked), "value") == {"width"}


def test_helpers_and_properties_are_read_in_the_scope_they_are_reached_through() -> None:
    linked, reads = _model()
    assert _reads(linked, reads, "by_helper") == {"leaf.lanes"}
    assert _reads(linked, reads, "by_property") == {"leaf.width"}


class Mutual(Space):
    """Two helpers that call each other, each reading another member."""

    width: int = Param()
    lanes: int = Decision(domain=(1, 2))

    def first(self, depth: int) -> int:
        return self.width if depth == 0 else self.second(depth - 1)

    def second(self, depth: int) -> int:
        return self.lanes if depth == 0 else self.first(depth - 1)

    @derived
    def by_first(self) -> int:
        return self.first(3)

    @derived
    def by_second(self) -> int:
        return self.second(3)


class PlacesMutual(Space):
    inner = Mutual(width=4)

    @derived
    def through(self) -> int:
        return self.inner.second(2)


def test_helpers_calling_each_other_each_read_what_both_read() -> None:
    """Inferred once per template and function, a helper cycle closes on what both read:
    each method's reads are the union, wherever it is reached first."""
    linked = compile_model(Mutual).scopes
    reads = infer(linked)
    assert (
        _reads(linked, reads, "by_first")
        == _reads(linked, reads, "by_second")
        == {
            "width",
            "lanes",
        }
    )
    linked = compile_model(PlacesMutual).scopes
    reads = infer(linked)
    assert _reads(linked, reads, "through") == {"inner.width", "inner.lanes"}
    assert _reads(linked, reads, "inner.by_first") == {"inner.width", "inner.lanes"}


def test_presence_reads_the_guards_on_the_way() -> None:
    linked, reads = _model()
    assert _reads(linked, reads, "by_presence") == {"guarded.$guard"}


def test_every_branch_is_a_possible_read() -> None:
    linked, reads = _model()
    assert _reads(linked, reads, "branching") == {"enabled", "direct"}


def test_explicit_functions_name_their_inputs_and_are_not_inferred() -> None:
    linked, reads = _model()
    assert linked.keys["explicit"] not in reads.methods
    assert all(index.call_style == "self" for index in reads.methods)


def test_function_metadata_cannot_replace_source_inference() -> None:
    def computed(self: Space) -> int:
        return sum(cast(int, getattr(self, name)) for name in ("a", "b"))

    def metadata(self: Space) -> int:
        return sum(cast(int, getattr(self, name)) for name in ("a", "b"))

    setattr(metadata, "__reads__", ("a", "b"))

    def fixed(self: Space) -> int:
        return cast(int, self.a) + cast(int, self.b)  # type: ignore[attr-defined]

    # A stale list can neither remove b nor add an invented source.
    setattr(fixed, "__reads__", ("a", "invented"))

    def handed_on(self: Space) -> int:
        return len(repr(self))

    members: dict[str, object] = {
        "__annotations__": {"a": int, "b": int},
        "a": Param(),
        "b": Param(),
        "computed": derived(computed),
        "metadata": derived(metadata),
        "fixed": derived(fixed),
        "handed_on": derived(handed_on),
    }
    linked = compile_model(cast(type[Space], type("Dynamic", (Space,), members))).scopes
    reads = infer(linked)
    for name in ("computed", "metadata", "handed_on"):
        assert reads.methods[linked.keys[name]] is None
    assert _reads(linked, reads, "fixed") == {"a", "b"}
    assert reads.unresolved.get(linked.keys["fixed"], ()) == ()
    found = summary(linked, reads)
    assert found["dynamic"] == 3 and found["dynamic_by_class"] == {
        "Dynamic": ["computed", "handed_on", "metadata"]
    }


class Aliased(Space):
    width: int = Param()
    leaf = Leaf(width=width)
    mode: Leaf | Wide = Select(leaf=Leaf(width=width), wide=Wide(width=width))

    @derived
    def through_a_local(self) -> int:
        leaf = self.leaf
        bits = leaf.bits
        return bits + leaf.width

    @derived
    def through_a_local_selection(self) -> int:
        core = self.mode
        return core.width

    @derived
    def rebound(self) -> int:
        core: Leaf | Wide = self.leaf
        if core.width > 4:
            core = self.mode
        return core.width


def test_a_local_bound_once_to_a_path_reads_through_that_path() -> None:
    linked = compile_model(Aliased).scopes
    reads = infer(linked)
    assert _reads(linked, reads, "through_a_local") == {"leaf.bits", "leaf.width"}
    assert _reads(linked, reads, "through_a_local_selection") == {
        "mode",
        "mode.leaf.width",
        "mode.wide.width",
    }
    # Bound twice, a local is no path: what is read through it is not inferred (the read
    # check refuses it when the method runs); the selector read to bind it is.
    assert _reads(linked, reads, "rebound") == {"mode"}


def test_unresolved_paths_are_kept_as_a_diagnostic() -> None:
    class Constants(Space):
        width: int = Param()
        LIMIT = 4

        @derived
        def capped(self) -> int:
            return min(self.width, self.LIMIT)

    linked = compile_model(Constants).scopes
    reads = infer(linked)
    assert _reads(linked, reads, "capped") == {"width"}
    assert reads.unresolved[linked.keys["capped"]] == ("LIMIT",)
    assert summary(linked, reads)["unresolved_paths"] == ["LIMIT"]


def test_existence_guards_innermost_first() -> None:
    linked, reads = _model()
    lanes = linked.keys["guarded.lanes"]
    assert [index.key for index in reads.guards(lanes)] == ["guarded.$guard"]
    span = linked.keys["mode.wide.span"]
    # A candidate's member exists while its case is selected.
    assert [index.key for index in reads.guards(span)] == ["mode.wide.$selected"]
    assert reads.guards(linked.keys["width"]) == ()
    assert existence(span) == reads.guards(span)


def test_nested_guards_reach_the_root() -> None:
    class Inner(Space):
        on: bool = Param()
        value: int = Decision(domain=(1,), when=on)

    class Middle(Space):
        on: bool = Param()
        inner = Inner(on=on, when=on)

    class Top(Space):
        on: bool = Param()
        middle = Middle(on=on, when=on)

    linked = compile_model(Top).scopes
    guards = existence(linked.keys["middle.inner.value"])
    assert [index.key for index in guards] == [
        "middle.inner.value.$guard",
        "middle.inner.$guard",
        "middle.$guard",
    ]


def test_a_chain_of_closures_is_inferred_from_one_parse() -> None:
    def step(previous: str) -> object:
        def compute(self: Space) -> int:
            return cast(int, getattr(self, previous)) + 1

        return derived(compute)

    members: dict[str, object] = {"__annotations__": {"value_0": int}, "value_0": Param()}
    for index in range(1, 2_001):
        members[f"value_{index}"] = step(f"value_{index - 1}")
    linked = compile_model(cast(type[Space], type("Chain", (Space,), members))).scopes
    reads = infer(linked)
    for index in range(1, 2_001):
        assert _reads(linked, reads, f"value_{index}") == {f"value_{index - 1}"}
