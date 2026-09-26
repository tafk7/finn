# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authored persistence identities cannot impersonate nesting/generated names."""

from typing import cast

import pytest

from finn.core.space import (
    Const,
    Decision,
    Param,
    Space,
    View,
    ViewKey,
    composite,
    configure,
    inspection,
)
from finn.core.space.compiler import compile_space
from finn.core.space.errors import DefinitionError


@pytest.mark.parametrize("name", ["", "a.b", "$selector", "has space", "ümlaut"])
def test_authored_names_are_unambiguous_segments(name: str) -> None:
    # ValueKey is removed; ViewKey is the only export key. A candidate key of a
    # Decision over nodes replaces a SubspaceChoice case id.
    with pytest.raises(DefinitionError, match="name segment"):
        ViewKey(name, int)
    with pytest.raises(DefinitionError, match="name segment"):
        Decision(values={name: Space()})
    with pytest.raises(DefinitionError, match="name segment"):
        composite("Named", {name: Const(3)})
    family = type("BadIdentity", (Space,), {name: Const(3)})
    with pytest.raises(DefinitionError, match="name segment"):
        compile_space(family)


def test_malformed_candidates_fail_as_definitions() -> None:
    # Subspace(F) is gone: a node is declared by calling its family, so the
    # malformed forms are a candidate or a reference input that is not a
    # node, and a composite over a non-Space base.
    class Holder(Space):
        held: Param[Space] = Param(Space)

    with pytest.raises(DefinitionError, match="Space base family"):
        composite("NotSpace", {}, base=cast(type[Space], int))
    with pytest.raises(DefinitionError, match="expected a node declaration or None"):
        Decision(values={"case": cast(Space, 4)})
    with pytest.raises(DefinitionError, match="takes a node declaration"):
        Holder(held=cast(Space, 4))
    with pytest.raises(DefinitionError, match="at least one candidate"):
        Decision(values={})
    with pytest.raises(DefinitionError, match="needs a node candidate"):
        Decision(values={"none": None})
    # Choice exports are removed; a duplicate export name is refused on the family.
    first = ViewKey("result", int)
    second = ViewKey("result", int)

    class Duplicate(Space):
        value = Const(1)
        result = View(value)
        exports = {first: result, second: result}

    with pytest.raises(DefinitionError, match="duplicate export key result"):
        compile_space(Duplicate)


def test_case_ids_can_use_hyphens_without_becoming_paths() -> None:
    class Root(Space):
        implementation = Decision(values={"low-area": Space()})

    point = configure(Root())
    (choice,) = inspection.choices(point)
    assert [case.name for case in choice.cases] == ["low-area"]
    assert [case.scope for case in choice.cases] == ["implementation.low-area"]
    assert [item.key for item in inspection.decisions(point)] == ["implementation"]
    assert point.with_choices(implementation="low-area").implementation is not None
