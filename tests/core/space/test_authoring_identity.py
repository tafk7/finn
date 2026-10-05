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
    design_space,
    inspection,
)
from finn.core.space.compiler import compile_model
from finn.core.space.errors import DefinitionError


@pytest.mark.parametrize("name", ["", "a.b", "$selector", "has space", "ümlaut"])
def test_authored_names_are_unambiguous_segments(name: str) -> None:
    # ValueKey is removed; ViewKey is the only export key. A candidate key of a
    # Decision over nodes replaces a SubspaceChoice case id.
    with pytest.raises(DefinitionError, match="name segment"):
        ViewKey(name, int)
    with pytest.raises(DefinitionError, match="name segment"):
        Decision({name: Space()})
    with pytest.raises(DefinitionError, match="name segment"):
        composite("Named", {name: Const(3)})
    space_type = type("BadIdentity", (Space,), {name: Const(3)})
    with pytest.raises(DefinitionError, match="name segment"):
        compile_model(space_type)


def test_malformed_candidates_fail_as_definitions() -> None:
    # Subspace(F) is gone: a node is declared by calling its Space class, so the
    # malformed forms are a candidate or a reference input that is not a
    # node, and a composite over a non-Space base.
    class Holder(Space):
        held: Space = Param()

    with pytest.raises(DefinitionError, match="Space base class"):
        composite("NotSpace", {}, base=cast(type[Space], int))
    with pytest.raises(DefinitionError, match="must be a Space class or a call on one"):
        Decision({"case": cast(Space, 4)})
    # The mapping spelling is retired: candidates are entries.
    with pytest.raises(DefinitionError, match="lists its candidates as entries"):
        Decision(values={"case": Holder()})
    with pytest.raises(DefinitionError, match="takes a node declaration"):
        Holder(held=cast(Space, 4))
    with pytest.raises(DefinitionError, match="at least one candidate"):
        Decision({})
    with pytest.raises(DefinitionError, match="at least one candidate"):
        Decision({}, optional=True)  # the None candidate alone places nothing
    # Choice exports are removed; a duplicate export name is refused on the Space class.
    first = ViewKey("result", int)
    second = ViewKey("result", int)

    class Duplicate(Space):
        value = Const(1)
        result = View(value)
        exports = {first: result, second: result}

    with pytest.raises(DefinitionError, match="duplicate export key result"):
        compile_model(Duplicate)


def test_case_ids_can_use_hyphens_without_becoming_paths() -> None:
    class Root(Space):
        implementation: Space = Decision({"low-area": Space()})

    point = design_space(Root())
    (choice,) = inspection.choices(point)
    assert [case.name for case in choice.cases] == ["low-area"]
    assert [case.scope for case in choice.cases] == ["implementation.low-area"]
    assert [item.key for item in inspection.decisions(point)] == ["implementation"]
    assert point.with_choices(implementation="low-area").implementation is not None
