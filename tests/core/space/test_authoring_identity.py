# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authored persistence identities cannot impersonate nesting/generated names."""

from typing import cast

import pytest
from support.refusals import refused

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
    # Export keys (ViewKey) and the candidate keys of a Decision over nodes are
    # authored names, so each must be one segment.
    with refused("name-invalid", error=DefinitionError, message="name segment"):
        ViewKey(name, int)
    with refused("name-invalid", error=DefinitionError):
        Decision({name: Space()})
    with refused("name-invalid", error=DefinitionError):
        composite("Named", {name: Const(3)})
    space_type = type("BadIdentity", (Space,), {name: Const(3)})
    with refused("name-invalid", error=DefinitionError):
        compile_model(space_type)


def test_malformed_candidates_fail_as_definitions() -> None:
    # A node is declared by calling its Space class, so the malformed forms are a
    # candidate that is not a node, and a composite over a non-Space base.
    class Holder(Space):
        held: int = Param()

    with refused("composite-base-not-space", error=DefinitionError, message="Space base class"):
        composite("NotSpace", {}, base=cast(type[Space], int))
    with refused("candidate-not-space", error=DefinitionError):
        Decision({"case": cast(Space, 4)})
    # Candidates are entries, not a mapping of values.
    with refused(
        "node-decision-not-entries",
        error=DefinitionError,
        message="lists its candidates as entries",
    ):
        Decision(values={"case": Holder()})
    with refused(
        "node-not-value",
        error=DefinitionError,
        message="a node or a Decision over nodes is not a value",
    ):
        Holder(held=cast(int, Holder(held=1)))
    with refused(
        "node-decision-no-candidate", error=DefinitionError, message="at least one candidate"
    ):
        Decision({})
    with refused("node-decision-no-candidate", error=DefinitionError):
        Decision({}, optional=True)  # the None candidate alone places nothing
    # A duplicate export name is refused on the Space class.
    first = ViewKey("result", int)
    second = ViewKey("result", int)

    class Duplicate(Space):
        value = Const(1)
        result = View(value)
        exports = {first: result, second: result}

    with refused("export-key-duplicate", error=DefinitionError, message=r"\bresult$"):
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
