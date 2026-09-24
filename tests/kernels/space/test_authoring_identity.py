# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authored persistence identities cannot impersonate nesting/generated names."""

from typing import cast

import pytest

from finn.kernels.space import (
    Const,
    Space,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ViewKey,
    compile_space,
)
from finn.kernels.space.errors import DefinitionError


@pytest.mark.parametrize("name", ["", "a.b", "$selector", "has space", "ümlaut"])
def test_authored_names_are_unambiguous_segments(name: str) -> None:
    with pytest.raises(DefinitionError, match="name segment"):
        ValueKey(name, int)
    with pytest.raises(DefinitionError, match="name segment"):
        ViewKey(name, int)
    with pytest.raises(DefinitionError, match="name segment"):
        SubspaceChoice({name: Subspace(Space)})
    family = type("BadIdentity", (Space,), {name: Const(3)})
    with pytest.raises(DefinitionError, match="name segment"):
        compile_space(family)


def test_malformed_alternatives_fail_as_definitions() -> None:
    with pytest.raises(DefinitionError, match="Space subclass"):
        Subspace(cast(type[Space], int))
    with pytest.raises(DefinitionError, match="Subspace placement"):
        SubspaceChoice({"case": cast(Subspace[Space], 4)})
    value_key = ValueKey("result", int)
    view_key = ViewKey("result", int)
    with pytest.raises(DefinitionError, match="duplicate choice export"):
        SubspaceChoice({"case": Subspace(Space)}, exports=(value_key, view_key))


def test_case_ids_can_use_hyphens_without_becoming_paths() -> None:
    class Root(Space):
        implementation = SubspaceChoice({"low-area": Subspace(Space)})

    assert Root.start().implementation.alternatives == ("low-area",)
