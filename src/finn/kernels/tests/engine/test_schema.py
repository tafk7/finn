############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Schema-authoring error contracts (E7).

Malformed schemas are caught once, at construction, via the dependency topo-sort:
duplicate axis name, dependency on an unknown axis, and a dependency cycle all raise
SchemaError. Independent axes keep declaration order so the schema reads predictably.
"""

import pytest

from finn.kernels.engine.axis import Axis, discrete_axis
from finn.kernels.engine.schema import Schema, SchemaError


def test_duplicate_axis_name_rejected():
    with pytest.raises(SchemaError, match="Duplicate"):
        Schema(axes=(discrete_axis("x", {1}, 1), discrete_axis("x", {2}, 2)))


def test_unknown_dependency_rejected():
    a = Axis("a", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"ghost"})
    with pytest.raises(SchemaError, match="unknown"):
        Schema(axes=(a,))


def test_cycle_detection():
    a = Axis("a", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"b"})
    b = Axis("b", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"a"})
    with pytest.raises(SchemaError, match="cycle"):
        Schema(axes=(a, b))


def test_independent_axes_preserve_declaration_order():
    x = discrete_axis("x", {1}, 1)
    y = discrete_axis("y", {2}, 2)
    z = discrete_axis("z", {3}, 3)
    schema = Schema(axes=(x, y, z))
    assert [a.name for a in schema.ordered_axes()] == ["x", "y", "z"]


def test_axis_names_property():
    schema = Schema(axes=(discrete_axis("a", {1}, 1), discrete_axis("b", {2}, 2)))
    assert schema.axis_names == frozenset({"a", "b"})
