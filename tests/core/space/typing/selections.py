# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Selection reads and replay are typed: a captured value is its decision's type."""

from typing import assert_type

from finn.core.space import (
    ConfigurationResult,
    Decision,
    Param,
    Selection,
    Space,
    design_space,
    selections,
)


class Example(Space):
    extent: int = Param()
    factor: int = Decision(values=(1, 2))
    style: str = Decision(values=("auto", "block"))


base = design_space(Example(extent=4))
selected = selections.capture(base)
assert_type(selections.capture(base), Selection)
assert_type(selected.value(Example.factor), int)
assert_type(selections.restore(base, selected), ConfigurationResult[Example])
