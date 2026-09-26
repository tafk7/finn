# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Static assertions for the declarative node API.

A positive typing fixture: strict mypy must accept this module unchanged. The
claims below are the ones a runtime test cannot make. A node declaration
``FixedImplementation(size=size)`` and the configuration ``pipeline.fixed``
both *work* at runtime whatever the annotations say, so the thing worth
pinning is that a contributor's editor and type checker see the authored
family, references typed as the values they stand for, and the bound view,
rather than ``Any`` or an internal declaration type.
"""

from __future__ import annotations

from typing_extensions import assert_type

from finn.core.space import (
    BoundView,
    ConfigurationResult,
    Const,
    Decision,
    Derived,
    Param,
    QueryResult,
    Space,
    View,
    ViewAssessment,
    ViewKey,
    configure,
    constraint,
    derived,
    inspection,
    selected,
    view,
)
from finn.core.space.inspection import ChoiceInfo

RESULT = ViewKey("result", int)


class FixedImplementation(Space):
    size: Param[int] = Param(int)
    lanes = Decision(int, values=(1, 2, 4))
    minimum = Const(1)

    @derived
    def result(self) -> int:
        return self.size // self.lanes

    physical = View(result)
    exports = {RESULT: physical}


class SmallImplementation(Space):
    size: Param[int] = Param(int)

    @derived
    def result(self) -> int:
        return self.size

    physical = View(result)
    exports = {RESULT: physical}


class Pipeline(Space):
    size: Param[int] = Param(int)
    # Calling a family declares a node, typed as the family.
    fixed = FixedImplementation(size=size)
    assert_type(fixed, FixedImplementation)
    # A reference in a class body is typed as the value it stands for.
    assert_type(fixed.result, int)
    assert_type(fixed.physical, BoundView[int])

    @derived
    def cycles(self) -> int:
        assert_type(self.fixed, FixedImplementation)
        assert_type(self.fixed.result, int)
        return self.fixed.result

    @constraint
    def supported(self) -> bool:
        return self.size > 0

    @view(requires=(supported,))
    def output(self) -> int:
        return self.cycles

    # The structural choice: a Decision over nodes, typed as its candidates.
    implementation = Decision[FixedImplementation | SmallImplementation](
        values={
            "fast": FixedImplementation(size=size),
            "small": SmallImplementation(size=size),
        }
    )
    assert_type(implementation, FixedImplementation | SmallImplementation)
    # A member every candidate has is read through the choice.
    assert_type(implementation.result, int)
    chosen = View(implementation.physical)
    case = selected(implementation)

    @derived
    def chosen_case(self) -> str:
        assert_type(self.case, str)
        return self.case


# Class access is the schema key; a node keeps its concrete family.
assert_type(Pipeline.fixed, FixedImplementation)
assert_type(Pipeline.fixed.result, int)
assert_type(Pipeline.implementation, FixedImplementation | SmallImplementation)
assert_type(Pipeline.size, Param[int])
assert_type(Pipeline.chosen, View[int])
assert_type(FixedImplementation.minimum, Const[int])
assert_type(FixedImplementation.result, Derived[int])
assert_type(FixedImplementation.physical, View[int])
assert_type(FixedImplementation.lanes, Decision[int])

# The one compile step preserves the authored root family.
assert_type(configure(Pipeline(size=8)), Pipeline)


def check(pipeline: Pipeline) -> None:
    # Configuration access binds the exact use site.
    assert_type(pipeline.fixed, FixedImplementation)
    assert_type(pipeline.implementation, FixedImplementation | SmallImplementation)
    assert_type(pipeline.case, str)

    # The replacements of the bound choice view stay typed.
    assert_type(pipeline.with_choices(implementation="fast"), Pipeline)
    assert_type(pipeline.with_choices({Pipeline.implementation: "small"}), Pipeline)
    assert_type(pipeline.try_with_choices(implementation="fast"), ConfigurationResult[Pipeline])
    assert_type(inspection.candidate(pipeline, Pipeline.implementation, "fast"), Space | None)
    assert_type(inspection.choices(pipeline), tuple[ChoiceInfo, ...])

    # A declared value read through a configuration has its declared type.
    assert_type(pipeline.fixed.result, int)
    assert_type(pipeline.fixed.with_choices(lanes=2), FixedImplementation)
    assert_type(pipeline.fixed.physical, BoundView[int])
    assert_type(pipeline.fixed.physical(), int)
    assert_type(pipeline.fixed.physical.inspect(), ViewAssessment[int])
    assert_type(pipeline.fixed.physical.query(), QueryResult[int])
    assert_type(pipeline.fixed.inspect(FixedImplementation.physical), ViewAssessment[int])
    assert_type(pipeline.fixed.view(FixedImplementation.physical), BoundView[int])
    assert_type(pipeline.fixed.view(FixedImplementation.physical)(), int)
    assert_type(pipeline.fixed.field(FixedImplementation.result).get(), int)
    assert_type(pipeline.fixed.field(FixedImplementation.result).query(), QueryResult[int])
    assert_type(pipeline.fixed.field(FixedImplementation.lanes).get(), int)
    assert_type(pipeline.fixed.field(FixedImplementation.lanes).query(), QueryResult[int])
    assert_type(pipeline.output(), int)
    assert_type(pipeline.chosen(), int)
    assert_type(pipeline.fixed.query(FixedImplementation.result), QueryResult[int])
    assert_type(pipeline.query(Pipeline.fixed.result), QueryResult[int])
