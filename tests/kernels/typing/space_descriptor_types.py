# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Static assertions for the class-centered descriptor API.

A positive typing fixture: strict mypy must accept this module unchanged. The
claims below are the ones a runtime test cannot make. ``pipeline.fixed`` and
``pipeline.implementation`` both *work* at runtime whatever the annotations say,
so the thing worth pinning is that a contributor's editor and type checker see
the authored child class and the bound view rather than ``Any`` or the
declaration — which is exactly what a descriptor overload silently loses.
"""

from __future__ import annotations

from typing import assert_type

from finn.core.space import (
    BoundView,
    ChoiceView,
    Const,
    Decision,
    Derived,
    Param,
    QueryResult,
    Space,
    SpaceModel,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    View,
    ViewAssessment,
    compile_space,
    constraint,
    derived,
    view,
)

RESULT = ValueKey("result", int)


class FixedImplementation(Space):
    size = Param(int)
    lanes = Decision(int, values=(1, 2, 4))
    minimum = Const(1)

    @derived
    def result(self) -> int:
        return self.size // self.lanes

    physical = View(result)
    exports = {RESULT: result}


class SmallImplementation(Space):
    size = Param(int)

    @derived
    def result(self) -> int:
        return self.size

    exports = {RESULT: result}


class Pipeline(Space):
    size = Param(int)
    fixed = Subspace(FixedImplementation, size=size)

    @derived
    def cycles(self) -> int:
        assert_type(self.fixed, FixedImplementation)
        assert_type(self.fixed.result, int)
        return self.fixed.result

    @constraint
    def supported(self) -> bool:
        return self.size > 0

    @view(constraints=(supported,))
    def output(self) -> int:
        return self.cycles

    implementation = SubspaceChoice(
        {
            "fast": Subspace(FixedImplementation, size=size),
            "small": Subspace(SmallImplementation, size=size),
        },
        exports=(RESULT,),
    )


# Class access is the declaration; the Subspace keeps its concrete child type.
assert_type(Pipeline.fixed, Subspace[FixedImplementation])
assert_type(Pipeline.implementation, SubspaceChoice)
assert_type(Pipeline.size, Param[int])
assert_type(FixedImplementation.minimum, Const[int])
assert_type(FixedImplementation.result, Derived[int])
assert_type(FixedImplementation.physical, View[int])
assert_type(Pipeline.implementation.ref(RESULT), ValueRef[int])

# The compiler service preserves the authored root class through the model.
model = compile_space(Pipeline)
assert_type(model, SpaceModel[Pipeline])
assert_type(model.bind({Pipeline.size: 8}), Pipeline)

# So does the one-shot entry, and so does an immutable successor.
pipeline = Pipeline({Pipeline.size: 8})
assert_type(pipeline, Pipeline)

# Instance access binds the exact use site.
assert_type(pipeline.fixed, FixedImplementation)
assert_type(pipeline.implementation, ChoiceView)

# And the bound view's own surface stays typed.
assert_type(pipeline.implementation.alternatives, tuple[str, ...])
assert_type(pipeline.implementation.select("fast"), ChoiceView)
assert_type(pipeline.implementation.alternative("fast"), Space)

# A declared value read through an occurrence has its declared type.
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
assert_type(pipeline.fixed.query(FixedImplementation.result), QueryResult[int])
