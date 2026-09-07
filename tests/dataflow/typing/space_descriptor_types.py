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

from typing_extensions import assert_type

from finn.dataflow.space import (
    Decision,
    Input,
    Problem,
    Space,
    Subspace,
    SubspaceChoice,
    ChoiceView,
    compile_space_model,
    derived,
)
from finn.dataflow.space.compiler import SpaceModel


class FixedImplementation(Space):
    size = Input(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(int, size=size, lanes=lanes)
    def result(*, size: int, lanes: int) -> int:
        return size // lanes

    exports = (result,)


class SmallImplementation(Space):
    size = Input(int)

    @derived(int, size=size)
    def result(*, size: int) -> int:
        return size

    exports = (result,)


class Pipeline(Space):
    size = Problem(int)
    fixed = Subspace(FixedImplementation, size=size)
    implementation = SubspaceChoice(
        {
            "fast": Subspace(FixedImplementation, size=size),
            "small": Subspace(SmallImplementation, size=size),
        },
        outputs=("result",),
    )


# Class access is the declaration; the Subspace keeps its concrete child type.
assert_type(Pipeline.fixed, Subspace[FixedImplementation])
assert_type(Pipeline.implementation, SubspaceChoice)

# The compiler service preserves the authored root class through the model.
model = compile_space_model(Pipeline, "root", problem_namespace="problem.root")
assert_type(model, SpaceModel[Pipeline])
assert_type(model.start({Pipeline.size: 8}), Pipeline)

# So does the one-shot entry, and so does an immutable successor.
pipeline = Pipeline.start({Pipeline.size: 8})
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
assert_type(pipeline.fixed.assign(FixedImplementation.lanes, 2), FixedImplementation)
