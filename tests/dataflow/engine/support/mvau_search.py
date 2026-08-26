# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tiny search-style consumer showing that policy remains outside the engine."""

from __future__ import annotations

from itertools import product

from dataflow.engine.support.mvau_design_space import (
    ActivationMode,
    Paths,
    build_mvau_design_space,
    example_problem,
)

from finn.dataflow._engine import Decided, DesignPoint, Engine, ProposalAdoptionMode


def first_feasible_design() -> DesignPoint | None:
    engine = Engine()
    space = engine.validate(build_mvau_design_space())
    root = engine.start(space, example_problem().as_mapping())
    pe = engine.enumerate_candidates(root, Paths.PE)
    simd = engine.enumerate_candidates(root, Paths.SIMD)
    if not isinstance(pe, Decided) or not isinstance(simd, Decided):
        return None

    for pe_value, simd_value in product(reversed(pe.value), reversed(simd.value)):
        explicit = engine.commit_assignments(
            root,
            {
                Paths.PE: pe_value,
                Paths.SIMD: simd_value,
                Paths.ACTIVATION_MODE: ActivationMode.ACCUMULATORS,
            },
        )
        proposed = engine.adopt_profile_proposals(
            explicit.point,
            "implementation",
            ProposalAdoptionMode.TO_FIXPOINT,
        )
        if (
            engine.evaluate_constraint_set(proposed.point, "all").verdict is True
            and engine.check_readiness(proposed.point, "implementation").ready is True
        ):
            return proposed.point
    return None


if __name__ == "__main__":
    point = first_feasible_design()
    if point is None:
        raise SystemExit("no feasible design found")
    for path, value in point.assignments.items():
        print(f"{path}: {value}")
