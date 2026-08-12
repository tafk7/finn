############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""T0.1 (F11) — feasibility trials narrow their ``except`` to the legitimate
"can't-resolve-for-this-probe" signals (``ValueError``/``KeyError``/``AbsentAxisError``).

A backend predicate that raises a TYPO-class bug (``AttributeError``) must PROPAGATE out of
``has_feasible_point`` rather than being silently read as "this backend is infeasible" — the
silent-skip class (INV5) the kernel migration exists to eliminate. The float-rejected-for-the-
right-reason case (a predicate that legitimately returns an Illegal reason) stays feasible-false
without raising, so the narrowing does not over-tighten.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.predicate import predicate
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.kernel import DataflowKernel, InterfaceSchema
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL


def _ifaces():
    return (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
        InterfaceSchema("weights", Direction.IN, block=[FULL, FULL]),
        InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
    )


def _ctx():
    return Context(
        shapes={"inp": (1, 8), "weights": (8, 4), "out": (1, 4)},
        datatypes={"inp": DataType["INT8"], "weights": DataType["INT8"], "out": DataType["INT32"]},
        initializers={"weights": np.ones((8, 4), dtype=np.float32)},
        fpgapart="xcvc1902-vsva2197-2MP-e-S",
    )


def _kernel_with_predicate(pred):
    backend = Backend(
        name="core",
        predicates=(pred,),
        ports=ports_from(stream={"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]}),
    )
    return DataflowKernel(name="MVU", interfaces=_ifaces(), pool=(backend,))


def test_typo_class_bug_propagates_not_silently_infeasible():
    """A backend predicate raising ``AttributeError`` (a typo-class kernel bug) must propagate
    out of ``has_feasible_point`` — NOT be swallowed and reported as "no feasible point"."""

    @predicate("buggy predicate with a typo-class AttributeError")
    def _buggy(p, ctx):
        # A genuine typo-class bug: calling a method that does not exist on Context. (Reading
        # an unknown name off the Point ``p`` would raise AbsentAxisError — a legitimate,
        # narrowed signal — so we hit Context, whose __getattr__ raises a real AttributeError.)
        return ctx.this_method_does_not_exist()

    kernel = _kernel_with_predicate(_buggy)
    with pytest.raises(AttributeError):
        kernel.has_feasible_point(_ctx())


def test_legitimate_rejection_reads_as_infeasible_without_raising():
    """A predicate that legitimately returns a reason string (an Illegal, not an exception)
    yields ``has_feasible_point() is False`` — the narrowing does not over-tighten."""

    @predicate("always-illegal predicate (a real, non-exceptional rejection)")
    def _reject(p, ctx):
        return "rejected for a legitimate reason"

    kernel = _kernel_with_predicate(_reject)
    assert kernel.has_feasible_point(_ctx()) is False
