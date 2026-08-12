############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``first_feasible_backend`` precedence, across datatypes / devices / shapes.

Formerly "the stratum-driven feasibility fast path". That fast path — ``_early_verdict``,
which pre-checked each member against its fold-independent rules to skip a full resolve —
is DELETED. It was measured at −1% on both realistic paths (net negative on two of three),
and the one case it won, rejecting a float MatMul, is decided at stratum 0 from Context
alone, which needs no selection-root concept. Per-realization construction also makes each
trial cheaper on its own, so the pre-check bought less than it cost to reason about.

What survives is the part that always mattered more than the speed: **precedence is
unchanged.** ``first_feasible_backend`` defines selection order for ``SpecializeKernels``,
so returning a different backend would silently change what hardware gets built. It must
return the same NAME for every context, not merely the same feasible/infeasible verdict —
and ``_reference_first_feasible`` below is the independent implementation it is checked
against.

The two tests that asserted the fast path was TAKEN (zero resolves on a float rejection,
at most one on an accepted node) are gone with the mechanism they pinned. Rewriting them to
assert the new call counts would be inventing a requirement: nothing depends on how many
times resolution runs, only on what it decides.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.point import AbsentAxisError, Point
from finn.kernels.engine.predicate import predicate
from finn.kernels.engine.resolve import resolve as real_resolve
from finn.kernels.model.backend import BACKEND_AXIS

VERSAL = "xcvc1902-vsvd1760-2MP-e-S"
SEVEN_SERIES = "xc7z020clg400-1"
ULTRASCALE = "xcku3p-ffva676-1-e"


def _ctx(idt, wdt, odt, mw=8, mh=8, fpgapart=VERSAL, n_thresholds=None, clk_ns=5.0):
    shapes = {"inp": (1, mw), "weights": (mw, mh), "out": (1, mh)}
    dts = {"inp": idt, "weights": wdt, "out": odt}
    rng = np.random.default_rng(0)
    inits = {"weights": rng.integers(-7, 8, (mw, mh)).astype(np.float32)}
    if n_thresholds is not None:
        shapes["thresholds"] = (mh, n_thresholds)
        dts["thresholds"] = DataType["INT16"]
        inits["thresholds"] = np.sort(
            rng.integers(-100, 100, (mh, n_thresholds)).astype(np.float32), axis=1
        )
    return Context(
        shapes=shapes, datatypes=dts, initializers=inits, fpgapart=fpgapart, clk_ns=clk_ns
    )


def _reference_first_feasible(kernel, context):
    """The MERGED-space implementation: a full resolve per pool member against
    ``kernel.compile()``. Kept as the ORACLE — per-realization resolution is only correct
    insofar as it agrees with the merge it replaces, which is precisely T7's claim."""
    schema = kernel.compile()
    for impl in kernel.pool:
        try:
            result = real_resolve(schema, context, {BACKEND_AXIS: impl.name})
        except (ValueError, KeyError, AbsentAxisError):
            continue
        if isinstance(result, Point):
            return impl.name
    return None


D = DataType
_DTYPES = [
    (D["INT4"], D["INT4"], D["INT16"]),
    (D["INT8"], D["INT8"], D["INT32"]),
    (D["UINT8"], D["INT8"], D["INT32"]),
    (D["INT2"], D["INT2"], D["INT8"]),
    (D["INT16"], D["INT8"], D["INT32"]),
    (D["FLOAT32"], D["FLOAT32"], D["FLOAT32"]),
    (D["BINARY"], D["BINARY"], D["INT8"]),
    (D["BIPOLAR"], D["BIPOLAR"], D["INT8"]),
]


@pytest.fixture(scope="module")
def mvau():
    from finn.kernels.compute.mvau.op import mvau_kernel

    return mvau_kernel()


@pytest.mark.parametrize("fpgapart", [VERSAL, SEVEN_SERIES, ULTRASCALE])
@pytest.mark.parametrize("idt,wdt,odt", _DTYPES, ids=[str(d[1]) + "w" for d in _DTYPES])
@pytest.mark.parametrize("shape", [(8, 8), (32, 32), (1, 1), (64, 16)])
def test_precedence_matches_full_resolve(mvau, fpgapart, idt, wdt, odt, shape):
    """THE gate: per-realization resolution must select the same backend the merged
    full-resolve loop does, for every datatype/device/shape combination — not just the same
    yes/no."""
    ctx = _ctx(idt, wdt, odt, shape[0], shape[1], fpgapart)
    assert mvau.first_feasible_backend(ctx) == _reference_first_feasible(mvau, ctx)


@pytest.mark.parametrize("fpgapart", [VERSAL, SEVEN_SERIES])
def test_precedence_matches_with_thresholds(mvau, fpgapart):
    """Thresholds change which backends are eligible (the DSP cores cannot consume them),
    so the fused case is its own precedence risk."""
    ctx = _ctx(D["INT4"], D["INT4"], D["UINT4"], fpgapart=fpgapart, n_thresholds=15)
    assert mvau.first_feasible_backend(ctx) == _reference_first_feasible(mvau, ctx)


def test_aggressive_clock_precedence(mvau):
    """A clock tight enough to trip SEGMENTLEN feasibility — a fold-dependent rule, so the
    verdict genuinely needs a resolved configuration rather than a datatype comparison."""
    ctx = _ctx(D["INT4"], D["INT4"], D["INT16"], clk_ns=0.5)
    assert mvau.first_feasible_backend(ctx) == _reference_first_feasible(mvau, ctx)


def test_typo_class_bug_still_propagates():
    """INV5: `first_feasible_backend` catches ValueError/KeyError to mean "this member is not
    feasible HERE", and must not thereby swallow a kernel BUG. An AttributeError from a typo
    in a rule is not a feasibility signal and still escapes.

    Newly load-bearing: with `_early_verdict` gone the trial loop is the ONLY thing standing
    between a buggy rule and a silent skip, so what used to be a backstop is now the whole
    guarantee."""
    from finn.kernels.model.backend import Backend, ports_from
    from finn.kernels.model.kernel import InterfaceSchema, _LegacyKernel
    from finn.kernels.model.ports import Direction
    from finn.kernels.model.tiling import FULL

    @predicate("buggy: typo-class AttributeError, reads no point key (so stratum 0)")
    def _buggy(p, ctx):
        return ctx.this_method_does_not_exist()

    kernel = _LegacyKernel(
        name="MVU",
        interfaces=(
            InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
            InterfaceSchema("weights", Direction.IN, block=[FULL, FULL]),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
        ),
        pool=(Backend(name="b", predicates=(_buggy,), ports=ports_from()),),
    )
    ctx = Context(
        shapes={"inp": (1, 8), "weights": (8, 8), "out": (1, 8)},
        datatypes={
            "inp": D["INT4"],
            "weights": D["INT4"],
            "out": D["INT16"],
        },
        initializers={"weights": np.zeros((8, 8), dtype=np.float32)},
        fpgapart=VERSAL,
    )
    with pytest.raises(AttributeError):
        kernel.has_feasible_point(ctx)
