############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The stratum-driven feasibility fast path (engine hone Task 3.3).

``first_feasible_backend`` used to run a FULL resolve per pool member — 3 resolves and
~37 ms to reject a float MatMul, on a question answerable from the node's datatypes. It now
offers each member to a fold-independent pre-check first, and falls back to the full trial
whenever that cannot decide soundly.

Two things must hold, and the first matters more than the speed:

1. **Precedence is unchanged.** ``first_feasible_backend`` defines selection order for
   ``SpecializeKernels``; returning a different backend would silently change what hardware
   gets built. It must return the same NAME for every context, not merely the same
   feasible/infeasible verdict.
2. **The fast path is actually taken** — otherwise this is dead code that still costs.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

import finn.kernels.model.kernel as kernel_mod
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
    """The pre-3.3 implementation: a full resolve per pool member, no early exit. Kept here
    as the ORACLE — the fast path is only correct insofar as it agrees with this."""
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
    """THE gate: the fast path must select the same backend the full-resolve loop does, for
    every datatype/device/shape combination — not just the same yes/no."""
    ctx = _ctx(idt, wdt, odt, shape[0], shape[1], fpgapart)
    assert mvau.first_feasible_backend(ctx) == _reference_first_feasible(mvau, ctx)


@pytest.mark.parametrize("fpgapart", [VERSAL, SEVEN_SERIES])
def test_precedence_matches_with_thresholds(mvau, fpgapart):
    """Thresholds change which backends are eligible (the DSP cores cannot consume them),
    so the fused case is its own precedence risk."""
    ctx = _ctx(D["INT4"], D["INT4"], D["UINT4"], fpgapart=fpgapart, n_thresholds=15)
    assert mvau.first_feasible_backend(ctx) == _reference_first_feasible(mvau, ctx)


def test_aggressive_clock_precedence(mvau):
    """A clock tight enough to trip SEGMENTLEN feasibility — a stratum-2 rule, so this
    context MUST fall back to the full resolve and still agree."""
    ctx = _ctx(D["INT4"], D["INT4"], D["INT16"], clk_ns=0.5)
    assert mvau.first_feasible_backend(ctx) == _reference_first_feasible(mvau, ctx)


def test_float_rejection_does_zero_full_resolves(mvau, monkeypatch):
    """The headline F2 case. Rejecting a float MatMul must not resolve at all — every rule
    that rejects it reads only datatypes."""
    calls = []
    monkeypatch.setattr(
        kernel_mod, "resolve", lambda *a, **k: (calls.append(1), real_resolve(*a, **k))[1]
    )
    ctx = _ctx(D["FLOAT32"], D["FLOAT32"], D["FLOAT32"])
    assert mvau.has_feasible_point(ctx) is False
    assert calls == [], f"float rejection should do 0 full resolves, did {len(calls)}"


def test_accepted_node_resolves_at_most_once(mvau, monkeypatch):
    """An accepted node still needs one real resolve (24 of 35 MVAU rules are fold-dependent
    — divisibility genuinely depends on the fold). It must not need more than one."""
    calls = []
    monkeypatch.setattr(
        kernel_mod, "resolve", lambda *a, **k: (calls.append(1), real_resolve(*a, **k))[1]
    )
    ctx = _ctx(D["INT4"], D["INT4"], D["INT16"])
    assert mvau.has_feasible_point(ctx) is True
    assert len(calls) <= 1, f"accepted node did {len(calls)} full resolves"


def test_typo_class_bug_still_propagates_through_the_fast_path():
    """The early path swallows exceptions to mean "cannot decide" — it must not thereby
    swallow a kernel BUG. Falling through to the full resolve is what preserves INV5: a
    typo-class error still escapes, just from the trial rather than the pre-check."""
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
