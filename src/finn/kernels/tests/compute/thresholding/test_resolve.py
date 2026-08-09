############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding resolve — the model-stressing op.

Proves "op-level shared is per-op": the RTL bundle contributes impl-local axes
(depth_trigger_*, uniform_thres, deep_pipeline) ABSENT under the HLS sibling, while both
share the SAME integer dtype envelope (no fabricated per-bundle dtype gate). PE folds the
channel dim; numSteps is derived from the threshold tensor; data-dependent predicates
reject unsorted (RTL) and negative-under-unsigned thresholds.
"""

from dataclasses import replace

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.point import AbsentAxisError, Illegal, Point
from finn.kernels.engine.context import Context
from finn.kernels.engine.resolve import resolve
from finn.kernels.compute.thresholding import (
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    thresholding_pool,
    thresholding_space,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _language_of(point, pool=None):
    by_name = {b.name: b for b in (pool if pool is not None else thresholding_pool())}
    return by_name[point["backend"]].language


@pytest.fixture
def schema():
    return thresholding_space()


def sorted_thresholds(channels=8, steps=7, signed=False):
    rng = np.random.RandomState(0)
    lo = -50 if signed else 0
    return np.sort(rng.randint(lo, 50, size=(channels, steps)).astype(np.float32), axis=-1)


def make_context(channels=8, steps=7, idt="UINT8", tdt="UINT8", odt="UINT3", thresholds=None):
    if thresholds is None:
        thresholds = sorted_thresholds(channels, steps)
    return Context(
        shapes={"thresholds": (channels, steps), "inp": (1, channels), "out": (1, channels)},
        datatypes={"thresholds": DataType[tdt], "inp": DataType[idt], "out": DataType[odt]},
        initializers={"thresholds": thresholds},
        fpgapart=VERSAL, clk_ns=5.0,
    )


def base_assignment(**overrides):
    a = {"backend": THRESHOLDING_HLS, "PE": 2}
    a.update(overrides)
    return a


def test_pool_is_hls_and_rtl():
    assert [b.name for b in thresholding_pool()] == [THRESHOLDING_HLS, THRESHOLDING_RTL]


def test_tmem_and_widths(schema):
    """The widths are now GENERATED per interface by the tiling engine from the declared
    stream fold (``stream_width.<iface>``), replacing the hand-written singular
    ``instream_width``/``outstream_width`` pair. The VALUES are unchanged: the generated
    ``elems x dtype.bitwidth()`` equals the old ``dtype.bitwidth() x PE`` because these
    ports declare no derived_dtype, so the dtype is the graph tensor dtype either way."""
    from finn.kernels.model.tiling import stream_width_key

    r = resolve(schema, make_context(channels=8), base_assignment(PE=2))
    assert isinstance(r, Point)
    assert r.TMEM == 8 // 2
    assert r[stream_width_key("inp")] == DataType["UINT8"].bitwidth() * 2  # i_bits * PE
    assert r[stream_width_key("out")] == DataType["UINT3"].bitwidth() * 2  # o_bits * PE


def test_threshold_dtype_is_the_declared_dtype_not_the_narrowed_one(schema):
    """PARITY GATE: thresholdDataType is the DECLARED graph dtype, matching baseline FINN
    (``hls/thresholding_hls.py:210`` reads ``get_input_datatype(1)``).

    The storage owner still publishes a value-NARROWED ParamDatatype, and this test pins that
    the consumer deliberately does NOT read it — see ``_threshold_datatype``'s TODO. The
    narrowing changes the emitted ``ThresholdsActivation<>`` width, which is a real divergence
    from FINN that nothing has functionally validated. Parity outranks the optimization until
    it does; re-enabling means flipping this test back with an rtlsim case beside it."""
    from finn.kernels.model.param_names import param_datatype_key

    thr_key = param_datatype_key("thresholds")
    # UINT8-typed thresholds valued 0..49: the owner narrows to UINT6, we must still say UINT8.
    r = resolve(schema, make_context(channels=8, steps=7, tdt="UINT8"), base_assignment(PE=2))
    assert isinstance(r, Point)
    pd = r[thr_key]
    assert pd is not None and pd.values_visible is True
    assert pd.dtype == DataType["UINT6"], "the owner still publishes the narrowed authority"
    assert r["thresholdDataType"] == DataType["UINT8"], (
        "the consumer must use the DECLARED dtype for FINN parity, not the narrowed one"
    )


def test_threshold_datatype_declares_no_dep_while_narrowing_is_off(schema):
    """thresholdDataType is Context-only for now, so it declares no dep and needs no ordering.

    It USED to dep on ``parameters.thresholds.datatype`` and be ordered after it. Re-enabling
    the narrowing restores both the read and that dep — this test is the reminder that the two
    move together, since a restored read with a dropped dep is exactly the undeclared-read
    class ``test_deps_audit`` exists to catch."""
    tdt = next(d for d in schema.ordered_derived() if d.name == "thresholdDataType")
    assert tdt.deps == frozenset()
    assert tdt.optional_deps == frozenset()


def test_pe_must_divide_channels(schema):
    r = resolve(schema, make_context(channels=8), base_assignment(PE=3))  # 3 ∤ 8
    assert isinstance(r, Illegal)
    assert any("PE" in reason for reason in r.reasons)


def test_num_steps_is_the_threshold_step_extent(schema):
    """``numSteps`` IS the tensor's step dim, so it cannot disagree with it.

    This replaces a test asserting that pinning an INCONSISTENT ``numSteps`` was rejected.
    That was the weaker property: it required a predicate to notice a duplicate had drifted.
    Now there is no duplicate to drift — the value is derived, so agreement is structural and
    the rule that policed it is gone."""
    for steps in (7, 15):
        r = resolve(schema, make_context(steps=steps), base_assignment())
        assert isinstance(r, Point)
        assert r["numSteps"] == steps
    assert "numSteps" not in schema.axis_names, "a derivation, not a dial"


def test_a_non_2d_threshold_tensor_reports_its_rank(schema):
    """The RANK requirement survives the numSteps dissolution — and now reports LEGIBLY.

    It raises rather than returning ``Illegal`` because the geometry deriveds index this
    shape positionally, and deriveds run before predicates: a rule could never fire first. So
    the check moved to the shape read. The exception type matters — ``ValueError`` is in the
    set ``Kernel.first_feasible_backend`` treats as "not resolvable for this context", whereas
    the bare ``IndexError`` this used to raise would escape as a kernel bug (INV5) on a node
    that is merely ineligible."""
    ctx = replace(make_context(), shapes={**make_context().shapes, "thresholds": (8,)})
    with pytest.raises(ValueError, match="must be 2D"):
        resolve(schema, ctx, base_assignment())


# --- RTL-local axes disjoint from HLS --------------------------------------


def test_rtl_local_axes_absent_under_hls(schema):
    r = resolve(schema, make_context(), base_assignment(backend=THRESHOLDING_HLS))
    assert isinstance(r, Point)
    for name in ("depth_trigger_bram", "depth_trigger_uram", "uniform_thres", "deep_pipeline"):
        assert name not in r, f"{name} must NOT exist under the HLS bundle"
        with pytest.raises(AbsentAxisError):
            _ = r[name]


def test_rtl_local_axes_present_under_rtl(schema):
    r = resolve(
        schema, make_context(),
        base_assignment(backend=THRESHOLDING_RTL, depth_trigger_bram=1024, uniform_thres=1),
    )
    assert isinstance(r, Point)
    assert r.depth_trigger_bram == 1024
    assert r.uniform_thres == 1
    assert r.deep_pipeline == 1  # default
    assert _language_of(r) == "rtl"


def test_assigning_rtl_axis_under_hls_is_illegal(schema):
    r = resolve(
        schema, make_context(),
        base_assignment(backend=THRESHOLDING_HLS, depth_trigger_bram=1024),
    )
    assert isinstance(r, Illegal)
    assert any("depth_trigger_bram" in reason for reason in r.reasons)


# --- no fabricated per-bundle dtype gate -----------------------------------


def test_same_integer_dtype_resolves_under_both_bundles(schema):
    ctx = make_context(idt="INT8", tdt="INT8", odt="INT4", thresholds=sorted_thresholds(signed=True))
    assert isinstance(resolve(schema, ctx, base_assignment(backend=THRESHOLDING_HLS)), Point)
    assert isinstance(resolve(schema, ctx, base_assignment(backend=THRESHOLDING_RTL)), Point)


# --- data-dependent predicates ---------------------------------------------


def test_rtl_rejects_unsorted_thresholds(schema):
    thr = sorted_thresholds()[:, ::-1].copy()  # descending
    r = resolve(schema, make_context(thresholds=thr), base_assignment(backend=THRESHOLDING_RTL))
    assert isinstance(r, Illegal)
    assert any("sorted" in reason for reason in r.reasons)


def test_hls_allows_unsorted_thresholds(schema):
    thr = sorted_thresholds()[:, ::-1].copy()
    r = resolve(schema, make_context(thresholds=thr), base_assignment(backend=THRESHOLDING_HLS))
    assert isinstance(r, Point)


def test_unsigned_input_requires_nonneg_thresholds(schema):
    thr = sorted_thresholds(signed=True)  # contains negatives
    r = resolve(schema, make_context(idt="UINT8", thresholds=thr), base_assignment(backend=THRESHOLDING_HLS))
    assert isinstance(r, Illegal)
    assert any("thresholds >= 0" in reason or "non-negative" in reason.lower() for reason in r.reasons)


# --- composability ---------------------------------------------------------


def test_third_implementation_composes_additively():
    from dataclasses import replace
    from finn.kernels.model.backend import Backend
    from finn.kernels.compute.thresholding import thresholding_kernel, thresholding_pool

    # A 4th backend adds to the pool with zero edits. Compose through the Kernel (the full
    # space, so the parameters pool folds in and thresholdDataType's cross-pool dep resolves) —
    # the compute pool alone is a fragment, not a schema.
    #
    # The stub declares the shared stream like any real backend: PE is a GENERATED fold dial
    # owned by the backends that declare a fold, not an op-level axis every member carries.
    # (A stub declaring no stream would legitimately have no PE, and pinning it would be the
    # "assigned but absent" error.) MVAU's equivalent stub does the same.
    from finn.kernels.model.backend import ports_from
    from finn.kernels.compute.thresholding.names import COMPUTE_STREAM

    stub = Backend(
        name="thresholding_stub",
        language="stub",
        sources=("stub.sv",),
        ports=ports_from(stream=COMPUTE_STREAM),
    )
    pool3 = thresholding_pool() + (stub,)
    schema3 = replace(thresholding_kernel(), pool=pool3).compile()
    r = resolve(schema3, make_context(), base_assignment(backend="thresholding_stub"))
    assert isinstance(r, Point)
    assert _language_of(r, pool3) == "stub"
    assert r.sources == ("stub.sv",)
