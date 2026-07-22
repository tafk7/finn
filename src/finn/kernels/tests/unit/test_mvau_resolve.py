############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The MVAU fixture proves the six acceptance criteria (RESOLVE-CORE-HANDOFF §4).

1. A guarded axis (ram_style) is absent when its guard is false; reading errors.
2. The device prunes the implementation pool (DSP58-only impl -> Illegal on 7-series).
3. Forced-derived values (dsp_primitive, accDataType) are computed, never axes.
4. The combination predicate fires (ram_style=ultra & not versal => rw=1).
5. resolve returns a Point carrying derived on legal input; Illegal on illegal.
6. Enumerating a slice yields the dependent-space count, not the cartesian product.
"""

import itertools

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import AbsentAxisError, Context, Derived, Illegal, Point, resolve
from finn.kernels.ops.mvau import (
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    mvau_schema,
)
from finn.kernels.ops.parameters.names import (
    DECOUPLED as PARAM_DECOUPLED,
    EMBEDDED as PARAM_EMBEDDED,
    PUMPED_MEMORY as PARAM_PUMPED_MEMORY,
    RAM_STYLE as PARAM_RAM_STYLE,
    RUNTIME_WRITEABLE as PARAM_RUNTIME_WRITEABLE,
    TOPOLOGY as PARAM_TOPOLOGY,
)
from finn.util.basic import is_versal

SEVEN_SERIES = "xc7z020clg400-1"  # Zynq-7000, DSP48E1, not Versal
ULTRASCALE = "xcku040-ffva1156-2-e"  # Kintex UltraScale, DSP48E2, not Versal
VERSAL = "xcvc1902-vsva2197-2MP-e-S"  # Versal, DSP58


@pytest.fixture
def schema():
    return mvau_schema()


def narrow_weights(shape=(6, 8), wdt="INT4"):
    """Weights whose minimum is strictly above the dtype minimum, so narrow_weights
    derives to 1 (required for the RTL soft-vec core on a DSP48E1 part)."""
    lo = int(DataType[wdt].min()) + 1
    hi = int(DataType[wdt].max())
    rng = np.random.RandomState(0)
    return rng.randint(lo, hi + 1, size=shape).astype(np.float32)


def make_context(fpgapart=SEVEN_SERIES, weights=None, wdt="INT4", idt="INT4"):
    if weights is None:
        rng = np.random.RandomState(0)
        weights = rng.randint(-8, 8, size=(6, 8)).astype(np.float32)
    return Context(
        shapes={"weights": weights.shape, "inp": (1, weights.shape[0]), "out": (1, weights.shape[1])},
        datatypes={"weights": DataType[wdt], "inp": DataType[idt], "out": DataType["INT16"]},
        initializers={"weights": weights},
        fpgapart=fpgapart,
        clk_ns=5.0,
    )


def base_assignment(**overrides):
    """Build an MVAU assignment. The weight-delivery cluster moved to the composed
    ``parameters`` pool, so this shim translates the legacy delivery kwargs
    (``mem_mode``/``ram_style``/``runtime_writeable_weights``/``pumpedMemory``) into the
    namespaced ``parameters.*`` keys, keeping call sites terse. ``mem_mode`` maps to a
    storage topology: internal_embedded→embedded, internal_decoupled→decoupled."""
    a = {
        "implementation": MVAU_HLS,
        "PE": 4,
        "SIMD": 2,
        "mem_mode": "internal_decoupled",
        "noActivation": 1,
    }
    a.update(overrides)
    return _translate_parameters(a)


_MEM_MODE_TO_TOPOLOGY = {
    "internal_embedded": PARAM_EMBEDDED,
    "internal_decoupled": PARAM_DECOUPLED,
}
_DELIVERY_KEYS = {
    "ram_style": PARAM_RAM_STYLE,
    "runtime_writeable_weights": PARAM_RUNTIME_WRITEABLE,
    "pumpedMemory": PARAM_PUMPED_MEMORY,
}


def _translate_parameters(a):
    """Rewrite legacy delivery kwargs to composed ``parameters.*`` keys."""
    out = {}
    for k, v in a.items():
        if k == "mem_mode":
            out[PARAM_TOPOLOGY] = _MEM_MODE_TO_TOPOLOGY[v]
        elif k in _DELIVERY_KEYS:
            out[_DELIVERY_KEYS[k]] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# #1 — guarded axis absent, reading errors
# ---------------------------------------------------------------------------


def test_ram_style_absent_when_not_decoupled(schema):
    r = resolve(schema, make_context(), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r, Point)
    assert PARAM_RAM_STYLE not in r
    assert PARAM_RUNTIME_WRITEABLE not in r
    with pytest.raises(AbsentAxisError):
        _ = r[PARAM_RAM_STYLE]


def test_ram_style_present_when_decoupled(schema):
    r = resolve(schema, make_context(), base_assignment(mem_mode="internal_decoupled"))
    assert isinstance(r, Point)
    assert PARAM_RAM_STYLE in r
    assert r[PARAM_RAM_STYLE] == "auto"


# ---------------------------------------------------------------------------
# #2 — device prunes the implementation pool
# ---------------------------------------------------------------------------


def test_packed_impl_illegal_on_seven_series(schema):
    # mvau_dsp_packed (DSP58 INT8-packed core) physically requires DSP58.
    r = resolve(
        schema,
        make_context(SEVEN_SERIES, weights=narrow_weights()),
        base_assignment(implementation=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Illegal)
    assert any("DSP58" in reason for reason in r.reasons)


def test_other_impls_remain_on_seven_series(schema):
    # softvec (any DSP part, needs narrow weights on DSP48E1) and hls (no DSP
    # requirement) both resolve on 7-series.
    r_sv = resolve(
        schema,
        make_context(SEVEN_SERIES, weights=narrow_weights()),
        base_assignment(implementation=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r_sv, Point)
    r_hls = resolve(schema, make_context(SEVEN_SERIES), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r_hls, Point)


def test_packed_impl_legal_on_versal(schema):
    # INT8 weights (w=8, a=4) on DSP58 -> NUM_LANES=2 (<=3), so packed is feasible.
    # (INT4 would give 4 lanes and route to softvec -- see test_packed_num_lanes_gate.)
    r = resolve(
        schema,
        make_context(VERSAL, weights=narrow_weights(wdt="INT8"), wdt="INT8"),
        base_assignment(implementation=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Point)
    assert r.dsp_primitive == "DSP58"


def test_packed_impl_illegal_on_versal_with_wide_weights(schema):
    # The packed core is feasible on DSP58 only for w<=8 & a<=9. A DSP58 part is
    # necessary but NOT sufficient — wide (INT16) weights rule packed out even on
    # Versal, while softvec/hls remain. Proves the feasibility gate reads dtype,
    # not just device (mvu_vvu_axi.sv:313).
    ctx = make_context(VERSAL, weights=narrow_weights(shape=(6, 8), wdt="INT16"), wdt="INT16")
    r = resolve(
        schema,
        ctx,
        base_assignment(implementation=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Illegal)
    assert any("weight_width<=8" in reason for reason in r.reasons)
    # softvec is still feasible on the same wide-weight Versal context
    r2 = resolve(
        schema, ctx,
        base_assignment(implementation=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r2, Point)


# ---------------------------------------------------------------------------
# #3 — forced-derived values computed, never enumerated as axes
# ---------------------------------------------------------------------------


def test_dsp_primitive_forced_from_fpgapart(schema):
    # dsp_primitive is FORCED from the device, not chosen. softvec runs on any DSP
    # part; use narrow weights so the DSP48E1 RTL-feasibility gate is satisfied.
    for part, expected in [(SEVEN_SERIES, "DSP48E1"), (ULTRASCALE, "DSP48E2"), (VERSAL, "DSP58")]:
        r = resolve(
            schema,
            make_context(part, weights=narrow_weights()),
            base_assignment(implementation=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_embedded"),
        )
        assert isinstance(r, Point), r
        assert r.dsp_primitive == expected


def test_forced_and_derived_names_are_not_axes(schema):
    axis_names = schema.axis_names
    for name in ("dsp_primitive", "accDataType", "WMEM", "language", "SEGMENTLEN"):
        assert name not in axis_names, f"{name} must be Derived, not an Axis"


def test_acc_datatype_data_dependent_static(schema):
    # Small-valued static weights -> narrower accumulator than the worst case.
    small = np.ones((6, 8), dtype=np.float32)  # all +1
    r_small = resolve(schema, make_context(weights=small), base_assignment())
    big = np.full((6, 8), -8.0, dtype=np.float32)  # INT4 extreme everywhere
    r_big = resolve(schema, make_context(weights=big), base_assignment())
    assert isinstance(r_small, Point) and isinstance(r_big, Point)
    assert r_small.accDataType.bitwidth() < r_big.accDataType.bitwidth()


def test_acc_datatype_worst_case_when_runtime_writeable(schema):
    # runtime_writeable_weights forces worst-case bounds regardless of actual values.
    small = np.ones((6, 8), dtype=np.float32)
    r_static = resolve(schema, make_context(weights=small), base_assignment(noActivation=1))
    r_rtw = resolve(
        schema,
        make_context(weights=small),
        base_assignment(noActivation=1, runtime_writeable_weights=1),
    )
    assert isinstance(r_static, Point) and isinstance(r_rtw, Point)
    assert r_rtw.accDataType.bitwidth() >= r_static.accDataType.bitwidth()


# ---------------------------------------------------------------------------
# #4 — the combination predicate (config + device) fires
# ---------------------------------------------------------------------------


def test_uram_requires_runtime_writeable_on_ultrascale(schema):
    illegal = resolve(
        schema,
        make_context(ULTRASCALE),
        base_assignment(ram_style="ultra", runtime_writeable_weights=0),
    )
    assert isinstance(illegal, Illegal)
    assert any("URAM" in reason for reason in illegal.reasons)

    legal = resolve(
        schema,
        make_context(ULTRASCALE),
        base_assignment(ram_style="ultra", runtime_writeable_weights=1),
    )
    assert isinstance(legal, Point)


def test_uram_ok_on_versal_without_runtime_writeable(schema):
    # On Versal the URAM rule does not apply.
    r = resolve(
        schema,
        make_context(VERSAL),
        base_assignment(ram_style="ultra", runtime_writeable_weights=0),
    )
    assert isinstance(r, Point)


# ---------------------------------------------------------------------------
# #5 — legal -> Point with derived; illegal -> Illegal([reasons])
# ---------------------------------------------------------------------------


def test_legal_point_carries_derived(schema):
    r = resolve(schema, make_context(), base_assignment(PE=4, SIMD=2))
    assert isinstance(r, Point)
    assert r.WMEM == 6 * 8 // (4 * 2)
    assert r.language == "hls"
    assert r.outstream_width == r.outputDataType.bitwidth() * 4


def test_domain_violation_illegal(schema):
    # PE=5 does not divide MH=8 -> not in divisor domain.
    r = resolve(schema, make_context(), base_assignment(PE=5))
    assert isinstance(r, Illegal)
    assert "PE" in r.reasons[0]


def test_predicate_violation_illegal(schema):
    # pumpedCompute with SIMD=1 fires the config predicate.
    r = resolve(
        schema,
        make_context(),
        base_assignment(
            implementation=MVAU_DSP_SOFTVEC, resType="dsp", SIMD=1, pumpedCompute=1
        ),
    )
    assert isinstance(r, Illegal)
    assert any("pumpedCompute" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# #6 — guards compress: dependent-space count < naive cartesian product
# ---------------------------------------------------------------------------


def test_guards_compress_the_space(schema):
    # Enumerate a slice over the composed parameters pool: topology x the decoupled-
    # only ram cluster. Under a naive cartesian product every topology would carry
    # ram_style x rw; guards make the embedded branch collapse to a single point.
    ctx = make_context()
    topologies = ["internal_embedded", "internal_decoupled"]  # mem_mode shim -> topology
    ram_styles = ["auto", "block", "distributed"]  # skip ultra (needs rw=1 pairing)
    rw = [0, 1]

    naive = len(topologies) * len(ram_styles) * len(rw)

    # Count distinct legal points in the dependent space.
    seen = set()
    for mm, rs, w in itertools.product(topologies, ram_styles, rw):
        assignment = base_assignment(mem_mode=mm)
        if mm == "internal_decoupled":
            assignment[PARAM_RAM_STYLE] = rs
            assignment[PARAM_RUNTIME_WRITEABLE] = w
        r = resolve(schema, ctx, assignment)
        if isinstance(r, Point):
            # Identify the point by the parameters axes that actually exist in it.
            key = (
                r[PARAM_TOPOLOGY],
                r.get(PARAM_RAM_STYLE),
                r.get(PARAM_RUNTIME_WRITEABLE),
            )
            seen.add(key)

    dependent = len(seen)
    # Decoupled: 3 ram x 2 rw = 6 distinct; embedded collapses to 1.
    assert dependent == 6 + 1
    assert dependent < naive


# ---------------------------------------------------------------------------
# F1 — packed feasibility computes NUM_LANES for real (not the dropped-term bug)
# ---------------------------------------------------------------------------


def test_packed_num_lanes_gate(schema):
    # F1 regression: w<=8 & a<=9 on DSP58 is necessary but NOT sufficient -- packed
    # also needs NUM_LANES<=3. Small widths yield MORE lanes: INT4 (w=4,a=4) on DSP58
    # gives NUM_LANES = 1 + (27-0-4)//7 = 4 (>3), so packed is INFEASIBLE and FINN
    # routes to softvec. The old predicate dropped this term and wrongly accepted it.
    ctx = make_context(VERSAL, weights=narrow_weights(wdt="INT4"), wdt="INT4", idt="INT4")
    r = resolve(
        schema,
        ctx,
        base_assignment(implementation=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Illegal)
    assert any("NUM_LANES" in reason for reason in r.reasons)
    # softvec remains feasible on the very same context.
    r_sv = resolve(
        schema,
        ctx,
        base_assignment(implementation=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r_sv, Point)


def test_packed_num_lanes_ok_for_int8(schema):
    # The complement: INT8 (w=8) on DSP58 gives NUM_LANES=2 (<=3) -> packed feasible.
    ctx = make_context(VERSAL, weights=narrow_weights(wdt="INT8"), wdt="INT8")
    r = resolve(
        schema,
        ctx,
        base_assignment(implementation=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Point)


# ---------------------------------------------------------------------------
# F3 — true-binary rejection reads the full condition (input OR weight, no xnor)
# ---------------------------------------------------------------------------


def test_hls_rejects_binary_weights(schema):
    # F3: binary WEIGHTS (not just binary input) must be rejected on HLS when not in
    # binaryXnorMode. The old predicate checked only the input and under-rejected.
    ctx = make_context(weights=np.ones((6, 8), dtype=np.float32), wdt="BINARY", idt="INT4")
    r = resolve(schema, ctx, base_assignment(implementation=MVAU_HLS, binaryXnorMode=0))
    assert isinstance(r, Illegal)
    assert any("binary" in reason.lower() for reason in r.reasons)


def test_hls_binary_ok_in_xnor_mode(schema):
    # F3 escape: binaryXnorMode reinterprets binary as bipolar -> allowed.
    ctx = make_context(weights=np.ones((6, 8), dtype=np.float32), wdt="BINARY", idt="BINARY")
    r = resolve(schema, ctx, base_assignment(implementation=MVAU_HLS, binaryXnorMode=1))
    assert isinstance(r, Point)


# ---------------------------------------------------------------------------
# The composability thesis test (acceptance §3.3): a 4th bundle composes with
# ZERO edits to the three real bundles or the op-level shared elements.
# ---------------------------------------------------------------------------


def test_fourth_implementation_composes_additively():
    from finn.kernels.space import Implementation
    from finn.kernels.ops.mvau import mvau_kernel, mvau_pool
    from finn.kernels.ops.mvau.op import COMPUTE_TILING

    # A hypothetical LUT-based RTL MVU, declared as ONE new bundle. It carries its
    # OWN feasibility (say: only legal on non-Versal parts) and its own axes/sources.
    # It folds like the other MVU compute cores (same COMPUTE_TILING), so the engine
    # derives its SIMD/PE dials identically.
    def lut_rtl_feasible(p, ctx):
        if is_versal(ctx.fpgapart):
            return "mvau_lut_rtl targets non-Versal parts only (hypothetical)"
        return None

    lut_rtl = Implementation(
        name="mvau_lut_rtl",
        feasible=lut_rtl_feasible,
        axes=(),  # inherits only op-level shared axes + engine-derived fold dials
        derived=(Derived("language", lambda p, ctx: "rtl"),),
        predicates=(),
        sources=("mvu_lut.sv",),
        tiling=COMPUTE_TILING,
    )

    # Assemble the FULL kernel with the 4th member appended -- the three real bundles
    # and the shared elements are used verbatim, unedited. Route through the Kernel
    # facade so the tiling engine generates each impl's fold dials.
    from dataclasses import replace

    base = mvau_kernel()
    kernel4 = replace(base, pool=mvau_pool() + (lut_rtl,))
    schema4 = kernel4.schema()

    # It appears as a pool member and resolves per its OWN feasibility.
    legal = resolve(
        schema4,
        make_context(SEVEN_SERIES),  # non-Versal -> feasible
        base_assignment(implementation="mvau_lut_rtl", mem_mode="internal_embedded"),
    )
    assert isinstance(legal, Point)
    assert legal.language == "rtl"
    assert legal.sources == ("mvu_lut.sv",)

    illegal = resolve(
        schema4,
        make_context(VERSAL),  # Versal -> its own feasible() rejects
        base_assignment(implementation="mvau_lut_rtl", mem_mode="internal_embedded"),
    )
    assert isinstance(illegal, Illegal)
    assert any("non-Versal" in reason for reason in illegal.reasons)

    # And the three original implementations still resolve unchanged in the extended
    # pool -- adding the 4th did not perturb them.
    r_hls = resolve(schema4, make_context(SEVEN_SERIES), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r_hls, Point)
    assert r_hls.language == "hls"


def test_registry_makes_addition_structural():
    # The structural form of the thesis: a bundle registered from its OWN module
    # appears in mvau_pool() with ZERO edits to the package, shared.py, or a sibling.
    # This simulates a third-party `impl_*.py` that self-registers on import.
    from finn.kernels.space import Implementation
    from finn.kernels.ops.mvau import mvau_pool, mvau_schema
    from finn.kernels.ops.mvau.op import COMPUTE_TILING
    from finn.kernels.ops.mvau.registry import register, unregister

    before = {b.name for b in mvau_pool()}
    assert "mvau_stub_backend" not in before

    @register
    def _stub_bundle():
        # A real MVU compute peer folds like the others, so it carries COMPUTE_TILING;
        # the engine derives its SIMD/PE dials from that.
        return Implementation(
            name="mvau_stub_backend", sources=("stub.sv",), tiling=COMPUTE_TILING
        )

    try:
        after = {b.name for b in mvau_pool()}
        assert after == before | {"mvau_stub_backend"}
        # It resolves as a real pool member via the normal schema path.
        r = resolve(
            mvau_schema(),
            make_context(SEVEN_SERIES),
            base_assignment(implementation="mvau_stub_backend", mem_mode="internal_embedded"),
        )
        assert isinstance(r, Point)
        assert r.sources == ("stub.sv",)
    finally:
        # Keep the registry clean for other tests (registration is a global side effect).
        unregister("mvau_stub_backend")
