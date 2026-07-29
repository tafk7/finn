############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Acid tests for the derived stream-tiling expression language
(kernelop-tensor-block-stream.md §5.1/§6.2).

The two cases the naive last-axis tiling model cannot express, both resolved here
against a plain Point:

  * MVU tiled weight port  WSIMD = (PE * SIMD) / TH   (mvu_tiled_axi_wrapper.v:20)
  * elementwise rhs        broadcast_aware — size-1 axis replicates, does not fold

Plus: introspectable deps (the reason for an AST over an opaque closure), exact-fold
enforcement, and absent-dep diagnostics.
"""

import pytest

from finn.kernels.space import (
    Const,
    TileError,
    broadcast_aware,
    const,
    derive,
    entry_deps,
    eval_entry,
    param,
)
from finn.kernels.engine.point import Point


# ---------------------------------------------------------------------------
# MVU weight port: WSIMD = PE * SIMD / TH — the derived cross-interface acid test.
# ---------------------------------------------------------------------------


def test_mvu_tiled_weight_port_wsimd():
    # A point as mvau_rtl_tiled would resolve it: PE, SIMD folds + backend-local TH.
    p = Point({"PE": 4, "SIMD": 8, "TH": 2})
    wsimd = derive("PE") * derive("SIMD") / param("TH")
    assert eval_entry(wsimd, p) == (4 * 8) // 2  # == 16


def test_mvu_untiled_weight_port_is_pe_simd():
    # The untiled backend's weight port has no TH: WSIMD = PE * SIMD.
    p = Point({"PE": 4, "SIMD": 8})
    assert eval_entry(derive("PE") * derive("SIMD"), p) == 32


def test_weight_port_deps_are_backend_local():
    # The expr must declare exactly {PE, SIMD, TH} so the schema can order it and so we
    # can confirm every operand is in scope on the bundle that owns TH.
    wsimd = derive("PE") * derive("SIMD") / param("TH")
    assert wsimd.deps() == frozenset({"PE", "SIMD", "TH"})
    assert entry_deps("SIMD") == frozenset({"SIMD"})
    assert entry_deps(1) == frozenset()


def test_div_requires_exact_fold():
    # TH must divide PE*SIMD; a fractional fold is a hard error, not a silent floor.
    p = Point({"PE": 3, "SIMD": 8, "TH": 5})
    with pytest.raises(TileError, match="not an exact integer fold"):
        eval_entry(derive("PE") * derive("SIMD") / param("TH"), p)


# ---------------------------------------------------------------------------
# Elementwise rhs: broadcast_aware — the size-1 replicate exception.
# ---------------------------------------------------------------------------


def test_broadcast_aware_folds_when_not_broadcast():
    # rhs last-dim extent 128, PE=16 -> folds normally to 16.
    p = Point({"rhs_len": 128, "PE": 16})
    assert eval_entry(broadcast_aware("rhs_len", derive("PE")), p) == 16


def test_broadcast_aware_replicates_when_size_one():
    # rhs is broadcast (extent 1): do NOT fold by PE; the stream carries 1 (replicated).
    p = Point({"rhs_len": 1, "PE": 16})
    assert eval_entry(broadcast_aware("rhs_len", derive("PE")), p) == 1


def test_broadcast_aware_deps_include_extent():
    expr = broadcast_aware("rhs_len", derive("PE"))
    assert expr.deps() == frozenset({"rhs_len", "PE"})


# ---------------------------------------------------------------------------
# Diagnostics + coercion.
# ---------------------------------------------------------------------------


def test_absent_dep_is_diagnosed():
    p = Point({"PE": 4})  # SIMD guarded out / not declared
    with pytest.raises(TileError, match="SIMD"):
        eval_entry(derive("PE") * derive("SIMD"), p)


def test_non_integer_value_rejected():
    p = Point({"PE": "wide"})
    with pytest.raises(TileError, match="expected an integer"):
        eval_entry(derive("PE"), p)


def test_bare_int_and_name_coerce():
    p = Point({"SIMD": 8})
    assert eval_entry("SIMD", p) == 8
    assert eval_entry(4, p) == 4
    assert isinstance(const(4), Const)


def test_rmul_and_const_compose():
    # 2 * derive("PE") must work (int on the left).
    p = Point({"PE": 8})
    assert eval_entry(2 * derive("PE"), p) == 16
