############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""MVAU ↔ parameters CROSS-COORDINATE couplings.

When the ``parameters`` pool composes into the MVAU schema, a few quantities read
BOTH coordinate surfaces at once — the compute fold (PE/SIMD/WMEM, from MVAU) AND the
chosen storage topology (``parameters.topology`` etc.). Those cannot live in either
pool alone (a parameters bundle can't see the fold; MVAU op-level can't see the
topology). They live HERE and are appended to the MVAU op schema right before
:func:`~finn.design_space.space.compose` merges the two pools — the one place both
surfaces are in scope. This is exactly the "couplings are parent Derived/Predicate"
rule from the composition analysis
(``kernel-design/kernel-final-design/param-delivery-design-space.md`` §2).

Increment 1 provides the ``weight_stream_width`` derived and the pumpedMemory/fold
gate. The memstream GEOMETRY couplings (``parameters.depth``/``width``/``sets``/
``init_file``) arrive with the memstream emit (Phase 3).
"""

from __future__ import annotations

from qonnx.util.basic import roundup_to_integer_multiple

from finn.design_space.space import Derived, Predicate
from finn.design_space.fixtures.parameters.names import (
    DECOUPLED,
    EMBEDDED,
    PARAM_DEPTH,
    PARAM_INIT_FILE,
    PARAM_SETS,
    PARAM_WIDTH,
    PUMPED_MEMORY,
    RAM_STYLE,
    TOPOLOGY,
)

from .names import WEIGHTS


def _weight_stream_width(p, ctx):
    # base:256-275 — 0 for embedded (weights baked in, no stream port); PE*SIMD*wbits
    # otherwise. Reads the topology (parameters) AND the fold (compute) — the canonical
    # cross-coordinate coupling.
    if p.get(TOPOLOGY) == EMBEDDED:
        return 0
    return p.PE * p.SIMD * ctx.tensor_datatype(WEIGHTS).bitwidth()


def _pumped_memory_not_1x1(p, ctx):
    # pumpedMemory splits each weight word across a double-pumped memory; with
    # PE==SIMD==1 there is nothing to split (base:717 "known bug"). Cross-coordinate:
    # pumpedMemory is a parameters axis, PE/SIMD are the compute fold.
    if p.get(PUMPED_MEMORY, 0) and p.PE == 1 and p.SIMD == 1:
        return "pumpedMemory with PE=SIMD=1 is a known-bad configuration (base:717)"
    return None


# --- Memstream GEOMETRY (decoupled only; None for embedded) ------------------
# These reproduce hwcustomop.py:307-353 generate_hdl_memstream as pure functions of
# the composed point + context. They read the compute fold (PE/SIMD/WMEM) AND the
# parameters topology, so they live here, not in the parameters bundle. Absent
# (None) for embedded, where there is no streamer.


def _is_decoupled(p) -> bool:
    return p.get(TOPOLOGY) == DECOUPLED


def _mem_width(p, ctx):
    # WIDTH = get_instream_width_padded(1) = roundup(PE*SIMD*wbits, 8). base:326.
    if not _is_decoupled(p):
        return None
    wbits = ctx.tensor_datatype(WEIGHTS).bitwidth()
    return int(roundup_to_integer_multiple(p.PE * p.SIMD * wbits, 8))


def _mem_depth(p, ctx):
    # DEPTH = calc_wmem() * TH (MVAU). base:323. TH defaults to 1 (no tiling here).
    if not _is_decoupled(p):
        return None
    return int(p.WMEM * p.get("TH", 1))


def _mem_sets(p, ctx):
    # SETS = mlo_max_iter or 1. base:316-319. Coordinate B (cardinality) — 1 until MLO.
    if not _is_decoupled(p):
        return None
    return int(p.get("mlo_max_iter", 0) or 1)


def _mem_init_file(p, ctx):
    # INIT_FILE = "memblock.dat", blanked for URAM on a non-Versal part (base:331-332):
    # URAM cannot be preloaded from a .dat on UltraScale (loaded via AXI-lite instead).
    if not _is_decoupled(p):
        return None
    from finn.util.basic import is_versal

    if p.get(RAM_STYLE) == "ultra" and not is_versal(ctx.fpgapart):
        return ""
    return "memblock.dat"


def coupling_derived():
    """Cross-coordinate derived to append to the MVAU op schema before compose."""
    return (
        Derived("weight_stream_width", _weight_stream_width),
        # memstream geometry (decoupled only; None for embedded) — consumed by emit
        Derived(PARAM_WIDTH, _mem_width),
        Derived(PARAM_DEPTH, _mem_depth),
        Derived(PARAM_SETS, _mem_sets),
        Derived(PARAM_INIT_FILE, _mem_init_file),
    )


def coupling_predicates():
    """Cross-coordinate predicates to append to the MVAU op schema before compose."""
    return (
        Predicate(
            check=_pumped_memory_not_1x1,
            description="parameters.pumpedMemory => not (PE == SIMD == 1)",
        ),
    )
