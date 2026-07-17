############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Storage topology: ``decoupled`` — on-chip replay streamer (memstream).

Parameters live in on-chip RAM (BRAM/URAM) and are replayed onto the compute core's
weight stream by a memstream block, initialized from a ``.dat`` at build time (and
optionally reloadable via AXI-lite). This bundle owns the SELECTION axes of that
topology (``ram_style``, ``runtime_writeable_weights``, ``pumpedMemory``) and its
self-contained feasibility gates. The memstream geometry (``depth``/``width``/``sets``/
``init_file``) is CROSS-COORDINATE — it reads the compute fold (PE/SIMD/WMEM) and the
build dir — so it is contributed as a parent derived at compose time, not here
(keeping this bundle resolvable standalone). Emit (the memstream wrapper ``.v`` + the
weight ``.dat``) is wired in Phase 3.

Axes/predicates relocated verbatim from ``fixtures/mvau/shared.py`` (the old "reserved
composition seam"); the ``mem_mode`` string is gone — being the ``decoupled`` topology
IS "internal_decoupled".
"""

from __future__ import annotations

from finn.design_space.space import discrete_axis, predicate
from finn.util.basic import is_versal

from .names import DECOUPLED, PUMPED_MEMORY, RAM_STYLE, RUNTIME_WRITEABLE
from .registry import register
from .topology import storage_topology


# =============================================================================
# Selection axes — the free choices of the decoupled (memstream) topology.
# These are always present when this topology is selected (the pool guards them on
# selection); no further guard is needed inside the bundle. Names are namespaced
# ``parameters.*`` (self-identification; keeps the composed op point collision-safe).
# =============================================================================


def _decoupled_axes():
    return (
        discrete_axis(RAM_STYLE, {"auto", "block", "distributed", "ultra"}, "auto"),
        discrete_axis(RUNTIME_WRITEABLE, {0, 1}, 0),
        discrete_axis(PUMPED_MEMORY, {0, 1}, 0),
    )


# =============================================================================
# Feasibility predicates.
#
# The URAM gate below is self-contained: it reads only this bundle's axes + the
# device. The `pumpedMemory => not (PE == SIMD == 1)` gate is genuinely
# CROSS-COORDINATE (it reads the compute fold PE/SIMD), so it is NOT a bundle
# predicate — it is contributed by the composing op alongside the memstream geometry
# derived, where the fold is in scope. Keeping it out here lets the pool resolve
# standalone.
# =============================================================================


@predicate("parameters.ram_style=ultra & not versal => runtime_writeable=1")
def _uram_requires_ultrascale(p, ctx):
    # THE combination gate — reads point AND device in one condition (hls:147).
    if p.get(RAM_STYLE) == "ultra" and not is_versal(ctx.fpgapart) and (
        p.get(RUNTIME_WRITEABLE, 0) != 1
    ):
        return (
            "URAM weights on a non-Versal (UltraScale) device require "
            "runtime_writeable_weights=1 (hls:147)"
        )
    return None


@register
def decoupled_topology():
    return storage_topology(
        DECOUPLED,
        axes=_decoupled_axes(),
        predicates=(_uram_requires_ultrascale,),
        sources=("memstream_axi.sv", "memstream.sv", "axilite.sv"),
        # emit wired in Phase 3 (emit_memstream).
    )
