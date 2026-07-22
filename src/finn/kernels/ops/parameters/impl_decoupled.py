############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Storage topology: ``decoupled`` — on-chip replay streamer (memstream).

Parameters live in on-chip RAM (BRAM/URAM) and are replayed onto the compute core's
weight stream by a memstream block, initialized from a ``.dat`` at build time (and
optionally reloadable via AXI-lite). This bundle owns the SELECTION axes of that
topology (``ram_style``, ``runtime_writeable_weights``, ``pumpedMemory``), its
self-contained feasibility gates, AND its memstream GEOMETRY (``depth``/``width``/
``sets``/``init_file``) — which it now derives ITSELF from the compute→memory
:class:`~finn.kernels.ops.parameters.demand.ParamDemand` the composing op publishes
under the ``parameters.demand`` key (rather than the op reaching in to compute the
memstream realization). When the demand is absent (standalone resolve, no compute
core), the geometry derived return ``None`` — so this bundle still resolves alone.

The ``pumpedMemory ⇒ not(PE==SIMD==1)`` gate likewise now reads the demand's
``parallelism`` (= PE*SIMD) instead of the raw compute fold, so it too is owned here.

Axes/predicates relocated verbatim from ``fixtures/mvau/shared.py`` (the old "reserved
composition seam"); the ``mem_mode`` string is gone — being the ``decoupled`` topology
IS "internal_decoupled".
"""

from __future__ import annotations

from qonnx.util.basic import roundup_to_integer_multiple

from finn.kernels.space import Derived, discrete_axis, predicate
from finn.util.basic import is_versal

from .demand import ParamDemand
from .emit_memstream import emit_memstream
from .names import (
    DECOUPLED,
    DEMAND,
    PARAM_DEPTH,
    PARAM_INIT_FILE,
    PARAM_SETS,
    PARAM_WIDTH,
    PUMPED_MEMORY,
    RAM_STYLE,
    RUNTIME_WRITEABLE,
    WEIGHT_STREAM_WIDTH,
)
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
# The URAM gate reads only this bundle's axes + the device. The `pumpedMemory =>
# not(parallelism==1)` gate reads the compute→memory DEMAND's parallelism (= PE*SIMD),
# not the raw compute fold — so it too is owned here now, not brokered by the op. Both
# no-op when the demand is absent, so the pool still resolves standalone.
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


@predicate("parameters.pumpedMemory => not (parallelism == 1)")
def _pumped_memory_needs_parallelism(p, ctx):
    # pumpedMemory splits each weight word across a double-pumped memory; with a
    # 1-element demand (PE==SIMD==1) there is nothing to split (base:717 "known bug").
    # Reads the demand's parallelism, not the raw compute fold — so the gate is owned
    # here, not brokered by the op. Absent demand ⇒ no gate.
    demand = _demand(p)
    if p.get(PUMPED_MEMORY, 0) and demand is not None and demand.parallelism == 1:
        return "pumpedMemory with parallelism=1 (PE=SIMD=1) is a known-bad configuration (base:717)"
    return None


# =============================================================================
# Memstream GEOMETRY — derived HERE from the compute→memory demand spec.
#
# Historically the composing op computed these (it had the compute fold in scope and
# the parameters bundle did not). With the demand contract the op publishes a
# realization-free ParamDemand under `parameters.demand`, and THIS bundle turns it into
# memstream geometry — the memory backend owning its own realization. Each returns None
# when the demand is absent (standalone resolve, no compute core), preserving the pool's
# stand-alone resolvability.
# =============================================================================


def _demand(p):
    """The compute→memory demand the op published, or None (standalone / no core)."""
    return p.get(DEMAND, None)


def _mem_width(p, ctx):
    # WIDTH = roundup(bit_rate, 8) = roundup(PE*SIMD*wbits, 8) (base:326).
    demand = _demand(p)
    if demand is None:
        return None
    return int(roundup_to_integer_multiple(demand.bit_rate, 8))


def _mem_depth(p, ctx):
    # DEPTH = the demand's word count per set (= WMEM*TH for MVAU) (base:323).
    demand = _demand(p)
    if demand is None:
        return None
    return int(demand.depth)


def _mem_sets(p, ctx):
    # SETS = MLO set count (cardinality — coord B). 1 until MLO is first-class; the
    # demand carries no cardinality yet, so it is 1 whenever a demand is present.
    demand = _demand(p)
    if demand is None:
        return None
    return 1


def _mem_init_file(p, ctx):
    # INIT_FILE = "memblock.dat", blanked for URAM on a non-Versal part (base:331-332):
    # URAM cannot be preloaded from a .dat on UltraScale (loaded via AXI-lite instead).
    if _demand(p) is None:
        return None
    if p.get(RAM_STYLE) == "ultra" and not is_versal(ctx.fpgapart):
        return ""
    return "memblock.dat"


def _stream_width(p, ctx):
    # A decoupled weight port carries bit_rate = PE*SIMD*wbits (un-padded — the memstream
    # WIDTH pads to a byte; the compute-side stream width does not). 0 with no demand.
    demand = _demand(p)
    return 0 if demand is None else int(demand.bit_rate)


def _geometry_derived():
    return (
        Derived(WEIGHT_STREAM_WIDTH, _stream_width),
        Derived(PARAM_WIDTH, _mem_width),
        Derived(PARAM_DEPTH, _mem_depth),
        Derived(PARAM_SETS, _mem_sets),
        Derived(PARAM_INIT_FILE, _mem_init_file),
    )


@register
def decoupled_topology():
    return storage_topology(
        DECOUPLED,
        axes=_decoupled_axes(),
        derived=_geometry_derived(),
        predicates=(_uram_requires_ultrascale, _pumped_memory_needs_parallelism),
        sources=("memstream_axi.sv", "memstream.sv", "axilite.sv"),
        emit=emit_memstream,
    )
