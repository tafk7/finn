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
from finn.kernels.model.param_names import (
    STREAM,
    param_stream_width_key,
    demand_key,
    depth_key,
    init_file_key,
    pumped_memory_key,
    ram_style_key,
    runtime_writeable_key,
    sets_key,
    width_key,
)
from finn.util.basic import is_versal

from .emit_memstream import _MEMSTREAM_WRAPPER_SCHEMA, MEMSTREAM_MANIFEST, emit_memstream
from .names import DECOUPLED
from .registry import register
from finn.kernels.model.memory_backend import memory_backend


# =============================================================================
# Selection axes — the free choices of the decoupled (memstream) topology.
# These are always present when this topology is selected (the pool guards them on
# selection); no further guard is needed inside the bundle. Names are namespaced AND
# interface-keyed ``parameters.<iface>.*`` (self-identification; keeps the composed op
# point collision-safe AND lets the pool compose once per parameter interface).
# =============================================================================


def _decoupled_axes(iface):
    return (
        discrete_axis(ram_style_key(iface), {"auto", "block", "distributed", "ultra"}, "auto"),
        discrete_axis(runtime_writeable_key(iface), {0, 1}, 0),
        discrete_axis(pumped_memory_key(iface), {0, 1}, 0),
    )


# =============================================================================
# Feasibility predicates.
#
# The URAM gate reads only this bundle's axes + the device. The `pumpedMemory =>
# not(parallelism==1)` gate reads the compute→memory DEMAND's parallelism (= PE*SIMD),
# not the raw compute fold — so it too is owned here now, not brokered by the op. Both
# no-op when the demand is absent, so the pool still resolves standalone.
# =============================================================================


def _uram_gate(iface):
    @predicate(f"{ram_style_key(iface)}=ultra & not versal => runtime_writeable=1")
    def _uram_requires_ultrascale(p, ctx):
        # THE combination gate — reads point AND device in one condition (hls:147).
        if p.get(ram_style_key(iface)) == "ultra" and not is_versal(ctx.fpgapart) and (
            p.get(runtime_writeable_key(iface), 0) != 1
        ):
            return (
                "URAM weights on a non-Versal (UltraScale) device require "
                "runtime_writeable_weights=1 (hls:147)"
            )
        return None

    return _uram_requires_ultrascale


def _pumped_gate(iface):
    @predicate(f"{pumped_memory_key(iface)} => not (parallelism == 1)")
    def _pumped_memory_needs_parallelism(p, ctx):
        # pumpedMemory splits each weight word across a double-pumped memory; with a
        # 1-element demand (PE==SIMD==1) there is nothing to split (base:717 "known bug").
        # Reads the demand's parallelism, not the raw compute fold — so the gate is owned
        # here, not brokered by the op. Absent demand ⇒ no gate.
        demand = _demand(p, iface)
        if p.get(pumped_memory_key(iface), 0) and demand is not None and demand.parallelism == 1:
            return (
                "pumpedMemory with parallelism=1 (PE=SIMD=1) is a known-bad "
                "configuration (base:717)"
            )
        return None

    return _pumped_memory_needs_parallelism


# =============================================================================
# Memstream GEOMETRY — derived HERE from the compute→memory demand spec.
#
# Historically the composing op computed these (it had the compute fold in scope and
# the parameters bundle did not). With the demand contract the op publishes a
# realization-free ParamDemand under `parameters.<iface>.demand`, and THIS bundle turns it
# into memstream geometry — the memory backend owning its own realization. Each returns
# None when the demand is absent (standalone resolve, no compute core, OR the interface is
# consumed in constant mode), preserving the pool's stand-alone resolvability.
# =============================================================================


def _demand(p, iface):
    """The compute→memory demand the op published for ``iface``, or None (standalone /
    no core / constant-mode consumption)."""
    return p.get(demand_key(iface), None)


def _geometry_derived(iface):
    def _mem_width(p, ctx):
        # WIDTH = roundup(bit_rate, 8) = roundup(PE*SIMD*wbits, 8) (base:326).
        demand = _demand(p, iface)
        if demand is None:
            return None
        return int(roundup_to_integer_multiple(demand.bit_rate, 8))

    def _mem_depth(p, ctx):
        # DEPTH = the demand's word count per set (= WMEM*TH for MVAU) (base:323).
        demand = _demand(p, iface)
        if demand is None:
            return None
        return int(demand.depth)

    def _mem_sets(p, ctx):
        # SETS = MLO set count (cardinality — coord B). 1 until MLO is first-class; the
        # demand carries no cardinality yet, so it is 1 whenever a demand is present.
        demand = _demand(p, iface)
        if demand is None:
            return None
        return 1

    def _mem_init_file(p, ctx):
        # INIT_FILE = "memblock.dat", blanked for URAM on a non-Versal part (base:331-332):
        # URAM cannot be preloaded from a .dat on UltraScale (loaded via AXI-lite instead).
        if _demand(p, iface) is None:
            return None
        if p.get(ram_style_key(iface)) == "ultra" and not is_versal(ctx.fpgapart):
            return ""
        return "memblock.dat"

    def _stream_width(p, ctx):
        # A decoupled weight port carries bit_rate = PE*SIMD*wbits (un-padded — the
        # memstream WIDTH pads to a byte; the compute-side stream width does not). 0 with
        # no demand.
        demand = _demand(p, iface)
        return 0 if demand is None else int(demand.bit_rate)

    # The compute-facing delivery-port stream width, namespaced per interface
    # (``parameters.<iface>.stream_width``) so it never collides — emitted unconditionally
    # for every delivered interface (no ``weights``-only special case).
    return (
        Derived(param_stream_width_key(iface), _stream_width),
        Derived(width_key(iface), _mem_width),
        Derived(depth_key(iface), _mem_depth),
        Derived(sets_key(iface), _mem_sets),
        Derived(init_file_key(iface), _mem_init_file),
    )


@register
def decoupled_topology(iface):
    return memory_backend(
        DECOUPLED,
        mode=STREAM,  # an AXIS port a memstream block feeds
        language="rtl",  # emits its own memstream Verilog streamer
        axes=_decoupled_axes(iface),
        derived=_geometry_derived(iface),
        predicates=(_uram_gate(iface), _pumped_gate(iface)),
        # One source-of-truth: the same manifest the emit resolves for the build copy (F9).
        sources=MEMSTREAM_MANIFEST.filenames,
        emit=emit_memstream,
        # The typed contract for the emitted memstream wrapper (a streaming topology).
        schema=_MEMSTREAM_WRAPPER_SCHEMA,
    )
