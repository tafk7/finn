############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Generic parameter-delivery vocabulary — the point-key builders + consumption-mode
constants the delivery WIRING (:mod:`~finn.kernels.model.parameter_source`) emits and reads.

This is MECHANISM, not op content: a delivery pool is a pool of :class:`Backend`\\ s
selected by a ``parameters.<iface>.topology`` root axis, and the generic DataflowKernel wiring
that connects a compute pool to a delivery pool needs to name that pool's keys and read a
topology's consumption mode. So the vocabulary lives in ``model/`` beside the op-model;
the CONCRETE source backends (``embedded``/``decoupled`` + their memstream emit) and the
identity strings that name them live in ``dataflow/parameters/``.

Point keys are **namespaced AND interface-keyed**: ``parameters.<iface>.<field>`` (e.g.
``parameters.weights.topology``). The interface segment lets a kernel compose the delivery
pool ONCE PER parameter interface (weights, thresholds, …) with no key collision — the
prerequisite for a second parameter interface. See
``scratchpad/archive/pending-harvest/param-delivery-design-space.md`` and
``consumption-mode-delivery.md``.
"""

from __future__ import annotations

# The subsystem namespace, applied to every point KEY the delivery pool contributes.
NS = "parameters"


def _key(iface: str, field: str) -> str:
    """The interface-keyed point key ``parameters.<iface>.<field>``."""
    return f"{NS}.{iface}.{field}"


# --- Interface-keyed point-key builders --------------------------------------
# Each takes the parameter interface name and returns its namespaced point key, so the
# pool composes once per interface with no collision. The topology backends + demand stage
# + emit thread the interface through these.
def topology_key(iface: str) -> str:
    return _key(iface, "topology")  # root selection axis (coordinate A: storage/topology)


def ram_style_key(iface: str) -> str:
    return _key(iface, "ram_style")


def runtime_writeable_key(iface: str) -> str:
    return _key(iface, "runtime_writeable_weights")


def pumped_memory_key(iface: str) -> str:
    return _key(iface, "pumpedMemory")


def sources_key(iface: str) -> str:
    return _key(iface, "sources")  # this pool's sources-derived key, per interface


def demand_key(iface: str) -> str:
    # The compute→memory DEMAND spec — a ParamDemand the composing op publishes (pure
    # compute facts: parallelism, elem_bits, depth), read by the delivery topology to size
    # its own realization. See model/demand.py. NOT a resolve "stage": it is one Derived,
    # ordered between the pools by its declared deps, nothing more.
    return _key(iface, "demand")


# Memstream GEOMETRY — derived by the DECOUPLED topology backend FROM the demand spec
# (impl_decoupled.py). Consumed by the memstream emit. Present-but-None for topologies with
# no streamer (embedded).
def depth_key(iface: str) -> str:
    return _key(iface, "depth")  # memory lines = demand.depth


def width_key(iface: str) -> str:
    return _key(iface, "width")  # padded stream width in bits = roundup(demand.bit_rate, 8)


def sets_key(iface: str) -> str:
    return _key(iface, "sets")  # 1, or the MLO set count (cardinality — coord B, later)


def init_file_key(iface: str) -> str:
    return _key(iface, "init_file")  # memblock.dat basename, or "" for URAM-non-Versal


def param_datatype_key(iface: str) -> str:
    # The storage owner's published datatype AUTHORITY — a
    # :class:`~finn.kernels.engine.param_datatype.ParamDatatype` (value-optimized dtype + a
    # ``values_visible`` permission bit), NOT a bare dtype. The COMPUTE side reads it to size
    # parameter-dependent derivations (e.g. the MVAU accumulator) without re-deriving authority
    # or peeking at storage it does not own. Dispatched per topology (embedded/decoupled)
    # exactly like the stream width; present-but-None when the interface is unwired (standalone
    # resolve). Namespaced per interface so a second parameter interface never collides.
    return _key(iface, "datatype")


def param_stream_width_key(iface: str) -> str:
    # The parameter-delivery stream WIDTH in bits, dispatched per topology (a per-topology
    # fact, NOT an op-level branch): 0 for an embedded-mode topology (no port),
    # demand.bit_rate for a decoupled-mode topology. The COMPUTE side reads it.
    # Namespaced per interface (``parameters.<iface>.stream_width``) so a second streamed
    # parameter interface never collides — the uniform ``parameters.<iface>.*`` rule. (The
    # tiling engine's ``stream_width.<iface>`` is a DIFFERENT key — the compute-fold width;
    # this is the delivery-port width the topology publishes.)
    return _key(iface, "stream_width")


# --- Memory-realization modes (mem_mode — the narrowed realization axis) ------
# The MEMORY-REALIZATION MODE a topology presents to the compute core (consumption-mode-
# delivery.md): a tag OVER coordinate A, not a new coordinate. ``embedded`` = baked into the
# core's fabric (no port); ``decoupled`` = an AXIS port a separate memory backend feeds. A
# delivery topology CARRIES its mode in ``Backend.mem_mode``; a compute backend's
# ``mem_modes`` (per interface) filters the topology domain to matching modes.
#
# De-fusion note: mem_mode is the narrowed realization axis — ``{embedded, decoupled}`` ONLY.
# Staticness (coordinate C), storage location (on/off-chip), and cardinality/MLO (coordinate
# B) are SEPARATE axes, not values here.
EMBEDDED = "embedded"
DECOUPLED = "decoupled"

# Default: an interface a backend says nothing about accepts BOTH modes (permissive — no
# regression vs today, where every backend could take embedded or decoupled weights).
ALL_MODES = frozenset({EMBEDDED, DECOUPLED})
