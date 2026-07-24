############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Generic parameter-delivery vocabulary — the point-key builders + consumption-mode
constants the delivery WIRING (``space/delivery.py``) emits and reads.

This is MECHANISM, not op content: a delivery pool is a pool of :class:`Backend`\\ s
selected by a ``parameters.<iface>.topology`` root axis, and the generic Kernel wiring
that connects a compute pool to a delivery pool needs to name that pool's keys and read a
topology's consumption mode. So the vocabulary lives in ``space/`` beside the engine; the
CONCRETE delivery backends (``embedded``/``decoupled`` + their memstream emit) and the
identity strings that name them stay in ``ops/parameters``.

Point keys are **namespaced AND interface-keyed**: ``parameters.<iface>.<field>`` (e.g.
``parameters.weights.topology``). The interface segment lets a kernel compose the delivery
pool ONCE PER parameter interface (weights, thresholds, …) with no key collision — the
prerequisite for a second parameter interface. See
``kernel-design/kernel-final-design/param-delivery-design-space.md`` and
``consumption-mode-delivery.md``.
"""

from __future__ import annotations

# The subsystem namespace, applied to every point KEY the delivery pool contributes.
NS = "parameters"


def ns(name: str) -> str:
    """Namespace a bare field name into the ``parameters.*`` point key (iface-agnostic —
    used only for the ``weights``-only global :data:`WEIGHT_STREAM_WIDTH` alias below)."""
    return f"{NS}.{name}"


def _key(iface: str, field: str) -> str:
    """The interface-keyed point key ``parameters.<iface>.<field>``."""
    return f"{NS}.{iface}.{field}"


# --- Interface-keyed point-key builders --------------------------------------
# Each takes the parameter interface name and returns its namespaced point key, so the
# pool composes once per interface with no collision. The topology bundles + demand stage
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
    # compute facts: parallelism, elem_bits, depth, cadence), read by the delivery topology
    # to size its own realization. See space/demand.py.
    return _key(iface, "demand")


# Memstream GEOMETRY — derived by the DECOUPLED topology bundle FROM the demand spec
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


# The weight-delivery stream WIDTH in bits, dispatched per topology (a per-topology fact,
# NOT an op-level branch): 0 for embedded (no port), demand.bit_rate for decoupled. Kept
# UN-namespaced and weights-only this increment — the compute side reads it, and weights is
# the only live parameter interface; a second interface generalizes it (thresholds increment).
WEIGHT_STREAM_WIDTH = "weight_stream_width"


# --- Consumption modes (the tag OVER coordinate A) ---------------------------
# The CONSUMPTION MODE a topology presents to the compute core (consumption-mode-
# delivery.md): a tag OVER coordinate A, not a new coordinate. ``constant`` = baked into the
# core (no port); ``stream`` = an AXIS port a delivery block feeds. A delivery topology
# CARRIES its mode in ``Backend.mode``; a compute backend's ``consumes`` (per interface)
# filters the topology domain to matching modes.
CONSTANT = "constant"
STREAM = "stream"

# Default: an interface a backend says nothing about accepts BOTH modes (permissive — no
# regression vs today, where every backend could take embedded or decoupled weights).
ALL_MODES = frozenset({CONSTANT, STREAM})
