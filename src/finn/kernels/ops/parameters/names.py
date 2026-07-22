############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Shared string constants for the ``parameters`` package.

The parameters subsystem owns how a kernel's parameters reach compute (storage +
transport + selection), never their values. Its point keys are **namespaced**
``parameters.*`` — the bundles ARE this subsystem, so naming their own fields with the
subsystem prefix is self-identification, and it keeps the composed op point
self-documenting + collision-safe (a compute axis and a parameters axis can never
clash). Member IDENTITY values (``embedded``/``decoupled``) are plain values on the
``topology`` axis, not keys, so they stay unprefixed. See
``kernel-design/kernel-final-design/param-delivery-design-space.md``.
"""

from __future__ import annotations

# The subsystem namespace, applied to every point KEY this pool contributes.
NS = "parameters"


def ns(name: str) -> str:
    """Namespace a bare field name into the ``parameters.*`` point key."""
    return f"{NS}.{name}"


# --- Namespaced point keys (axis + reserved-derived names) -------------------
TOPOLOGY = ns("topology")  # root selection axis (coordinate A: storage/topology)
RAM_STYLE = ns("ram_style")
RUNTIME_WRITEABLE = ns("runtime_writeable_weights")
PUMPED_MEMORY = ns("pumpedMemory")
SOURCES = ns("sources")  # this pool's sources-derived key (distinct from compute's)

# Memstream GEOMETRY — cross-coordinate derived (read the compute fold), contributed
# by the composing op (ops/mvau/op.py §5), consumed by the memstream emit.
PARAM_DEPTH = ns("depth")  # memory lines = WMEM * TH
PARAM_WIDTH = ns("width")  # padded stream width in bits = roundup(PE*SIMD*wbits, 8)
PARAM_SETS = ns("sets")  # 1, or mlo_max_iter (MLO set count)
PARAM_INIT_FILE = ns("init_file")  # memblock.dat basename, or "" for URAM-non-Versal

# --- Storage-topology member identities (VALUES of the topology axis) --------
# Increment 1 ships two; external/dynamic/off-chip-DMA topologies are later members.
EMBEDDED = "embedded"  # params compiled into the compute core (no streamer, no port)
DECOUPLED = "decoupled"  # on-chip replay streamer (memstream): BRAM/URAM + .dat

# Context tensor name the couplings read (the parameter tensor).
WEIGHTS = "weights"
