############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Storage-topology member IDENTITIES for the ``parameters`` package.

Only the ops-authored bits live here: the topology member-identity strings (the VALUES of
the ``parameters.<iface>.topology`` axis) and the default parameter-interface name. The
GENERIC delivery vocabulary — the ``parameters.<iface>.*`` point-key builders and the
consumption-mode constants — moved DOWN to :mod:`finn.kernels.model.param_names` (it is
engine mechanism the generic Kernel wiring emits, not op content). Import key builders /
mode constants from there; import topology identities from here.
"""

from __future__ import annotations

# --- Storage-topology member identities (VALUES of the topology axis) --------
# Increment 1 ships two; external/dynamic/off-chip-DMA topologies are later members. These
# identities are interface-INDEPENDENT (the same topology can deliver any interface) and are
# ops-authored (the concrete backends live in this package).
EMBEDDED = "embedded"  # params compiled into the compute core (no streamer, no port)
DECOUPLED = "decoupled"  # on-chip replay streamer (memstream): BRAM/URAM + .dat

# Context tensor name the couplings read (the parameter tensor) — the default/only live
# parameter interface this increment.
WEIGHTS = "weights"
