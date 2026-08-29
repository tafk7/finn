# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU compute pool's shared vocabulary.

Its name, the interface every streamed member demands, and the names it
presents its members' exports under.  A leaf module because every member needs
these and the members cannot all import each other.
"""

from __future__ import annotations

from enum import Enum


class MVAUComputeKernelId(str, Enum):
    """Stable identities of the MVAU compute Kernel pool."""

    LEGACY_HLS = "legacy_hls"
    #: Being replaced by ``DOT_PRODUCT``; kept while elaboration still runs
    #: through them.
    SOFT_VECTOR = "rtl_softvec"
    PACKED_DSP = "rtl_packed"
    BATCH_INTERLEAVED_DSP = "rtl_batch_interleaved_dsp58"
    #: The decomposed member: an activation replay plus this dot product.
    DOT_PRODUCT = "dot_product"


#: The Kernel-pool selection name and its owned path root.
MVAU_COMPUTE_SELECTION_NAME = "mvau.compute"

#: The name under which a compute Kernel exports its Region form.
REGION_FORM_EXPORT = "region_form"

#: The parameter interface every streamed compute Kernel demands.
WEIGHT_INTERFACE = "weight"

#: The name under which a compute Kernel exports its natural full-tile weight
#: contract.  A supplier that organizes its output independently of the demand
#: produces this sequence instead, which is what makes an adapter necessary.
FULL_TILE_WEIGHT_EXPORT = "full_tile_weight_port"

__all__ = [
    "MVAUComputeKernelId",
    "FULL_TILE_WEIGHT_EXPORT",
    "MVAU_COMPUTE_SELECTION_NAME",
    "REGION_FORM_EXPORT",
    "WEIGHT_INTERFACE",
]
