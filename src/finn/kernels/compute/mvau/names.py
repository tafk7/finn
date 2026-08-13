############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Shared MVAU tensor names — the Context keys the whole package reads against.

A LEAF module, and this is the only reason it exists: eight sibling modules key
``Context`` lookups by these strings, and ``op.py`` — their natural home, since the
interface list declares them — imports the backend modules, which are among the eight.
Declaring them there would close that loop.

They are constants rather than literals because they are DICT KEYS:
``ctx.tensor_datatype(WEIGHTS)`` and friends index plain dicts, so a typo is a resolve-time
``KeyError`` where a name is an import-time ``NameError``. Thirty-one such lookups.

Pool-member identities are NOT here — each backend names itself in its own ``impl_*.py``,
because nothing dispatches on the value.

Tensor-name convention for the Context these key into:
    "inp"        the activation input tensor   (dynamic)
    "weights"    the weight tensor             (static initializer VALUES)
    "out"        the output tensor             (dtype derived when there is no activation)
    "thresholds" the OPTIONAL activation operand — present iff a 3-input fused node
"""

from __future__ import annotations

WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"
# Presence is EMERGENT (initializer attached?), superseding the classic ``noActivation`` flag.
THRESHOLDS = "thresholds"
