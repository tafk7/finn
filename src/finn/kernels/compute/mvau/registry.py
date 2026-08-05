############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU's implementation registry — a thin binding of the op-agnostic
:func:`finn.kernels.model.registry.make_registry` factory to the ``mvau``
op. Bundle modules import ``register`` from here; the package ``__init__`` uses
``build_pool``. See the factory module for the mechanism and rationale.
"""

from __future__ import annotations

from finn.kernels.model.registry import make_registry

register, build_pool, registered_names, unregister, generation = make_registry("mvau")
