############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""MVAU's implementation registry — a thin binding of the op-agnostic
:func:`finn.kernels.ops._registry.make_registry` factory to the ``mvau``
op. Bundle modules import ``register`` from here; the package ``__init__`` uses
``build_pool``. See the factory module for the mechanism and rationale.
"""

from __future__ import annotations

from finn.kernels.ops._registry import make_registry

register, build_pool, registered_names, unregister = make_registry("mvau")
