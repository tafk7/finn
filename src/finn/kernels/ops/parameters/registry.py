############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The ``parameters`` pool registry — a thin binding of the op-agnostic
:func:`finn.kernels.ops._registry.make_registry` factory to the
``parameters`` subsystem. Topology bundle modules import ``register`` from here; the
package ``__init__`` uses ``build_pool``. Adding a storage topology = add one
self-registering ``impl_*.py``, edit nothing else.
"""

from __future__ import annotations

from finn.kernels.ops._registry import make_registry

register, build_pool, registered_names, unregister = make_registry("parameters")
