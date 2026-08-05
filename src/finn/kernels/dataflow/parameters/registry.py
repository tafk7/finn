############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The ``parameters`` pool registry — a thin binding of the op-agnostic
:func:`finn.kernels.model.registry.make_registry` factory to the
``parameters`` subsystem. Topology bundle modules import ``register`` from here; the
package ``__init__`` uses ``build_pool``. Adding a storage topology = add one
self-registering ``impl_*.py``, edit nothing else.

Unlike a plain compute pool, a storage topology is built FOR a specific parameter
interface (its point keys are interface-namespaced, ``parameters.<iface>.*``). So bundle
factories take the interface name, and ``build_pool(iface)`` threads it. The topology
IDENTITY (``.name`` = ``embedded``/``decoupled``) is interface-independent, so the
registry probes with ``WEIGHTS`` (the default/only live interface) to read ``.name``.
"""

from __future__ import annotations

from finn.kernels.model.registry import make_registry

from .names import WEIGHTS

register, build_pool, registered_names, unregister, generation = make_registry(
    "parameters", probe_arg=WEIGHTS
)
