############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The ``parameters`` subsystem as a pool of storage-topology bundles.

The parameters subsystem owns how a kernel's parameters reach compute — storage,
transport, and selection — never their values. It is a SECOND selection pool that
composes into an op's schema alongside the compute pool (namespaced ``parameters.*``),
proven to need no new engine primitive (``tests/test_composition_mapping.py``,
``kernel-design/kernel-final-design/param-delivery-design-space.md``).

The pool's root axis is ``topology`` (coordinate A: storage/topology). Increment 1
ships two members:

  * ``embedded`` (``impl_embedded.py``) — params compiled in; no streamer, no free
    axes. The pool default.
  * ``decoupled`` (``impl_decoupled.py``) — on-chip memstream replay; owns
    ``ram_style``/``runtime_writeable_weights``/``pumpedMemory`` + URAM/pumped gates.

``parameters_schema()`` builds the pool standalone (for unit-testing selection +
guarding in isolation); an op folds the pool in via a
:class:`~finn.kernels.model.interface.DeliverySeam` per delivered interface,
which namespaces its keys and guards its topology root to the backend's consumable modes.
"""

from __future__ import annotations

from finn.kernels.engine.schema import Schema
from finn.kernels.model.backend import pool_schema
from finn.kernels.model.demand import ParamDemand  # noqa: F401 (re-exported)
from finn.kernels.model.param_names import (  # noqa: F401 (re-exported)
    demand_key,
    sources_key,
    topology_key,
)

from .names import DECOUPLED, EMBEDDED, WEIGHTS  # noqa: F401 (re-exported)
from .registry import build_pool

# Import the built-in topology modules for their registration side effect. A new
# storage topology adds one such import (or is discovered) and nothing else.
from . import impl_embedded  # noqa: E402,F401
from . import impl_decoupled  # noqa: E402,F401


def parameters_pool(iface: str = WEIGHTS):
    """The registered storage topologies (flat peers), in registration order, built for
    the parameter interface ``iface`` (its axes/geometry are ``parameters.<iface>.*``)."""
    return build_pool(iface)


def parameters_schema(iface: str = WEIGHTS) -> Schema:
    """The parameters subsystem for one parameter interface as a standalone resolve
    ``Schema``.

    Root axis ``parameters.<iface>.topology`` selects one storage topology; the selected
    topology's axes are present, the others absent. Has no op-level shared axes of its own
    — topology selection IS the surface. Cross-coordinate couplings (memstream
    depth/width, which read the compute fold; the topology-domain guard, which reads the
    selected compute backend's ``mem_modes``) are contributed by the composing op, not
    here, so this standalone schema resolves without a compute context. Composing the pool
    for a second interface is just a second ``parameters_schema(other)`` — the keys never
    collide.
    """
    return pool_schema(
        topology_key(iface), (), (), (), parameters_pool(iface), sources_key=sources_key(iface)
    )
