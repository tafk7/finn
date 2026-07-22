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
guarding in isolation); an op composes the pool via the ``compose`` helper, which
namespaces it and guards its root so a param-free op simply omits it.
"""

from __future__ import annotations

from finn.kernels.space import Schema, pool_schema

from .demand import ParamDemand  # noqa: F401 (re-exported)
from .names import DECOUPLED, DEMAND, EMBEDDED, SOURCES, TOPOLOGY  # noqa: F401 (re-exported)
from .registry import build_pool

# Import the built-in topology modules for their registration side effect. A new
# storage topology adds one such import (or is discovered) and nothing else.
from . import impl_embedded  # noqa: E402,F401
from . import impl_decoupled  # noqa: E402,F401


def parameters_pool():
    """The registered storage topologies (flat peers), in registration order."""
    return build_pool()


def parameters_schema() -> Schema:
    """The parameters subsystem as a standalone resolve ``Schema``.

    Root axis ``topology`` selects one storage topology; the selected topology's axes
    are present, the others absent. Has no op-level shared axes of its own — topology
    selection IS the surface. Cross-coordinate couplings (memstream depth/width, which
    read the compute fold) are contributed by the composing op, not here, so this
    standalone schema resolves without a compute context.
    """
    return pool_schema(TOPOLOGY, (), (), (), parameters_pool(), sources_key=SOURCES)
