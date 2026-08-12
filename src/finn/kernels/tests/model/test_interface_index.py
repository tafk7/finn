############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""T0.2 (F9) — ``index`` folded onto ``InterfaceSchema`` (``PortSpec`` deleted).

The interface's own ``index`` is the node-slot the adapter reads to resolve its Context
tensor. A sentinel ``-1`` resolves to declaration order among same-direction peers; an
explicit index (an operand wired at a shifted slot) is honored verbatim.
"""

from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.kernel import InterfaceSchema
from .synthetic import synthetic_op
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL


def _kernel(ifaces):
    backend = Backend(
        name="core",
        ports=ports_from(stream={i.name: [1, "PE"] for i in ifaces if i.direction == Direction.IN}),
    )
    return synthetic_op(interfaces=ifaces, pool=(backend,), name="K")


def test_sentinel_indices_resolve_to_declaration_order():
    k = _kernel(
        (
            InterfaceSchema("a", Direction.IN, block=[1, FULL]),
            InterfaceSchema("b", Direction.IN, block=[1, FULL]),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
        )
    )
    by_name = {i.name: i for i in k.interfaces}
    assert by_name["a"].index == 0
    assert by_name["b"].index == 1  # second same-direction peer
    assert by_name["out"].index == 0  # index is DIRECTION-relative


def test_explicit_index_is_honored_and_indices_are_direction_relative():
    # An operand wired at a SHIFTED slot: 'b' declared at index 2 though it is the 2nd input.
    k = _kernel(
        (
            InterfaceSchema("a", Direction.IN, block=[1, FULL], index=0),
            InterfaceSchema("b", Direction.IN, block=[1, FULL], index=2),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL], index=0),
        )
    )
    by_name = {i.name: i for i in k.interfaces}
    assert by_name["b"].index == 2  # explicit index untouched, not renumbered to 1
    assert by_name["out"].index == 0
