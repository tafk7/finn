############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""T1.3 (F1) — ``protocol`` + ``multiplicity`` on ``InterfaceSchema``.

``protocol`` defaults to ``Stream`` and is restricted to the three dataflow protocols
(Stream/MemoryMapped/Config) — a Sideband/Clock/Reset interface is a construction error.
``multiplicity`` defaults to ``Fixed(1)``; a ``Variadic(count_from=...)`` interface expands
to N concrete peers under :meth:`Kernel.expanded_interfaces` reading ``ctx.arity``.
"""

import pytest

from finn.kernels.engine.context import Context
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.kernel import InterfaceSchema, Kernel
from finn.kernels.model.ports import Direction, Fixed, Protocol, Variadic
from finn.kernels.model.tiling import FULL


def _kernel(ifaces):
    backend = Backend(
        name="core",
        ports=ports_from(stream={i.name: [1, "PE"] for i in ifaces if i.direction == Direction.IN}),
    )
    return Kernel(name="K", interfaces=ifaces, pool=(backend,))


def test_defaults_are_stream_and_fixed_one():
    i = InterfaceSchema("inp", Direction.IN, block=[1, FULL])
    assert i.protocol is Protocol.Stream
    assert i.multiplicity == Fixed(1)


def test_non_dataflow_protocol_rejected():
    with pytest.raises(ValueError, match="not a dataflow protocol"):
        InterfaceSchema("clk", Direction.IN, protocol=Protocol.Clock)


def test_memorymapped_and_config_protocols_accepted():
    mm = InterfaceSchema("mem", Direction.IN, protocol=Protocol.MemoryMapped)
    cfg = InterfaceSchema("cfg", Direction.IN, protocol=Protocol.Config)
    assert mm.protocol is Protocol.MemoryMapped
    assert cfg.protocol is Protocol.Config


def test_fixed_arity_kernel_returns_declared_list_verbatim():
    k = _kernel(
        (
            InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
        )
    )
    ctx = Context(shapes={"inp": (1, 8), "out": (1, 8)})
    # The all-Fixed(1) common case is unchanged: same objects, no ctx.arity read.
    assert k.expanded_interfaces(ctx) == k.interfaces


def test_variadic_expands_to_n_peers_under_ctx_arity():
    k = _kernel(
        (
            InterfaceSchema("ins", Direction.IN, block=[1, FULL], multiplicity=Variadic("ins")),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
        )
    )
    ctx = Context(shapes={}, arities={"ins": 3})
    expanded = k.expanded_interfaces(ctx)
    ins = [i for i in expanded if i.direction == Direction.IN]
    assert [i.name for i in ins] == ["ins_0", "ins_1", "ins_2"]
    assert [i.index for i in ins] == [0, 1, 2]
    assert all(i.multiplicity == Fixed(1) for i in ins)
    # The output interface follows the variadic inputs at its own direction-relative index 0.
    out = [i for i in expanded if i.direction == Direction.OUT]
    assert [i.name for i in out] == ["out"]
    assert out[0].index == 0
