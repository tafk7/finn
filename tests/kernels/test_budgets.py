# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a kernel gives a shell to count, as derived values: whether it takes the doubled
clock (``takes_doubled_clock``) and the memory ports it initiates itself
(``memory_initiators``), on the Chain (``support.chain``). A parent's doubled clock reads its
child's: a MatMul its selected core's, a channel its source's, False where neither is
present.
"""

from __future__ import annotations

from typing import Any

from support import chain
from support.helpers import choose

from finn.core.space import Available, Unresolved
from finn.kernels.matmul import MatMulKernel


def test_no_kernel_initiates_a_memory_port() -> None:
    network = chain.chain()
    for name in network.members:
        assert network.member(name).memory_initiators == 0, name


def _value(point: Any, member: Any) -> object:
    found = point.query(member)
    return found.value if isinstance(found, Available) else found


def test_a_parent_takes_the_doubled_clock_its_child_takes() -> None:
    """A MatMul takes it where its selected core pumps (its own ``clocking`` does not: a
    parent's is its own), never reading a case not selected; a channel where its source's
    memory pumps; a channel with no source does not, where reading a source would fail."""
    network = chain.network()
    first = network.member("first")
    assert isinstance(_value(first, MatMulKernel.takes_doubled_clock), Unresolved)  # no core
    packed = choose(network, {"first.compute": "packed"})
    for pumped in (False, True):
        point = choose(packed, {"first.compute.packed.compute_pumping": pumped})
        mm = point.member("first")
        assert mm.takes_doubled_clock is pumped and mm.clocking.doubling is False
        assert mm.compute.takes_doubled_clock is pumped
    assert network.member("x").takes_doubled_clock is False  # no known value, no source
    for pumped in (False, True):
        point = choose(network, {"w1.source.pumped_memory": pumped})
        assert point.member("w1").takes_doubled_clock is pumped
