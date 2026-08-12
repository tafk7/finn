############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""DataflowKernel/space assembly is memoized, and INVALIDATED by registration (engine hone 4.1).

``compile()`` ran on every query — ``_assignment``, ``configure``,
``first_feasible_backend`` and ``get_nodeattr_types`` each rebuilt the entire space, and
``mvau_kernel()`` rebuilt every ``Backend`` from the registry each call.

Caching that is easy; caching it CORRECTLY is the point of these tests. A stale cache would
silently pin the pool as it stood at first call, so "add a backend, edit nothing else" —
the property the registry exists to provide — would quietly stop holding. Invalidation is
therefore structural: every cache keys on a registry ``generation()`` counter that
registration bumps, rather than on anyone remembering to clear it.
"""

import pytest

from finn.kernels.compute.mvau import kernel as mvau_kernel_mod
from finn.kernels.compute.mvau.op import mvau_kernel
from finn.kernels.compute.thresholding.op import thresholding_kernel
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.registry import make_registry


def test_kernel_and_space_are_memoized():
    assert mvau_kernel() is mvau_kernel()
    assert thresholding_kernel() is thresholding_kernel()
    k = mvau_kernel()
    assert k.compile() is k.compile()


def test_registering_a_backend_invalidates_the_kernel_cache():
    """THE reason this needs a test. A newly registered backend must appear in the very next
    mvau_kernel() — otherwise memoization silently breaks the registry's whole promise."""
    from finn.kernels.compute.mvau.op import COMPUTE_STREAM
    from finn.kernels.compute.mvau.registry import register, unregister

    before = mvau_kernel()
    assert "mvau_memo_stub" not in {b.name for b in before.pool}

    @register
    def _stub():
        return Backend(
            name="mvau_memo_stub",
            sources=("stub.sv",),
            ports=ports_from(stream=COMPUTE_STREAM),
        )

    try:
        after = mvau_kernel()
        assert after is not before, "cache was not invalidated by register()"
        assert "mvau_memo_stub" in {b.name for b in after.pool}
        # and it is really in the compiled space, not just the pool list
        root = next(a for a in after.compile().axes if a.name == "backend")
        assert "mvau_memo_stub" in root.domain(None, None)
    finally:
        unregister("mvau_memo_stub")

    restored = mvau_kernel()
    assert "mvau_memo_stub" not in {b.name for b in restored.pool}, (
        "cache was not invalidated by unregister()"
    )


def test_parameters_registration_also_invalidates():
    """A compute kernel delivers through the PARAMETERS pool, so it must key on that
    registry too — keying only on its own would pin a stale storage half."""
    from finn.kernels.dataflow.parameters.names import DECOUPLED
    from finn.kernels.dataflow.parameters.registry import register, unregister
    from finn.kernels.model.source_backend import source_backend

    before = mvau_kernel()

    @register
    def _stub_topology(iface):
        return source_backend("memo_stub_topology", mem_mode=DECOUPLED, language="rtl")

    try:
        assert mvau_kernel() is not before, (
            "a parameters registration must invalidate the compute kernel cache"
        )
    finally:
        unregister("memo_stub_topology")


def test_memoized_space_is_frozen_and_its_caches_idempotent():
    """Sharing one space across queries is only safe if nothing mutates it. It is frozen,
    and its lazily-built order/stratum caches recompute to the same values."""
    space = mvau_kernel().compile()
    with pytest.raises(Exception):
        space.axes = ()  # frozen dataclass

    first_order = [a.name for a in space.ordered_axes()]
    first_strata = {p.describe(): space.stratum_of(p) for p in space.predicates}
    space.finalize()  # idempotent
    assert [a.name for a in space.ordered_axes()] == first_order
    assert {p.describe(): space.stratum_of(p) for p in space.predicates} == first_strata


def test_generation_counter_tracks_registration():
    """The invalidation primitive itself, on a private registry."""
    register, build_pool, _names, unregister, generation = make_registry("probe")

    start = generation()

    @register
    def _a():
        return Backend(name="a")

    assert generation() > start
    after_register = generation()

    unregister("a")
    assert generation() > after_register

    unregister("not_present")  # no-op must not bump
    assert generation() == after_register + 1
