############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Space assembly is memoized per op class, and safe to share.

``compile()`` ran on every query — ``_assignment``, ``configure``,
``first_feasible_backend`` and ``get_nodeattr_types`` each rebuilt the entire space. Caching
it is easy; caching it CORRECTLY is what these tests are for.

**What changed when the container merged into the op class (F6).** Two tests here pinned
registry-driven invalidation: the kernel was a VALUE built by a factory from a mutable
registry, so its cache had to key on a ``generation()`` counter that ``register`` bumped, or
"add a backend, edit nothing else" would silently stop holding. With the pool a plain class
attribute there is no mutable registry behind the compute pool and nothing to invalidate — a
backend is added by naming it in a class body, and a widened pool is a different class with
its own cache. Those two tests are DELETED rather than rewritten: they pinned a mechanism,
and the mechanism is gone. :func:`test_a_subclass_gets_its_own_cache` replaces them, because
that is the property the caching still has to get right.

``make_registry`` itself survives — the STORAGE pool is interface-threaded and still
assembles through it — so its generation counter is still tested at the bottom.
"""

import pytest

from finn.kernels.compute.mvau.op import MvauDataflowOp
from finn.kernels.compute.thresholding.op import ThresholdingDataflowOp
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.registry import make_registry


def test_space_is_memoized_per_class():
    assert MvauDataflowOp.compile() is MvauDataflowOp.compile()
    assert ThresholdingDataflowOp.compile() is ThresholdingDataflowOp.compile()


def test_realized_spaces_are_memoized_per_member():
    first = MvauDataflowOp.realized_space("mvau_hls")
    assert first is MvauDataflowOp.realized_space("mvau_hls")
    assert first is not MvauDataflowOp.realized_space("mvau_dsp_softvec")


def test_a_subclass_gets_its_own_cache():
    """The property that replaces registry invalidation.

    A subclass widening the pool must NOT inherit the parent's compiled space — that would
    be the stale-cache bug in a new shape, silently pinning the space as the parent saw it.
    ``__init_subclass__`` hands each subclass a fresh cache."""
    from finn.kernels.compute.mvau.op import COMPUTE_STREAM

    stub = Backend(name="mvau_cache_stub", ports=ports_from(stream=COMPUTE_STREAM))

    class _Widened(MvauDataflowOp):
        pool = MvauDataflowOp.pool + (stub,)

    assert _Widened.compile() is not MvauDataflowOp.compile()
    assert "mvau_cache_stub" in {b.name for b in _Widened.pool}
    # ...and it is really in the compiled space, not just the pool list.
    root = next(a for a in _Widened.compile().axes if a.name == "backend")
    assert "mvau_cache_stub" in root.domain(None, None)

    # The parent is untouched — the registry path needed a try/finally to guarantee this.
    assert "mvau_cache_stub" not in {b.name for b in MvauDataflowOp.pool}
    assert MvauDataflowOp.compile() is MvauDataflowOp.compile()


def test_two_ops_do_not_share_a_cache():
    """The failure mode of declaring the cache as a ClassVar on the BASE: every op would
    write into one dict and the second op would read the first op's space."""
    assert MvauDataflowOp.compile() is not ThresholdingDataflowOp.compile()
    assert MvauDataflowOp._space_cache is not ThresholdingDataflowOp._space_cache


def test_memoized_space_is_frozen_and_its_caches_idempotent():
    """Sharing one space across queries is only safe if nothing mutates it. It is frozen,
    and its lazily-built order/stratum caches recompute to the same values."""
    space = MvauDataflowOp.compile()
    with pytest.raises(Exception):
        space.axes = ()  # frozen dataclass

    first_order = [a.name for a in space.ordered_axes()]
    first_strata = {p.describe(): space.stratum_of(p) for p in space.predicates}
    space.finalize()  # idempotent
    assert [a.name for a in space.ordered_axes()] == first_order
    assert {p.describe(): space.stratum_of(p) for p in space.predicates} == first_strata


def test_generation_counter_tracks_registration():
    """The invalidation primitive itself, on a private registry. Still live: the STORAGE
    pool is interface-threaded and assembled through ``make_registry``."""
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
