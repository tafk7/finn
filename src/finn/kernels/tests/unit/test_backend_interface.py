############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Interface`` — the realization-side per-port object (design pitch §2).

These tests prove the seam is correctly assembled: a ``Interface`` built for MVAU
``weights`` produces the (DEMAND stage, guarded delivery pool) pair, its demand closure
returns the sized :class:`ParamDemand`, its mode guard filters the topology domain by the
selected backend's ``consumes``, and a full resolve matches captured baseline literals.
"""

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Context,
    backend_interface_for,
)
from finn.kernels.model.backend import Backend
from finn.kernels.model.param_names import (
    ALL_MODES,
    CONSTANT,
    STREAM,
    demand_key,
    topology_key,
)
from finn.kernels.ops.mvau.op import _delivered_parameters, mvau_pool
from finn.kernels.ops.parameters import WEIGHTS, parameters_pool
from finn.kernels.ops.parameters.names import DECOUPLED, EMBEDDED

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _weights_dp():
    """The MVAU ``weights`` DeliveredParam (the always-live parameter interface)."""
    for dp in _delivered_parameters():
        if dp.iface == WEIGHTS:
            return dp
    raise AssertionError("no weights DeliveredParam")


def _mvau_ctx(part=VERSAL):
    w = np.random.RandomState(0).randint(-7, 7, size=(6, 8)).astype(np.float32)
    return Context(
        shapes={"weights": (6, 8), "inp": (1, 6), "out": (1, 8)},
        datatypes={"weights": DataType["INT8"], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=part,
        clk_ns=5.0,
    )


# ---------------------------------------------------------------------------
# to_subschemas() produces the (DEMAND stage, guarded delivery pool) pair for weights.
# ---------------------------------------------------------------------------


def _names(schema):
    return (
        tuple(a.name for a in schema.axes),
        tuple(d.name for d in schema.derived),
        tuple(p.describe() for p in schema.predicates),
    )


def test_to_subschemas_yields_demand_stage_and_guarded_delivery_pool():
    pool = mvau_pool()
    dp = _weights_dp()

    bi = backend_interface_for(dp, pool)
    demand, delivery = bi.to_subschemas()

    # DEMAND stage: a single derived-only schema publishing parameters.weights.demand.
    assert _names(demand) == ((), (demand_key(WEIGHTS),), ())

    # Delivery pool: the topology root axis first, then the per-topology dials.
    axes = tuple(a.name for a in delivery.axes)
    assert axes[0] == topology_key(WEIGHTS)
    assert f"{topology_key(WEIGHTS).rsplit('.', 1)[0]}.ram_style" in axes
    # The pool's derived + guarded predicates are present (per-topology geometry + gates).
    derived = tuple(d.name for d in delivery.derived)
    assert f"parameters.{WEIGHTS}.width" in derived
    assert f"parameters.{WEIGHTS}.depth" in derived
    assert f"parameters.{WEIGHTS}.sources" in derived
    preds = tuple(p.describe() for p in delivery.predicates)
    assert any("feasibility" in d for d in preds)


def test_backend_interface_declares_two_root_deps():
    bi = backend_interface_for(_weights_dp(), mvau_pool())
    assert bi.deps == frozenset({"backend", topology_key(WEIGHTS)})
    assert bi.schema == WEIGHTS


# ---------------------------------------------------------------------------
# The demand closure returns the same ParamDemand as the delivery path.
# ---------------------------------------------------------------------------


def _probe_point(width_bits, topo=DECOUPLED):
    """A resolved-enough probe: a plain dict standing in for the point, carrying the
    resolved stream width + topology the demand closure reads."""
    from finn.kernels.model.tiling import stream_width_key

    return {stream_width_key(WEIGHTS): width_bits, topology_key(WEIGHTS): topo}


def test_publishes_matches_delivery_demand_for():
    from finn.kernels.model.param_contract import _demand_for

    pool = mvau_pool()
    dp = _weights_dp()
    ctx = _mvau_ctx()

    bi = backend_interface_for(dp, pool)
    want = _demand_for(dp)

    # Stream mode (decoupled): PE*SIMD=4 elems/cycle at INT8 -> demand sized.
    p = _probe_point(width_bits=32, topo=DECOUPLED)
    got_demand = bi.publishes(p, ctx)
    assert got_demand == want(p, ctx)
    assert got_demand is not None
    assert got_demand.parallelism == 4
    assert got_demand.elem_bits == 8
    assert got_demand.depth == 12

    # Constant mode (embedded): nothing to stream -> None, matching the delivery path.
    pc = _probe_point(width_bits=32, topo=EMBEDDED)
    assert bi.publishes(pc, ctx) is None
    assert want(pc, ctx) is None


# ---------------------------------------------------------------------------
# The mode guard filters the topology domain identically per compute backend.
# ---------------------------------------------------------------------------


def test_constrains_filters_topology_domain_like_delivery():
    from finn.kernels.model.param_contract import _topology_domain

    pool = mvau_pool()
    dp = _weights_dp()

    bi = backend_interface_for(dp, pool)
    _, got_legal = bi.constrains
    _, want_legal = _topology_domain(pool, dp)

    # For every compute backend the guard keeps EXACTLY the same topologies as the split
    # delivery path (whatever each backend's consumes declares — permissive or restricted).
    for b in pool:
        p = {"backend": b.name}
        assert set(got_legal(p)) == set(want_legal(p))
        assert got_legal(p)  # non-empty: at least one topology stays selectable


def test_constrains_excludes_embedded_for_stream_only_backend():
    # A synthetic stream-only backend filters embedded (constant) out of the domain.
    backend = Backend(
        name="stream_only",
        stream={"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]},
        consumes={WEIGHTS: {STREAM}},
    )
    dp = _weights_dp()
    bi = backend_interface_for(dp, (backend,))
    _, legal = bi.constrains
    allowed = set(legal({"backend": "stream_only"}))
    assert EMBEDDED not in allowed
    assert DECOUPLED in allowed


def test_backend_interface_reads_backend_consumes_map():
    backend = Backend(name="stream_only", consumes={WEIGHTS: {STREAM}})
    bi = backend_interface_for(_weights_dp(), (backend,))
    assert bi.consumes["stream_only"] == frozenset({STREAM})
    # A permissive backend surfaces None (both modes accepted).
    perm = Backend(name="perm")
    bi2 = backend_interface_for(_weights_dp(), (perm,))
    assert bi2.consumes["perm"] is None
    assert ALL_MODES == frozenset({CONSTANT, STREAM})


# ---------------------------------------------------------------------------
# End-to-end: a full mvau_kernel().configure() for a decoupled-weights node resolves
# to the SAME parameters.weights.{width,depth,demand} as the pre-T4 baseline (captured
# as literals — the Interface wiring is behavior-invariant vs the split path).
# ---------------------------------------------------------------------------


def test_decoupled_weights_resolve_matches_pre_t4_baseline():
    from finn.kernels.ops.mvau.op import mvau_kernel
    from finn.kernels.model.param_names import depth_key, width_key

    k = mvau_kernel()
    ctx = _mvau_ctx()
    r = k.configure(
        ctx,
        {
            "backend": "mvau_hls",
            "PE": 2,
            "SIMD": 2,
            "resType": "lut",
            topology_key(WEIGHTS): DECOUPLED,
        },
    )
    assert r[width_key(WEIGHTS)] == 32
    assert r[depth_key(WEIGHTS)] == 12
    demand = r[demand_key(WEIGHTS)]
    assert (demand.parallelism, demand.elem_bits, demand.depth, demand.cadence) == (4, 8, 12, 1)
