############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The CONSUMPTION-MODE delivery spine (consumption-mode-delivery.md).

"embedded" is not a delivery topology — it is the consumption MODE ``constant``: the
compute core bakes the parameter in, so there is no stream to deliver and no demand. A
compute :class:`~finn.kernels.space.backend.Backend` declares, per parameter interface,
which modes it can consume (``consumes``); the per-interface delivery-topology domain is
filtered to matching modes. These tests exercise that spine on the live ``weights``
interface, plus the per-interface namespacing that makes a SECOND parameter interface
additive.

Scope note: the built MVAU backends declare no ``consumes`` (permissive default = both
modes), so the domain is unrestricted and ``embedded`` stays the default — nothing
regresses. The RESTRICTION path (a backend that consumes only ``stream``) is proven here
with a synthetic backend, since populating restrictive values on the real DSP cores is a
deliberate later behavior change (it would make ``embedded`` illegal for them).
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Backend,
    Context,
    Illegal,
    Interface,
    Kernel,
    KernelSchema,
    Direction,
    FULL,
    pool_schema,
    resolve,
)
from finn.kernels.ops.parameters import WEIGHTS, parameters_pool
from finn.kernels.ops.parameters.names import (
    CONSTANT,
    DECOUPLED,
    EMBEDDED,
    STREAM,
    TOPOLOGY_MODE,
    demand_key,
    depth_key,
    topology_key,
    width_key,
)


VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _mvau_ctx(part=VERSAL):
    w = np.random.RandomState(0).randint(-7, 7, size=(6, 8)).astype(np.float32)
    return Context(
        shapes={"weights": (6, 8), "inp": (1, 6), "out": (1, 8)},
        datatypes={"weights": DataType["INT8"], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=part,
        clk_ns=5.0,
    )


def _resolve_mvau(assignment):
    from finn.kernels.ops.mvau import mvau_schema

    base = {"implementation": "mvau_hls", "PE": 2, "SIMD": 2, "resType": "lut", "noActivation": 1}
    base.update(assignment)
    return resolve(mvau_schema(), _mvau_ctx(), base)


# ---------------------------------------------------------------------------
# Topology mode tags — the constant/stream partition over coordinate A.
# ---------------------------------------------------------------------------


def test_embedded_is_constant_decoupled_is_stream():
    # The one structural claim: embedded IS the constant mode, decoupled a stream mode.
    assert TOPOLOGY_MODE[EMBEDDED] == CONSTANT
    assert TOPOLOGY_MODE[DECOUPLED] == STREAM


# ---------------------------------------------------------------------------
# Constant-mode demand is None; stream-mode demand is sized.
# ---------------------------------------------------------------------------


def test_constant_mode_has_no_demand():
    # embedded (the pool default) consumes weights in constant mode -> nothing to deliver,
    # so the demand derived resolves present-but-None and the geometry no-ops.
    r = _resolve_mvau({})  # default topology = embedded
    assert not isinstance(r, Illegal), getattr(r, "reasons", None)
    assert r[topology_key(WEIGHTS)] == EMBEDDED
    assert r[demand_key(WEIGHTS)] is None
    assert r[depth_key(WEIGHTS)] is None  # geometry present-but-None under constant mode


def test_stream_mode_sizes_the_demand():
    r = _resolve_mvau({topology_key(WEIGHTS): DECOUPLED})
    assert not isinstance(r, Illegal), getattr(r, "reasons", None)
    demand = r[demand_key(WEIGHTS)]
    assert demand is not None
    # PE*SIMD = 4 elems/cycle, INT8 -> bit_rate 32, WMEM=6*8/4=12 words.
    assert demand.parallelism == 4
    assert demand.elem_bits == 8
    assert demand.depth == 12
    assert r[width_key(WEIGHTS)] == 32  # roundup(bit_rate, 8)


# ---------------------------------------------------------------------------
# Default backends are permissive — both topologies remain selectable.
# ---------------------------------------------------------------------------


def test_permissive_backend_keeps_both_topologies():
    # The real MVAU HLS backend declares no consumes -> both embedded and decoupled legal.
    assert not isinstance(_resolve_mvau({}), Illegal)
    assert not isinstance(_resolve_mvau({topology_key(WEIGHTS): DECOUPLED}), Illegal)


# ---------------------------------------------------------------------------
# The RESTRICTION path — a backend that consumes only `stream` for weights forbids
# the constant (embedded) topology. Proven with a synthetic stream-only kernel.
# ---------------------------------------------------------------------------


def test_consumes_stream_only_domain_excludes_embedded():
    # A backend that consumes weights in STREAM mode only filters the topology domain to
    # stream topologies: embedded (constant) is excluded, decoupled remains. This is the
    # restriction path the real DSP cores will adopt later (a deliberate behavior change).
    backend = Backend(
        name="stream_only",
        stream={"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]},
        consumes={WEIGHTS: {STREAM}},
    )
    topologies = [b.name for b in parameters_pool(WEIGHTS)]
    legal = {t for t in topologies if TOPOLOGY_MODE[t] in backend.consumes[WEIGHTS]}
    assert EMBEDDED not in legal
    assert DECOUPLED in legal


def _restricted_kernel(consumes):
    """A minimal MVU-shaped Kernel whose single backend declares ``consumes`` for weights,
    composed with the weights parameters pool under the same consumption-mode guard MVAU
    uses — so the restriction is exercised through a real ``resolve``, not just the filter
    math. Mirrors ``_params_subschema``/``_topology_domain`` generically (no MVAU internals)."""
    from dataclasses import replace

    ifaces = (
        Interface("inp", Direction.IN, block=[1, FULL]),
        Interface("weights", Direction.IN, block=[FULL, FULL]),
        Interface("out", Direction.OUT, block=[1, FULL]),
    )
    backend = Backend(
        name="core",
        stream={"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]},
        consumes=consumes,
    )

    def legal(p):
        modes = backend.consumes.get(WEIGHTS) or frozenset({CONSTANT, STREAM})
        topos = [b.name for b in parameters_pool(WEIGHTS)]
        return tuple(t for t in topos if TOPOLOGY_MODE[t] in modes)

    def topology_domain(p, ctx):
        return frozenset(legal(p))

    params = pool_schema(topology_key(WEIGHTS), (), (), (), parameters_pool(WEIGHTS))
    root = params.axes[0]

    def topology_default(p, ctx):
        allowed = legal(p)
        d = root.default(p, ctx)
        return d if d in allowed else (allowed[0] if allowed else d)

    guarded = replace(root, domain=topology_domain, default=topology_default)
    params = replace(params, axes=(guarded,) + tuple(params.axes[1:]))
    return Kernel(
        identity=KernelSchema(name="MVU", interfaces=ifaces),
        pool=(backend,),
        sub_schemas=(params,),
    )


def _mvu_ctx():
    return Context(
        shapes={"inp": (1, 128), "weights": (128, 64), "out": (1, 64)},
        datatypes={"inp": DataType["INT8"], "weights": DataType["INT8"], "out": DataType["INT32"]},
        initializers={"weights": np.ones((128, 64), dtype=np.float32)},
        fpgapart=VERSAL,
    )


def test_stream_only_backend_rejects_embedded_at_resolve():
    # End-to-end: a stream-only backend makes the embedded (constant) topology illegal at
    # resolve — the guard filters the domain, so pinning embedded is rejected.
    k = _restricted_kernel({WEIGHTS: {STREAM}})
    ctx = _mvu_ctx()
    # decoupled (stream) is in-domain and resolves.
    ok = k.configure(ctx, {"SIMD": 16, "PE": 4, topology_key(WEIGHTS): DECOUPLED})
    assert not isinstance(ok, Illegal), getattr(ok, "reasons", None)
    # embedded (constant) is filtered out of the domain -> pinning it is illegal.
    bad = k.configure(ctx, {"SIMD": 16, "PE": 4, topology_key(WEIGHTS): EMBEDDED})
    assert isinstance(bad, Illegal)


def test_permissive_backend_default_is_a_stream_topology_when_only_stream_offered():
    # With a stream-only restriction the pool's first-registered default (embedded) is out
    # of domain; resolve must fall to a legal in-domain default (decoupled), not crash.
    k = _restricted_kernel({WEIGHTS: {STREAM}})
    r = k.configure(_mvu_ctx(), {"SIMD": 16, "PE": 4})
    assert not isinstance(r, Illegal), getattr(r, "reasons", None)
    assert r[topology_key(WEIGHTS)] == DECOUPLED


def test_backend_consumes_coerced_to_frozenset():
    b = Backend(name="x", consumes={WEIGHTS: {STREAM, CONSTANT}})
    assert isinstance(b.consumes[WEIGHTS], frozenset)
    # A backend that declares nothing is permissive (empty dict, not a restriction).
    assert Backend(name="y").consumes == {}


# ---------------------------------------------------------------------------
# Per-interface namespacing — a SECOND parameter interface composes with no collision.
# ---------------------------------------------------------------------------


def test_two_interfaces_do_not_collide():
    # The seam the increment opens: compose the parameters pool for two distinct interface
    # names and confirm their point keys are disjoint (no shared `parameters.*` clash).
    ws = pool_schema(topology_key("weights"), (), (), (), parameters_pool("weights"))
    ts = pool_schema(topology_key("thresholds"), (), (), (), parameters_pool("thresholds"))
    w_keys = set(ws.axis_names)
    t_keys = set(ts.axis_names)
    assert topology_key("weights") in w_keys
    assert topology_key("thresholds") in t_keys
    # No key of one interface appears in the other (disjoint namespaces).
    assert w_keys.isdisjoint(t_keys)
