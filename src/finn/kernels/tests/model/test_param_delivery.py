############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Parameter-feed contract: consumption-mode delivery + fold depth (D1-D7).

The behavioral anchor for the supply spine. "embedded" is the ``embedded`` mem_mode: the
compute core bakes the parameter in, so there is no stream and no demand. A compute
Backend declares which modes it accepts (``mem_modes``) per interface; the
delivery-topology domain is filtered to matching modes. Demand is sized from resolved
geometry (parallelism/elem_bits/depth) under decoupled mode, None under embedded mode.
``weight_fold_depth``/``threshold_fold_depth`` are one topology-independent quantity,
equal embedded and decoupled.

D1 waterfall order · D2 demand sizing/None · D3 topology-mode guard + re-guarded default ·
D4 memory geometry · D5 namespacing · D6 embedded constant fallback · D7 fold_depth.
"""

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.point import Illegal
from finn.kernels.engine.resolve import resolve
from finn.kernels.model.backend import Backend, pool_space, ports_from
from finn.kernels.model.fold_depth import threshold_fold_depth, weight_fold_depth
from finn.kernels.model.kernel import InterfaceSchema
from .synthetic import synthetic_op
from finn.kernels.model.param_names import (
    EMBEDDED as EMBEDDED_MODE,
    DECOUPLED as DECOUPLED_MODE,
    demand_key,
    depth_key,
    topology_key,
    width_key,
)
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL
from finn.kernels.dataflow.parameters import WEIGHTS, parameters_pool
from finn.kernels.dataflow.parameters.names import DECOUPLED, EMBEDDED

VERSAL = "xcvc1902-vsva2197-2MP-e-S"
THRESHOLDS = "thresholds"
TOPOLOGY = topology_key(WEIGHTS)


def _topo_mode(name, iface=WEIGHTS):
    return {b.name: b.mem_mode for b in parameters_pool(iface)}[name]


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
    from finn.kernels.compute.mvau import mvau_space

    base = {"backend": "mvau_hls", "PE": 2, "SIMD": 2, "resType": "lut"}
    base.update(assignment)
    return resolve(mvau_space(), _mvau_ctx(), base)


# --- D6: embedded is constant mode; decoupled is stream ---------------------


def test_embedded_is_constant_decoupled_is_stream():
    assert _topo_mode(EMBEDDED) == EMBEDDED_MODE
    assert _topo_mode(DECOUPLED) == DECOUPLED_MODE


# --- D2: demand sizing / None ----------------------------------------------


def test_constant_mode_has_no_demand():
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


# --- D3: topology-mode guard + re-guarded default --------------------------


def test_permissive_backend_keeps_both_topologies():
    assert not isinstance(_resolve_mvau({}), Illegal)
    assert not isinstance(_resolve_mvau({topology_key(WEIGHTS): DECOUPLED}), Illegal)


def test_consumes_stream_only_domain_excludes_embedded():
    backend = Backend(
        name="stream_only",
        ports=ports_from(
            stream={"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]},
            mem_modes={WEIGHTS: {DECOUPLED_MODE}},
        ),
    )
    legal = {
        b.name for b in parameters_pool(WEIGHTS) if b.mem_mode in backend.mem_modes[WEIGHTS]
    }
    assert EMBEDDED not in legal
    assert DECOUPLED in legal


def _restricted_kernel(mem_modes):
    ifaces = (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
        InterfaceSchema("weights", Direction.IN, block=[FULL, FULL]),
        InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
    )
    backend = Backend(
        name="core",
        ports=ports_from(
            stream={"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]},
            mem_modes=mem_modes,
        ),
    )
    return synthetic_op(interfaces=ifaces, pool=(backend,), name="MVU",
    )


def _mvu_ctx():
    return Context(
        shapes={"inp": (1, 128), "weights": (128, 64), "out": (1, 64)},
        datatypes={"inp": DataType["INT8"], "weights": DataType["INT8"], "out": DataType["INT32"]},
        initializers={"weights": np.ones((128, 64), dtype=np.float32)},
        fpgapart=VERSAL,
    )


def test_stream_only_backend_rejects_embedded_at_resolve():
    k = _restricted_kernel({WEIGHTS: {DECOUPLED_MODE}})
    ctx = _mvu_ctx()
    ok = k.configure(
        ctx, {"backend": "core", "SIMD": 16, "PE": 4, topology_key(WEIGHTS): DECOUPLED}
    )
    assert not isinstance(ok, Illegal), getattr(ok, "reasons", None)
    bad = k.configure(
        ctx, {"backend": "core", "SIMD": 16, "PE": 4, topology_key(WEIGHTS): EMBEDDED}
    )
    assert isinstance(bad, Illegal)


def test_default_falls_to_legal_topology_when_embedded_out_of_domain():
    # With a stream-only restriction embedded is out of domain; resolve must fall to a
    # legal in-domain default (decoupled), not crash.
    k = _restricted_kernel({WEIGHTS: {DECOUPLED_MODE}})
    r = k.configure(_mvu_ctx(), {"backend": "core", "SIMD": 16, "PE": 4})
    assert not isinstance(r, Illegal), getattr(r, "reasons", None)
    assert r[topology_key(WEIGHTS)] == DECOUPLED


def test_backend_mem_modes_coerced_to_frozenset():
    b = Backend(name="x", ports=ports_from(mem_modes={WEIGHTS: {DECOUPLED_MODE, EMBEDDED_MODE}}))
    assert isinstance(b.mem_modes[WEIGHTS], frozenset)
    assert Backend(name="y").mem_modes == {}  # declares nothing → permissive


# --- D5: per-interface namespacing ------------------------------------------


def test_two_interfaces_do_not_collide():
    ws = pool_space(topology_key("weights"), (), (), (), parameters_pool("weights"))
    ts = pool_space(topology_key("thresholds"), (), (), (), parameters_pool("thresholds"))
    w_keys = set(ws.axis_names)
    t_keys = set(ts.axis_names)
    assert topology_key("weights") in w_keys
    assert topology_key("thresholds") in t_keys
    assert w_keys.isdisjoint(t_keys)


# --- D7: fold_depth is one topology-independent quantity --------------------


def _fd_ctx(mw=6, mh=8):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": DataType["INT8"], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def _thresh_ctx(mw=6, mh=8, steps=7):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    thr = np.sort(
        np.random.RandomState(1).randint(0, 100, size=(mh, steps)).astype(np.float32), axis=1
    )
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh), "thresholds": (mh, steps)},
        datatypes={
            "weights": DataType["INT4"], "inp": DataType["INT4"],
            "out": DataType["INT16"], "thresholds": DataType["INT16"],
        },
        initializers={"weights": w, "thresholds": thr},
        fpgapart=VERSAL, clk_ns=5.0,
    )


def test_weight_fold_depth_matches_wmem_decoupled():
    from finn.kernels.compute.mvau import MVAU_DSP_SOFTVEC, mvau_space

    ctx = _fd_ctx()
    p = resolve(mvau_space(), ctx, {
        "backend": MVAU_DSP_SOFTVEC, "PE": 2, "SIMD": 2,
        "resType": "dsp", TOPOLOGY: DECOUPLED,
    })
    # WMEM = MW*MH/(PE*SIMD) = 6*8/(2*2) = 12; the decoupled depth_key traces to same math.
    assert weight_fold_depth(p, ctx, WEIGHTS) == 12
    assert weight_fold_depth(p, ctx, WEIGHTS) == p[depth_key(WEIGHTS)]


def test_weight_fold_depth_matches_wmem_embedded():
    from finn.kernels.compute.mvau import MVAU_HLS, mvau_space

    ctx = _fd_ctx()
    p = resolve(mvau_space(), ctx, {
        "backend": MVAU_HLS, "PE": 2, "SIMD": 2,
        "resType": "lut", TOPOLOGY: EMBEDDED,
    })
    # embedded HLS has no depth_key, but the topology-independent query still works —
    # equal to the decoupled value (D7).
    assert weight_fold_depth(p, ctx, WEIGHTS) == 12


def test_threshold_fold_depth_matches_tmem_present():
    from finn.kernels.compute.mvau import MVAU_HLS, mvau_space

    ctx = _thresh_ctx()
    p = resolve(mvau_space(), ctx, {
        "backend": MVAU_HLS, "PE": 2, "SIMD": 2,
        "resType": "lut", TOPOLOGY: EMBEDDED,
    })
    assert threshold_fold_depth(p, ctx, THRESHOLDS) == 4  # MH//PE = 8//2


def test_threshold_fold_depth_zero_when_absent():
    from finn.kernels.compute.mvau import MVAU_HLS, mvau_space

    ctx = _fd_ctx()  # no thresholds
    p = resolve(mvau_space(), ctx, {
        "backend": MVAU_HLS, "PE": 2, "SIMD": 2,
        "resType": "lut", TOPOLOGY: EMBEDDED,
    })
    assert threshold_fold_depth(p, ctx, THRESHOLDS) == 0
