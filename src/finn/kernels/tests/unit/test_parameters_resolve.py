############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Resolve tests for the ``parameters`` pool (storage topologies) STANDALONE.

The parameters subsystem is a second selection pool (root axis ``topology``) that an
op composes in. These tests exercise it in isolation: topology selection, guarded
per-topology axes, and the self-contained URAM device gate. Cross-coordinate couplings
(memstream geometry + the pumpedMemory/fold gate) are the composing op's job and are
tested at the MVAU-composition level, not here — this schema resolves with no compute
context, which is itself the property under test (a param-free / fold-free resolve).
"""

import pytest

from finn.kernels.space import AbsentAxisError, Context, Illegal, resolve
from finn.kernels.dataflow.memory import (
    DECOUPLED,
    EMBEDDED,
    WEIGHTS,
    parameters_pool,
    parameters_schema,
)
from finn.kernels.model.param_names import (
    param_stream_width_key,
    pumped_memory_key,
    ram_style_key,
    runtime_writeable_key,
    sources_key,
    topology_key,
)

# The parameters pool is composed per parameter interface; standalone tests use ``weights``
# (the default/only live interface), so every point key is ``parameters.weights.*``.
TOPOLOGY = topology_key(WEIGHTS)
RAM_STYLE = ram_style_key(WEIGHTS)
RUNTIME_WRITEABLE = runtime_writeable_key(WEIGHTS)
PUMPED_MEMORY = pumped_memory_key(WEIGHTS)
SOURCES = sources_key(WEIGHTS)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"
ULTRASCALE = "xcku040-ffva1156-2-e"  # not Versal


def _ctx(part=VERSAL):
    return Context(fpgapart=part)


def test_pool_has_the_two_topologies_in_order():
    names = [b.name for b in parameters_pool()]
    assert names == [EMBEDDED, DECOUPLED]  # embedded first → pool default


def test_default_topology_is_embedded_with_no_delivery_axes():
    r = resolve(parameters_schema(), _ctx(), {})
    assert r[TOPOLOGY] == EMBEDDED
    # embedded owns no free axes — the decoupled selection axes are absent
    assert RAM_STYLE not in r
    assert RUNTIME_WRITEABLE not in r
    assert PUMPED_MEMORY not in r
    with pytest.raises(AbsentAxisError):
        _ = r[RAM_STYLE]


def test_decoupled_exposes_its_selection_axes():
    r = resolve(
        parameters_schema(),
        _ctx(),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "block", RUNTIME_WRITEABLE: 1, PUMPED_MEMORY: 1},
    )
    assert r[TOPOLOGY] == DECOUPLED
    assert r[RAM_STYLE] == "block"
    assert r[RUNTIME_WRITEABLE] == 1
    assert r[PUMPED_MEMORY] == 1


def test_decoupled_axes_absent_under_embedded():
    """Selecting embedded guards out the decoupled axes; assigning one is illegal
    (the guarded-axis discipline, across the pool boundary)."""
    r = resolve(parameters_schema(), _ctx(), {TOPOLOGY: EMBEDDED, RAM_STYLE: "block"})
    assert isinstance(r, Illegal)


def test_uram_gate_fires_on_non_versal_without_runtime_writeable():
    """THE combination gate: URAM weights on a non-Versal part require
    runtime_writeable_weights=1 (relocated verbatim from MVAU shared)."""
    r = resolve(
        parameters_schema(),
        _ctx(ULTRASCALE),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 0},
    )
    assert isinstance(r, Illegal)
    assert any("runtime_writeable" in reason for reason in r.reasons)


def test_uram_gate_satisfied_when_runtime_writeable():
    r = resolve(
        parameters_schema(),
        _ctx(ULTRASCALE),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 1},
    )
    assert not isinstance(r, Illegal)
    assert r[RAM_STYLE] == "ultra"


def test_uram_on_versal_needs_no_runtime_writeable():
    """On Versal, URAM is fine without runtime-writable (the gate is device-specific)."""
    r = resolve(
        parameters_schema(),
        _ctx(VERSAL),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 0},
    )
    assert not isinstance(r, Illegal)


def test_selected_topology_sources_on_point():
    """The pool exposes the selected topology's RTL sources under its namespaced
    sources key — decoupled ships the memstream HDL, embedded ships none."""
    r_emb = resolve(parameters_schema(), _ctx(), {TOPOLOGY: EMBEDDED})
    assert r_emb[SOURCES] == ()
    r_dec = resolve(parameters_schema(), _ctx(), {TOPOLOGY: DECOUPLED})
    assert "memstream_axi.sv" in r_dec[SOURCES]


# =============================================================================
# Composition into MVAU — the parameters pool merged into a real op schema, with
# the cross-coordinate couplings firing (Phase 2 strangler result).
# =============================================================================


def _mvau_ctx(part=VERSAL):
    import numpy as np
    from qonnx.core.datatype import DataType

    from finn.kernels.space import Context as _C

    w = np.random.RandomState(0).randint(-7, 7, size=(6, 8)).astype(np.float32)
    return _C(
        shapes={"weights": (6, 8), "inp": (1, 6), "out": (1, 8)},
        datatypes={"weights": DataType["INT8"], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=part,
        clk_ns=5.0,
    )


def test_mvau_schema_carries_both_coordinates():
    """The composed MVAU schema has BOTH the compute pool (implementation) and the
    parameters pool (parameters.topology) — two selection surfaces, one point."""
    from finn.kernels.compute.mvau import mvau_schema

    r = resolve(
        mvau_schema(),
        _mvau_ctx(),
        {"backend": "mvau_hls", "PE": 2, "SIMD": 2, "resType": "lut"},
    )
    assert not isinstance(r, Illegal)
    assert r.backend == "mvau_hls"
    assert r[TOPOLOGY] == EMBEDDED  # pool default
    # the namespaced sources of BOTH pools coexist
    assert "sources" in r and SOURCES in r


def test_mvau_weight_stream_width_coupling():
    """The cross-coordinate parameters.weights.stream_width derived: 0 for embedded (baked
    in), PE*SIMD*wbits for decoupled — reads BOTH topology and the compute fold. Namespaced
    per interface (``parameters.weights.stream_width``), replacing the old un-namespaced
    ``weight_stream_width`` global."""
    from finn.kernels.compute.mvau import mvau_schema

    swk = param_stream_width_key(WEIGHTS)
    base = {"backend": "mvau_hls", "PE": 2, "SIMD": 2, "resType": "lut"}
    r_emb = resolve(mvau_schema(), _mvau_ctx(), {**base, TOPOLOGY: EMBEDDED})
    assert r_emb[swk] == 0
    r_dec = resolve(mvau_schema(), _mvau_ctx(), {**base, TOPOLOGY: DECOUPLED})
    assert r_dec[swk] == 2 * 2 * 8  # PE*SIMD*wbits(INT8)


def test_mvau_pumped_memory_fold_gate_fires():
    """The cross-coordinate pumpedMemory gate: PE==SIMD==1 with pumpedMemory is
    illegal (contributed at compose time, reads parameters axis + compute fold)."""
    from finn.kernels.compute.mvau import mvau_schema

    r = resolve(
        mvau_schema(),
        _mvau_ctx(),
        {
            "backend": "mvau_hls",
            "PE": 1,
            "SIMD": 1,
            "resType": "lut",
            TOPOLOGY: DECOUPLED,
            PUMPED_MEMORY: 1,
        },
    )
    assert isinstance(r, Illegal)
    assert any("pumpedMemory" in reason for reason in r.reasons)
