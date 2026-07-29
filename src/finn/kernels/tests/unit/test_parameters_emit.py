############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Emit tests: the ``decoupled`` (memstream) parameters topology.

Exercises the memstream weight-delivery emit through the COMPOSED MVAU schema — the
wrapper ``.v`` (geometry-parameterized), the ``memblock.dat`` weight file, and the
static memstream HDL. Hermetic: a dict-built Context, no graph. This is the delivery
HALF that the compute-core emit deferred; validated byte-for-byte against FINN in the
Docker differential test (``diff_mvau_emit_vs_finn.py``).
"""

import re

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import Artifacts, Context, emit_point, resolve
from finn.kernels.ops.mvau import MVAU_HLS, mvau_schema
from finn.kernels.ops.parameters import DECOUPLED, WEIGHTS, parameters_pool
from finn.kernels.ops.parameters.emit_memstream import emit_memstream
from finn.kernels.model.param_names import (
    depth_key,
    pumped_memory_key,
    ram_style_key,
    runtime_writeable_key,
    sets_key,
    topology_key,
    width_key,
)

# Composed for the ``weights`` interface -> ``parameters.weights.*`` point keys.
PARAM_DEPTH = depth_key(WEIGHTS)
PARAM_SETS = sets_key(WEIGHTS)
PARAM_WIDTH = width_key(WEIGHTS)
PUMPED_MEMORY = pumped_memory_key(WEIGHTS)
RAM_STYLE = ram_style_key(WEIGHTS)
RUNTIME_WRITEABLE = runtime_writeable_key(WEIGHTS)
TOPOLOGY = topology_key(WEIGHTS)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"
ULTRASCALE = "xcku040-ffva1156-2-e"
_UNFILLED = re.compile(r"\$[A-Z][A-Z0-9_]*\$")


def _ctx(part=VERSAL, mw=6, mh=8, wdt="INT8"):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": DataType[wdt], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=part,
        clk_ns=5.0,
    )


def _decoupled_point(ctx, pe=2, simd=2, **extra):
    a = {
        "backend": MVAU_HLS,
        "PE": pe,
        "SIMD": simd,
        "resType": "lut",
        TOPOLOGY: DECOUPLED,
        RAM_STYLE: "block",
    }
    a.update(extra)
    return resolve(mvau_schema(), ctx, a)


# --- geometry couplings -----------------------------------------------------


def test_memstream_geometry_derives_from_fold():
    ctx = _ctx()
    r = _decoupled_point(ctx, pe=2, simd=2)
    assert r[PARAM_DEPTH] == 12  # WMEM = MW*MH/(PE*SIMD) = 6*8/4
    assert r[PARAM_WIDTH] == 32  # roundup(PE*SIMD*wbits, 8) = roundup(2*2*8, 8)
    assert r[PARAM_SETS] == 1  # no MLO


def test_geometry_absent_for_embedded():
    from finn.kernels.ops.parameters import EMBEDDED

    ctx = _ctx()
    r = resolve(
        mvau_schema(),
        ctx,
        {"backend": MVAU_HLS, "PE": 2, "SIMD": 2, "resType": "lut", TOPOLOGY: EMBEDDED},
    )
    # geometry derived are present-but-None under embedded (no streamer)
    assert r[PARAM_DEPTH] is None
    assert r[PARAM_WIDTH] is None


# --- wrapper .v golden ------------------------------------------------------


def test_memstream_wrapper_golden():
    ctx = _ctx()
    arts = emit_memstream(_decoupled_point(ctx), ctx)
    assert isinstance(arts, Artifacts)
    assert len(arts.generated) == 1
    c = arts.generated[0].content()
    assert "parameter  SETS = 1" in c
    assert "parameter  DEPTH = 12" in c
    assert "parameter  WIDTH = 32" in c
    assert 'INIT_FILE = "memblock.dat"' in c
    assert 'RAM_STYLE = "block"' in c
    assert not _UNFILLED.search(c)  # every $SLOT$ filled ($clog2 is not a slot)


def test_memstream_ships_static_hdl():
    ctx = _ctx()
    arts = emit_memstream(_decoupled_point(ctx), ctx)
    static = [s.resource.split("/")[-1] for s in arts.static_files]
    assert "memstream_axi.sv" in static
    assert "memstream.sv" in static
    assert "axilite.sv" in static


# --- memblock.dat -----------------------------------------------------------


def test_memblock_dat_shape_and_hexwidth():
    ctx = _ctx()  # PE=SIMD=2, INT8 -> WIDTH=32 bits -> 8 hex chars; WMEM=12 lines
    arts = emit_memstream(_decoupled_point(ctx), ctx)
    dat = [d for d in arts.data_files if d.filename == "memblock.dat"]
    assert len(dat) == 1
    lines = dat[0].content.strip().split("\n")
    assert len(lines) == 12  # WMEM
    assert all(len(ln) == 8 for ln in lines)  # 32 bits / 4 = 8 hex chars
    assert all(re.fullmatch(r"[0-9a-fA-F]+", ln) for ln in lines)


def test_pumped_memory_splits_each_word_in_two():
    ctx = _ctx()
    arts = emit_memstream(_decoupled_point(ctx, pe=2, simd=2, **{PUMPED_MEMORY: 1}), ctx)
    dat = [d for d in arts.data_files if d.filename == "memblock.dat"][0]
    lines = dat.content.strip().split("\n")
    # pumped: 2x the entries, each half the hex width (8 -> 4)
    assert len(lines) == 24
    assert all(len(ln) == 4 for ln in lines)


# --- URAM-on-non-Versal blanks the init file (weights via AXI-lite) ---------


def test_uram_non_versal_blanks_init_file_and_omits_dat():
    ctx = _ctx(ULTRASCALE)
    r = _decoupled_point(ctx, **{RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 1})
    arts = emit_memstream(r, ctx)
    c = arts.generated[0].content()
    assert 'INIT_FILE = ""' in c
    # no .dat when the RAM is not file-initialized
    assert all(d.filename != "memblock.dat" for d in arts.data_files)


# --- dispatch via emit_point (the pool's own root axis) ---------------------


def test_emit_point_dispatches_parameters_pool():
    """emit_point routes to the topology's emit via the pool's root axis
    (parameters.topology), not the compute pool's 'implementation'."""
    ctx = _ctx()
    r = _decoupled_point(ctx)
    arts = emit_point(parameters_pool(), r, ctx, root=TOPOLOGY)
    assert isinstance(arts, Artifacts)
    assert arts.generated[0].filename.endswith("_memstream_wrapper.v")


# --- hermeticity ------------------------------------------------------------


def test_emit_needs_only_point_and_dict_context():
    ctx = _ctx()  # plain dicts
    arts = emit_memstream(_decoupled_point(ctx), ctx)
    assert arts.generated[0].content()
    assert not hasattr(ctx, "graph") and not hasattr(ctx, "model")
