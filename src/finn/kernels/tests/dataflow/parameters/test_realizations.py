############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Parameter-delivery realizations: the memory topologies (resolve + emit + serialize).

The ``parameters`` pool is a second selection pool (root axis ``topology``): embedded owns
no delivery axes (constant, baked); decoupled exposes ram_style/runtime_writeable/pumped
and derives memstream geometry (depth/width/sets) from the compute fold. The URAM device
gate is self-contained. Serialization: the unified ``layout`` produces the C++ initializer
and the memblock hex — asserted against KNOWN-GOOD fixed byte layouts (the old test embedded
verbatim reference impls as its oracle; those are replaced here with frozen expected bytes).
"""

import re

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.model.artifacts import Artifacts
from finn.kernels.engine.context import Context
from finn.kernels.engine.point import AbsentAxisError, Illegal
from finn.kernels.engine.resolve import resolve
from finn.kernels.compute.mvau import MVAU_HLS, mvau_space
from finn.kernels.dataflow.parameters import (
    DECOUPLED,
    EMBEDDED,
    WEIGHTS,
    parameters_pool,
    parameters_schema,
)
from finn.kernels.dataflow.parameters.emit_memstream import emit_memstream
from finn.kernels.dataflow.parameters.serialize import (
    CPP_HEADER,
    DAT_HEX,
    SerializedParam,
    layout,
    weight_constraint,
)
from finn.kernels.engine.param_datatype import ParamDatatype
from finn.kernels.model.param_names import (
    depth_key,
    param_stream_width_key,
    pumped_memory_key,
    ram_style_key,
    runtime_writeable_key,
    sets_key,
    sources_key,
    param_datatype_key,
    topology_key,
    width_key,
)

TOPOLOGY = topology_key(WEIGHTS)
RAM_STYLE = ram_style_key(WEIGHTS)
RUNTIME_WRITEABLE = runtime_writeable_key(WEIGHTS)
PUMPED_MEMORY = pumped_memory_key(WEIGHTS)
SOURCES = sources_key(WEIGHTS)
PARAM_DEPTH = depth_key(WEIGHTS)
PARAM_SETS = sets_key(WEIGHTS)
PARAM_WIDTH = width_key(WEIGHTS)
PARAM_DTYPE = param_datatype_key(WEIGHTS)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"
ULTRASCALE = "xcku040-ffva1156-2-e"  # not Versal
_UNFILLED = re.compile(r"\$[A-Z][A-Z0-9_]*\$")


def _ctx(part=VERSAL):
    return Context(fpgapart=part)


# ===========================================================================
# Standalone parameters-pool resolve.
# ===========================================================================


def test_pool_has_the_two_topologies_in_order():
    assert [b.name for b in parameters_pool()] == [EMBEDDED, DECOUPLED]  # embedded first → default


def test_default_topology_is_embedded_with_no_delivery_axes():
    r = resolve(parameters_schema(), _ctx(), {})
    assert r[TOPOLOGY] == EMBEDDED
    assert RAM_STYLE not in r
    assert RUNTIME_WRITEABLE not in r
    assert PUMPED_MEMORY not in r
    with pytest.raises(AbsentAxisError):
        _ = r[RAM_STYLE]


def test_decoupled_exposes_its_selection_axes():
    r = resolve(
        parameters_schema(), _ctx(),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "block", RUNTIME_WRITEABLE: 1, PUMPED_MEMORY: 1},
    )
    assert r[TOPOLOGY] == DECOUPLED
    assert r[RAM_STYLE] == "block"
    assert r[RUNTIME_WRITEABLE] == 1
    assert r[PUMPED_MEMORY] == 1


def test_decoupled_axes_absent_under_embedded():
    r = resolve(parameters_schema(), _ctx(), {TOPOLOGY: EMBEDDED, RAM_STYLE: "block"})
    assert isinstance(r, Illegal)


def test_uram_gate_fires_on_non_versal_without_runtime_writeable():
    r = resolve(
        parameters_schema(), _ctx(ULTRASCALE),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 0},
    )
    assert isinstance(r, Illegal)
    assert any("runtime_writeable" in reason for reason in r.reasons)


def test_uram_gate_satisfied_when_runtime_writeable():
    r = resolve(
        parameters_schema(), _ctx(ULTRASCALE),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 1},
    )
    assert not isinstance(r, Illegal)
    assert r[RAM_STYLE] == "ultra"


def test_uram_on_versal_needs_no_runtime_writeable():
    r = resolve(
        parameters_schema(), _ctx(VERSAL),
        {TOPOLOGY: DECOUPLED, RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 0},
    )
    assert not isinstance(r, Illegal)


def test_selected_topology_sources_on_point():
    assert resolve(parameters_schema(), _ctx(), {TOPOLOGY: EMBEDDED})[SOURCES] == ()
    r_dec = resolve(parameters_schema(), _ctx(), {TOPOLOGY: DECOUPLED})
    assert "memstream_axi.sv" in r_dec[SOURCES]


def test_embedded_topology_self_registers():
    # Both topologies come from the registry (import self-registration) — the pool being
    # non-empty and containing both names IS the registration proof.
    names = {b.name for b in parameters_pool()}
    assert {EMBEDDED, DECOUPLED} <= names


# ===========================================================================
# Composition into MVAU — geometry couplings fire.
# ===========================================================================


def _mvau_ctx(part=VERSAL, mw=6, mh=8, wdt="INT8"):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": DataType[wdt], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=part, clk_ns=5.0,
    )


def _decoupled_point(ctx, pe=2, simd=2, **extra):
    a = {"backend": MVAU_HLS, "PE": pe, "SIMD": simd, "resType": "lut",
         TOPOLOGY: DECOUPLED, RAM_STYLE: "block"}
    a.update(extra)
    return resolve(mvau_space(), ctx, a)


def test_memstream_geometry_derives_from_fold():
    r = _decoupled_point(_mvau_ctx(), pe=2, simd=2)
    assert r[PARAM_DEPTH] == 12  # WMEM = MW*MH/(PE*SIMD) = 6*8/4
    assert r[PARAM_WIDTH] == 32  # roundup(PE*SIMD*wbits, 8) = roundup(2*2*8, 8)
    assert r[PARAM_SETS] == 1  # no MLO


def test_geometry_absent_for_embedded():
    r = resolve(
        mvau_space(), _mvau_ctx(),
        {"backend": MVAU_HLS, "PE": 2, "SIMD": 2, "resType": "lut", TOPOLOGY: EMBEDDED},
    )
    assert r[PARAM_DEPTH] is None
    assert r[PARAM_WIDTH] is None


def test_stream_width_coupling_zero_embedded_sized_decoupled():
    swk = param_stream_width_key(WEIGHTS)
    base = {"backend": MVAU_HLS, "PE": 2, "SIMD": 2, "resType": "lut"}
    assert resolve(mvau_space(), _mvau_ctx(), {**base, TOPOLOGY: EMBEDDED})[swk] == 0
    assert resolve(mvau_space(), _mvau_ctx(), {**base, TOPOLOGY: DECOUPLED})[swk] == 2 * 2 * 8


def test_pumped_memory_fold_gate_fires():
    r = resolve(
        mvau_space(), _mvau_ctx(),
        {"backend": MVAU_HLS, "PE": 1, "SIMD": 1, "resType": "lut",
         TOPOLOGY: DECOUPLED, PUMPED_MEMORY: 1},
    )
    assert isinstance(r, Illegal)
    assert any("pumpedMemory" in reason for reason in r.reasons)


# ===========================================================================
# ParamDatatype AUTHORITY — what the storage OWNER publishes (B1).
# The owner publishes (value-optimized dtype, values_visible) — authority, not data.
# Trusted iff it has build-time value visibility: embedded always; decoupled iff static.
# ===========================================================================


def test_embedded_publishes_visible_narrowed_param_datatype():
    # Fixture weights are in -7..6 -> narrow to INT4 (graph dtype is INT8). Embedded always
    # sees its values, so it narrows AND authorizes narrowing.
    r = resolve(
        mvau_space(), _mvau_ctx(),
        {"backend": MVAU_HLS, "PE": 2, "SIMD": 2, "resType": "lut", TOPOLOGY: EMBEDDED},
    )
    desc = r[PARAM_DTYPE]
    assert isinstance(desc, ParamDatatype)
    assert desc.values_visible is True
    assert desc.dtype == DataType["INT4"]


def test_decoupled_static_publishes_visible_narrowed_param_datatype():
    # runtime_writeable=0 (default): owner has build-time visibility -> narrows, visible.
    desc = _decoupled_point(_mvau_ctx(), pe=2, simd=2)[PARAM_DTYPE]
    assert isinstance(desc, ParamDatatype)
    assert desc.values_visible is True
    assert desc.dtype == DataType["INT4"]


def test_decoupled_runtime_writeable_publishes_blind_envelope_param_datatype():
    # runtime_writeable=1: the host may overwrite cells post-build -> owner BLIND. It must
    # NOT narrow; it publishes the declared graph dtype (envelope) and withholds trust.
    desc = _decoupled_point(_mvau_ctx(wdt="INT8"), **{RUNTIME_WRITEABLE: 1})[PARAM_DTYPE]
    assert isinstance(desc, ParamDatatype)
    assert desc.values_visible is False
    assert desc.dtype == DataType["INT8"]  # graph dtype, NOT the INT4 narrowing


def test_param_datatype_none_on_standalone_resolve_without_tensor():
    # No compute context (bare Context, unwired interface): the ParamDatatype is present-but-None,
    # exactly like the geometry deriveds — the pool still resolves standalone.
    r_emb = resolve(parameters_schema(), _ctx(), {TOPOLOGY: EMBEDDED})
    assert r_emb[PARAM_DTYPE] is None
    r_dec = resolve(parameters_schema(), _ctx(), {TOPOLOGY: DECOUPLED})
    assert r_dec[PARAM_DTYPE] is None


# ===========================================================================
# Memstream emit.
# ===========================================================================


def test_memstream_wrapper_all_slots_filled():
    c = emit_memstream(_decoupled_point(_mvau_ctx()), _mvau_ctx()).generated[0].content()
    assert "parameter  SETS = 1" in c
    assert "parameter  DEPTH = 12" in c
    assert "parameter  WIDTH = 32" in c
    assert 'INIT_FILE = "memblock.dat"' in c
    assert 'RAM_STYLE = "block"' in c
    assert not _UNFILLED.search(c)


def test_memstream_ships_static_hdl():
    static = [s.resource.split("/")[-1] for s in emit_memstream(_decoupled_point(_mvau_ctx()), _mvau_ctx()).static_files]
    assert "memstream_axi.sv" in static
    assert "memstream.sv" in static
    assert "axilite.sv" in static


def test_memblock_dat_shape_and_hexwidth():
    ctx = _mvau_ctx()  # PE=SIMD=2, INT8 -> WIDTH=32 bits -> 8 hex chars; WMEM=12 lines
    arts = emit_memstream(_decoupled_point(ctx), ctx)
    dat = [d for d in arts.data_files if d.filename == "memblock.dat"]
    assert len(dat) == 1
    lines = dat[0].content.strip().split("\n")
    assert len(lines) == 12
    assert all(len(ln) == 8 and re.fullmatch(r"[0-9a-fA-F]+", ln) for ln in lines)


def test_pumped_memory_splits_each_word_in_two():
    ctx = _mvau_ctx()
    arts = emit_memstream(_decoupled_point(ctx, pe=2, simd=2, **{PUMPED_MEMORY: 1}), ctx)
    dat = [d for d in arts.data_files if d.filename == "memblock.dat"][0]
    lines = dat.content.strip().split("\n")
    assert len(lines) == 24  # 2x entries
    assert all(len(ln) == 4 for ln in lines)  # half the hex width


def test_uram_non_versal_blanks_init_file_and_omits_dat():
    ctx = _mvau_ctx(ULTRASCALE)
    arts = emit_memstream(_decoupled_point(ctx, **{RAM_STYLE: "ultra", RUNTIME_WRITEABLE: 1}), ctx)
    assert 'INIT_FILE = ""' in arts.generated[0].content()
    assert all(d.filename != "memblock.dat" for d in arts.data_files)


# ===========================================================================
# Serialize — asserted against KNOWN-GOOD fixed byte layouts (not embedded
# reference impls). Oracle: MW=MH=4, PE=SIMD=2, INT8, values -8..7 row-major.
# ===========================================================================

_MW = _MH = 4
_PE = _SIMD = 2
_WMEM = _MW * _MH // (_PE * _SIMD)  # 4
_INT8 = DataType["INT8"]
_FIXED_WEIGHTS = np.arange(_MW * _MH, dtype=np.float64).reshape(_MW, _MH) - 8.0

_EXPECTED_CPP = (
    '{{{ap_uint<16>("0xfcf8", 16), ap_uint<16>("0x0400", 16),\n'
    '   ap_uint<16>("0xfefa", 16), ap_uint<16>("0x0602", 16)},\n'
    '  {ap_uint<16>("0xfdf9", 16), ap_uint<16>("0x0501", 16),\n'
    '   ap_uint<16>("0xfffb", 16), ap_uint<16>("0x0703", 16)}}};'
)
_EXPECTED_DAT = "fdf9fcf8\n05010400\nfffbfefa\n07030602\n"
_EXPECTED_DAT_PUMPED = "fcf8\nfdf9\n0400\n0501\nfefa\nfffb\n0602\n0703\n"


def test_cpp_header_matches_known_good_layout():
    out = layout(_FIXED_WEIGHTS, weight_constraint(_PE, _SIMD, _WMEM, _INT8, _INT8, form=CPP_HEADER))
    assert isinstance(out, SerializedParam) and out.form == CPP_HEADER
    assert out.text == _EXPECTED_CPP


def test_dat_hex_matches_known_good_layout():
    out = layout(
        _FIXED_WEIGHTS,
        weight_constraint(_PE, _SIMD, _WMEM, _INT8, _INT8, form=DAT_HEX, decoupled_pe_flip=True),
    )
    assert out.form == DAT_HEX
    assert out.text == _EXPECTED_DAT


def test_dat_hex_pumped_splits_each_word():
    out = layout(
        _FIXED_WEIGHTS,
        weight_constraint(
            _PE, _SIMD, _WMEM, _INT8, _INT8, form=DAT_HEX, decoupled_pe_flip=True, pumped_split=True
        ),
    )
    assert out.text == _EXPECTED_DAT_PUMPED


def test_unknown_form_raises():
    with pytest.raises(ValueError):
        layout(_FIXED_WEIGHTS, weight_constraint(1, 1, 16, _INT8, _INT8, form="bogus"))
