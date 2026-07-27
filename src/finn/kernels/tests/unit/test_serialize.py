############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The unified parameter serializer ``layout`` (Arc 2b, F8).

Locks the byte-equivalence the unification claims: ``layout`` reproduces — byte-for-byte —
the pre-refactor emit-side serializers (the ``params.h`` C++ initializer body and the
``memblock.dat`` hex text) across dtypes (incl. BIPOLAR), fold shapes, and pumpedMemory.
The two emits now call this ONE function, so the embedded and decoupled forms cannot
diverge. The reference implementations below are the old code, verbatim.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string

from finn.kernels.ops.parameters.serialize import (
    CPP_HEADER,
    DAT_HEX,
    SerializedParam,
    layout,
    weight_constraint,
)


# --- verbatim pre-refactor reference serializers ------------------------------


def _old_hw_weight_tensor(weights, pe, simd, wmem, wdt):
    ret = weights.T
    if wdt == DataType["BIPOLAR"]:
        ret = (ret + 1) / 2
    ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
    ret = ret.reshape(1, pe, wmem, simd)
    return np.flip(ret, axis=-1)


def _old_params_body(weights, pe, simd, wmem, wdt, export_wdt):
    tensor = _old_hw_weight_tensor(weights, pe, simd, wmem, wdt)
    return numpy_to_hls_code(tensor, export_wdt, "weights", True, True)


def _old_memblock(weights, pe, simd, wmem, wdt, export_wdt, pumped):
    wt = _old_hw_weight_tensor(weights, pe, simd, wmem, wdt)
    unflipped = np.transpose(wt, (0, 2, 1, 3))
    pe_flipped = np.flip(unflipped, axis=-2).reshape(1, -1, pe * simd).copy()
    w = pe * simd * export_wdt.bitwidth()
    wp = roundup_to_integer_multiple(w, 4)
    packed = pack_innermost_dim_as_hex_string(pe_flipped, export_wdt, wp, prefix="").flatten().copy()
    if pumped:
        split = []
        for x in packed:
            split.append(x[len(x) // 2:])
            split.append(x[: len(x) // 2])
        packed = split
    return "".join(str(v) + "\n" for v in packed)


def _weights(mw, mh, wdt, seed):
    rng = np.random.default_rng(seed)
    if wdt == DataType["BIPOLAR"]:
        w = rng.integers(0, 2, size=(mw, mh)) * 2 - 1
    else:
        w = rng.integers(wdt.min(), wdt.max() + 1, size=(mw, mh))
    return w.astype(np.float64)


_SHAPES = [(6, 8, 2, 2), (4, 4, 1, 1), (8, 4, 4, 2), (12, 6, 3, 3), (2, 2, 1, 1)]
_DTYPES = ["INT8", "INT4", "UINT2", "BIPOLAR", "INT2"]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cpp_header_matches_old_serializer(shape, dtype):
    mw, mh, pe, simd = shape
    wmem = mw * mh // (pe * simd)
    wdt = DataType[dtype]
    export = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
    w = _weights(mw, mh, wdt, seed=1)

    old = _old_params_body(w, pe, simd, wmem, wdt, export)
    new = layout(w, weight_constraint(pe, simd, wmem, wdt, export, form=CPP_HEADER))
    assert isinstance(new, SerializedParam) and new.form == CPP_HEADER
    assert new.text == old


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("pumped", [False, True])
def test_dat_hex_matches_old_serializer(shape, dtype, pumped):
    mw, mh, pe, simd = shape
    if pumped and pe * simd == 1:
        pytest.skip("pumpedMemory with parallelism=1 is a gated-illegal config")
    wmem = mw * mh // (pe * simd)
    wdt = DataType[dtype]
    export = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
    w = _weights(mw, mh, wdt, seed=2)

    old = _old_memblock(w, pe, simd, wmem, wdt, export, pumped)
    new = layout(
        w,
        weight_constraint(
            pe, simd, wmem, wdt, export,
            form=DAT_HEX, decoupled_pe_flip=True, pumped_split=pumped,
        ),
    )
    assert new.form == DAT_HEX
    assert new.text == old


def test_unknown_form_raises():
    w = _weights(4, 4, DataType["INT8"], seed=3)
    with pytest.raises(ValueError):
        layout(w, weight_constraint(1, 1, 16, DataType["INT8"], DataType["INT8"], form="bogus"))
