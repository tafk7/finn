# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor

from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)


@pytest.mark.util
@pytest.mark.parametrize("simd", [1, 2, 4, 8])
@pytest.mark.parametrize("reverse_inner", [False, True])
@pytest.mark.parametrize("reverse_endian", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        DataType["BINARY"],
        DataType["BIPOLAR"],
        DataType["TERNARY"],
        DataType["UINT2"],
        DataType["UINT3"],
        DataType["INT7"],
        DataType["INT8"],
        DataType["INT22"],
        DataType["INT32"],
        DataType["UINT7"],
        DataType["UINT8"],
        DataType["UINT15"],
        DataType["UINT32"],
        DataType["UINT63"],
        DataType["FIXED<7,4>"],
        DataType["FIXED<9,6>"],
        DataType["FIXED<31,23>"],
        DataType["FLOAT32"],
    ],
)
def test_driver_pack_unpack(dtype, reverse_inner, reverse_endian, simd):
    folded_shape = (10, 8 // simd, simd)  # N H W FOLD SIMD
    input = gen_finn_dt_tensor(dtype, folded_shape)
    input_packed = finnpy_to_packed_bytearray(input, dtype, reverse_inner, reverse_endian)
    input_unpacked = packed_bytearray_to_finnpy(
        np.ascontiguousarray(input_packed), dtype, folded_shape, reverse_inner, reverse_endian
    )

    assert input_unpacked.dtype == np.float32
    # Check shape
    assert input.shape == input_unpacked.shape
    # Check values
    assert np.all(input == input_unpacked)
