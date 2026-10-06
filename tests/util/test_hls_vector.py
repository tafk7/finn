# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import subprocess
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor

from finn.util.basic import CppBuilder, make_build_dir, robust_rmtree
from finn.util.resources import resource_path


@pytest.mark.util
@pytest.mark.parametrize(
    "dtype",
    [
        DataType["BINARY"],
        DataType["UINT8"],
        DataType["INT32"],
        DataType["FIXED<9,6>"],
        DataType["FLOAT32"],
    ],
)
@pytest.mark.parametrize("test_shape", [(1, 2, 4), (1, 1, 64), (2, 64)])
@pytest.mark.vivado
def test_npy2vectorstream(test_shape, dtype, hls_toolchain):
    ndarray = gen_finn_dt_tensor(dtype, test_shape)
    test_dir = make_build_dir(prefix="test_npy2vectorstream_")
    shape = ndarray.shape
    elem_hls_type = dtype.get_hls_datatype_str()
    vLen = shape[-1]
    npy_in = test_dir + "/in.npy"
    npy_out = test_dir + "/out.npy"
    # restrict the np datatypes we can handle
    npyt_to_ct = {
        "float32": "float",
        "float64": "double",
        "int8": "int8_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "uint8": "uint8_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
    }
    npy_type = npyt_to_ct[str(ndarray.dtype)]
    shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")
    test_app_string = []
    test_app_string += ["#include <cstddef>"]
    test_app_string += ["#define AP_INT_MAX_W 8191"]
    test_app_string += ['#include "ap_int.h"']
    test_app_string += ['#include "stdint.h"']
    test_app_string += ['#include "hls_stream.h"']
    test_app_string += ['#include "hls_vector.h"']
    test_app_string += ['#include "cnpy.h"']
    test_app_string += ['#include "npy2vectorstream.hpp"']
    test_app_string += ["int main(int argc, char *argv[]) {"]
    test_app_string += ["hls::stream<hls::vector<%s, %d>> teststream;" % (elem_hls_type, vLen)]
    test_app_string += [
        'npy2vectorstream<%s, %s, %d>("%s", teststream);' % (elem_hls_type, npy_type, vLen, npy_in)
    ]
    test_app_string += [
        'vectorstream2npy<%s, %s, %d>(teststream, %s, "%s");'
        % (elem_hls_type, npy_type, vLen, shape_cpp_str, npy_out)
    ]
    test_app_string += ["return 0;"]
    test_app_string += ["}"]
    with open(test_dir + "/test.cpp", "w") as f:
        f.write("\n".join(test_app_string))
    builder = CppBuilder(toolchain=hls_toolchain)
    builder.append_includes(
        [
            "-I" + str(hls_toolchain.hls_installation() / "include"),
            "-I" + resource_path("custom_hls"),
            "--std=c++17",
            "-lz",
        ]
    )
    builder.append_sources(test_dir + "/test.cpp")
    builder.append_sources(resource_path("custom_hls", "cnpy.cpp"))
    builder.set_executable_path(test_dir + "/test_npy2vectorstream")
    builder.build(test_dir)
    # make copy before saving the array
    ndarray = ndarray.copy()
    np.save(npy_in, ndarray)
    subprocess.check_call(["./test_npy2vectorstream"], cwd=test_dir)
    produced = np.load(npy_out)
    assert (produced == ndarray).all()
    robust_rmtree(test_dir)
