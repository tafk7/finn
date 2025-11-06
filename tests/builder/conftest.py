# Copyright (c) 2025, AMD
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

"""Fixtures for builder integration tests."""

import pytest
import tempfile
import shutil
import torch
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.util.test import get_test_model_trained


@pytest.fixture
def temp_build_dir():
    """Create a temporary build directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="finn_test_build_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Minimal Model Fixtures (for fast unit-style tests)
# =============================================================================


@pytest.fixture
def minimal_fc_model(temp_build_dir):
    """
    Minimal single-layer FC model for fast unit testing.

    Creates a simple MatMul layer:
    - Input: [1, 8]
    - Weights: [8, 4]
    - Output: [1, 4]

    Fast execution, no external model loading.
    """
    # Create minimal ONNX graph with single MatMul
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])

    matmul_node = helper.make_node("MatMul", ["inp", "weights"], ["outp"])

    graph = helper.make_graph(
        nodes=[matmul_node],
        name="minimal_fc",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="minimal-fc-model")
    model = ModelWrapper(model)

    # Set datatypes and initializers
    model.set_tensor_datatype("inp", DataType["INT4"])
    model.set_tensor_datatype("weights", DataType["INT4"])
    model.set_tensor_datatype("outp", DataType["INT32"])

    # Create simple weights
    W = np.random.randint(-8, 7, (8, 4)).astype(np.float32)
    model.set_initializer("weights", W)

    model = model.transform(InferShapes())

    # Save to temp directory
    model_path = temp_build_dir + "/minimal_fc.onnx"
    model.save(model_path)

    return model


# =============================================================================
# Full Model Fixtures (for integration tests)
# =============================================================================


@pytest.fixture
def tfc_w1a1_model(temp_build_dir):
    """
    Real TFC-w1a1 (binary fully-connected network) for integration testing.

    This is the same model used in end2end tests. Includes full pipeline:
    - Brevitas export
    - QONNX cleanup
    - ConvertQONNXtoFINN
    - Basic streamlining

    Suitable for testing build pipeline stages.
    """
    # Get pre-trained TFC binary network
    tfc = get_test_model_trained("TFC", 1, 1)

    # Export to ONNX
    export_path = temp_build_dir + "/tfc_w1a1_export.onnx"
    export_qonnx(tfc, torch.randn(1, 1, 28, 28), export_path)
    qonnx_cleanup(export_path, out_file=export_path)

    # Convert to FINN and perform basic transformations
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Save streamlined version
    streamlined_path = temp_build_dir + "/tfc_w1a1_streamlined.onnx"
    model.save(streamlined_path)

    return model


@pytest.fixture
def cnv_w1a2_model(temp_build_dir):
    """
    Real CNV-w1a2 (quantized convolutional network) for conv integration testing.

    This is the same model used in end2end tests. Includes:
    - Brevitas export
    - QONNX cleanup
    - ConvertQONNXtoFINN
    - Streamlining (with convolution-specific transforms)

    Suitable for testing convolution layer handling in build pipeline.
    """
    # Get pre-trained CNV quantized network
    cnv = get_test_model_trained("CNV", 1, 2)

    # Export to ONNX
    export_path = temp_build_dir + "/cnv_w1a2_export.onnx"
    export_qonnx(cnv, torch.randn(1, 3, 32, 32), export_path)
    qonnx_cleanup(export_path, out_file=export_path)

    # Convert to FINN
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Apply streamlining (includes conv-specific transforms)
    model = model.transform(Streamline())

    # Save streamlined version
    streamlined_path = temp_build_dir + "/cnv_w1a2_streamlined.onnx"
    model.save(streamlined_path)

    return model
