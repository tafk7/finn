############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""A DataflowKernel-backed MVAU folded by the STOCK ``SetFolding`` transform (R1 seam).

Folding rides the ``get_folding_axes`` capability query — no op_type-string match. The
dials round-trip through nodeattrs and drive ``get_exp_cycles`` down monotonically.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

pytestmark = pytest.mark.integration

MW, MH = 128, 64
OP_TYPE = "MVAU"
DOMAIN = "finn.kernels"


def _build_model():
    node = helper.make_node(
        OP_TYPE, ["inp", "weights"], ["out"], domain=DOMAIN, backend="mvau_hls", SIMD=1, PE=1
    )
    graph = helper.make_graph(
        [node], "mvau_kernel_graph",
        [
            helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT32"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def test_set_folding_folds_kernel_backed_mvau():
    from finn.transformation.fpgadataflow.set_folding import SetFolding

    model = _build_model()
    unfolded_cycles = model.get_customop_wrapper(model.graph.node[0]).get_exp_cycles()  # 8192

    target = 512
    model = model.transform(SetFolding(target_cycles_per_frame=target))

    inst = model.get_customop_wrapper(model.graph.node[0])
    pe, simd = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
    folded_cycles = inst.get_exp_cycles()

    assert MH % pe == 0 and pe >= 1
    assert MW % simd == 0 and simd >= 1
    assert pe > 1 or simd > 1, "SetFolding did not fold the node at all"
    assert folded_cycles < unfolded_cycles
    assert folded_cycles < target


def test_folding_is_monotone_in_dials():
    model = _build_model()
    inst = model.get_customop_wrapper(model.graph.node[0])
    c_1_1 = inst.get_exp_cycles()
    inst.set_nodeattr("SIMD", 16)
    c_16_1 = inst.get_exp_cycles()
    inst.set_nodeattr("PE", 4)
    c_16_4 = inst.get_exp_cycles()
    assert c_1_1 > c_16_1 > c_16_4
