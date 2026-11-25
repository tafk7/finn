# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import (
    CustomOpRefused,
    get_supported_versions,
    getCustomOp,
)
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.onnx_exec import execute_onnx

pytestmark = pytest.mark.util

_DATAFLOW = "finn.custom_op.fpgadataflow"


def make_model(op_type, domain, **attrs):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(op_type, ["inp"], ["out"], name="stored_op", domain=domain, **attrs)
    graph = helper.make_graph([node], "stored_model", [inp], [out])
    return ModelWrapper(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 11), helper.make_opsetid(domain, 1)]
        )
    )


@pytest.mark.parametrize(
    "suffix,op_type,reason",
    [
        ("", "AddStreams_Batch", "renamed AddStreams at v0.10; replaced by ElementwiseAdd"),
        ("", "ChannelwiseOp_Batch", "replaced by ElementwiseAdd and ElementwiseMul"),
        ("", "CheckSum", "finn.custom_op.fpgadataflow.hls.CheckSum_hls at v0.10"),
        ("", "ConvolutionInputGenerator1D", "consolidated into ConvolutionInputGenerator at v0.10"),
        (
            "",
            "ConvolutionInputGenerator_rtl",
            "finn.custom_op.fpgadataflow.rtl.ConvolutionInputGenerator_rtl",
        ),
        ("", "DuplicateStreams_Batch", "renamed DuplicateStreams at v0.10"),
        ("", "FMPadding_Batch", "renamed FMPadding at v0.10"),
        ("", "FMPadding_rtl", "finn.custom_op.fpgadataflow.rtl.FMPadding_rtl at v0.10"),
        ("", "GlobalAccPool_Batch", "renamed GlobalAccPool at v0.10"),
        ("", "IODMA", "finn.custom_op.fpgadataflow.hls.IODMA_hls at v0.10"),
        ("", "LabelSelect_Batch", "renamed LabelSelect at v0.10"),
        (
            "",
            "MatrixVectorActivation",
            "renamed MVAU at v0.10 (MVAU_hls, MVAU_rtl after SpecializeLayers)",
        ),
        ("", "Pool_Batch", "renamed Pool at v0.10"),
        ("", "StreamingDataWidthConverter_Batch", "renamed StreamingDataWidthConverter at v0.10"),
        ("", "StreamingMaxPool_Batch", "renamed StreamingMaxPool at v0.10; replaced by Pool"),
        ("", "Thresholding_Batch", "renamed Thresholding at v0.10"),
        ("", "TLastMarker", "replaced by finn.custom_op.fpgadataflow.rtl.TLastMarker_rtl"),
        ("", "UpsampleNearestNeighbour_Batch", "renamed UpsampleNearestNeighbour at v0.10"),
        (
            "",
            "VectorVectorActivation",
            "renamed VVAU at v0.10 (VVAU_hls, VVAU_rtl after SpecializeLayers)",
        ),
        ("", "AddStreams", "replaced by ElementwiseAdd"),
        ("", "ChannelwiseOp", "replaced by ElementwiseAdd and ElementwiseMul"),
        ("", "DownSampler", "ConvolutionInputGenerator with a 1x1 kernel and stride"),
        (
            "",
            "StreamingEltwise",
            "replaced by ElementwiseAdd, ElementwiseSub and ElementwiseAbsDiff",
        ),
        ("", "StreamingMaxPool", "replaced by Pool"),
        (".hls", "AddStreams_hls", "replaced by ElementwiseAdd_hls"),
        (".hls", "ChannelwiseOp_hls", "replaced by ElementwiseAdd_hls and ElementwiseMul_hls"),
        (
            ".hls",
            "ConvolutionInputGenerator_hls",
            "finn.custom_op.fpgadataflow.rtl.ConvolutionInputGenerator_rtl",
        ),
        (".hls", "DownSampler_hls", "ConvolutionInputGenerator_rtl with a 1x1 kernel and stride"),
        (".hls", "FMPadding_hls", "replaced by finn.custom_op.fpgadataflow.rtl.FMPadding_rtl"),
        (
            ".hls",
            "StreamingEltwise_hls",
            "ElementwiseAdd_hls, ElementwiseSub_hls and ElementwiseAbsDiff_hls",
        ),
        (".hls", "StreamingMaxPool_hls", "replaced by Pool_hls"),
        (".hls", "TLastMarker_hls", "replaced by finn.custom_op.fpgadataflow.rtl.TLastMarker_rtl"),
    ],
)
def test_saved_retired_op_refused_at_lookup(tmp_path, suffix, op_type, reason):
    domain = _DATAFLOW + suffix
    path = tmp_path / "stored.onnx"
    make_model(op_type, domain).save(path)
    model = ModelWrapper(str(path))
    node = model.graph.node[0]
    assert (node.domain, node.op_type) == (domain, op_type)

    with pytest.raises(CustomOpRefused) as raised:
        getCustomOp(node, onnx_opset_version=1)
    assert isinstance(raised.value, KeyError)
    finding = raised.value.findings[0]
    assert finding.code == "domain-retired"
    assert finding.owner == "getCustomOp"
    assert reason in finding.message
    assert "is retired;" in finding.message
    assert dict(finding.details) == {"domain": domain, "op_type": op_type, "node": "stored_op"}

    with pytest.raises(CustomOpRefused) as transformed:
        model.transform(InferShapes())
    assert transformed.value.findings[0].code == "domain-retired"
    assert reason in transformed.value.findings[0].message

    with pytest.raises(CustomOpRefused) as versions:
        get_supported_versions(domain, op_type)
    assert versions.value.findings[0].code == "domain-retired"
    assert reason in versions.value.findings[0].message


@pytest.mark.parametrize("suffix", ["", ".hls"])
def test_unknown_op_is_still_unknown(suffix):
    model = make_model("NeverAnOp", _DATAFLOW + suffix)
    with pytest.raises(CustomOpRefused) as raised:
        getCustomOp(model.graph.node[0])
    assert isinstance(raised.value, KeyError)
    assert raised.value.findings[0].code == "op-unknown"


@pytest.mark.parametrize("op_type", ["ConvolutionInputGenerator_rtl", "FMPadding_rtl"])
def test_current_rtl_identity_still_found(op_type):
    node = make_model(op_type, _DATAFLOW + ".rtl").graph.node[0]
    assert getCustomOp(node).onnx_node is node
    assert get_supported_versions(node.domain, op_type) == [1]


def test_current_hls_mvau_still_found():
    node = make_model("MVAU_hls", _DATAFLOW + ".hls").graph.node[0]
    assert getCustomOp(node).onnx_node is node
    assert get_supported_versions(node.domain, node.op_type) == [1]


def test_current_dataflow_model_executes_unchanged():
    model = make_model(
        "DuplicateStreams", _DATAFLOW, NumChannels=4, PE=1, NumOutputStreams=2, inputDataType="INT8"
    )
    model.graph.node[0].output.append("out_copy")
    model.graph.output.append(helper.make_tensor_value_info("out_copy", TensorProto.FLOAT, [1, 4]))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])
    model.set_tensor_datatype("out_copy", DataType["INT8"])
    model = model.transform(InferShapes())
    inp = np.array([[-3, 0, 2, 7]], dtype=np.float32)
    outputs = execute_onnx(model, {"inp": inp})
    for name in ("out", "out_copy"):
        np.testing.assert_array_equal(outputs[name], inp)
