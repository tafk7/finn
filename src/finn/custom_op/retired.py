# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reasons for retiring stored FINN op identities, declared at custom-op lookup."""

from qonnx.custom_op.registry import Registration

_RETIRED = {
    "finn.custom_op.fpgadataflow": {
        "AddStreams_Batch": "renamed AddStreams at v0.10; replaced by ElementwiseAdd",
        "ChannelwiseOp_Batch": (
            "renamed ChannelwiseOp at v0.10; replaced by ElementwiseAdd and ElementwiseMul"
        ),
        "CheckSum": "moved to finn.custom_op.fpgadataflow.hls.CheckSum_hls at v0.10",
        "ConvolutionInputGenerator1D": (
            "consolidated into ConvolutionInputGenerator at v0.10; "
            "use ConvolutionInputGenerator_rtl after SpecializeLayers"
        ),
        "ConvolutionInputGenerator_rtl": (
            "moved to finn.custom_op.fpgadataflow.rtl.ConvolutionInputGenerator_rtl at v0.10"
        ),
        "DuplicateStreams_Batch": "renamed DuplicateStreams at v0.10",
        "FMPadding_Batch": "renamed FMPadding at v0.10",
        "FMPadding_rtl": "moved to finn.custom_op.fpgadataflow.rtl.FMPadding_rtl at v0.10",
        "GlobalAccPool_Batch": "renamed GlobalAccPool at v0.10",
        "IODMA": "moved to finn.custom_op.fpgadataflow.hls.IODMA_hls at v0.10",
        "LabelSelect_Batch": "renamed LabelSelect at v0.10",
        "MatrixVectorActivation": (
            "renamed MVAU at v0.10 (MVAU_hls, MVAU_rtl after SpecializeLayers)"
        ),
        "Pool_Batch": "renamed Pool at v0.10",
        "StreamingDataWidthConverter_Batch": "renamed StreamingDataWidthConverter at v0.10",
        "StreamingMaxPool_Batch": "renamed StreamingMaxPool at v0.10; replaced by Pool",
        "Thresholding_Batch": "renamed Thresholding at v0.10",
        "TLastMarker": (
            "moved to TLastMarker_hls at v0.10; "
            "replaced by finn.custom_op.fpgadataflow.rtl.TLastMarker_rtl"
        ),
        "UpsampleNearestNeighbour_Batch": "renamed UpsampleNearestNeighbour at v0.10",
        "VectorVectorActivation": (
            "renamed VVAU at v0.10 (VVAU_hls, VVAU_rtl after SpecializeLayers)"
        ),
        "AddStreams": "replaced by ElementwiseAdd",
        "ChannelwiseOp": "replaced by ElementwiseAdd and ElementwiseMul",
        "DownSampler": (
            "replaced by ConvolutionInputGenerator with a 1x1 kernel and stride; "
            "use ConvolutionInputGenerator_rtl after SpecializeLayers"
        ),
        "StreamingEltwise": "replaced by ElementwiseAdd, ElementwiseSub and ElementwiseAbsDiff",
        "StreamingMaxPool": "replaced by Pool",
    },
    "finn.custom_op.fpgadataflow.hls": {
        "AddStreams_hls": "replaced by ElementwiseAdd_hls",
        "ChannelwiseOp_hls": "replaced by ElementwiseAdd_hls and ElementwiseMul_hls",
        "ConvolutionInputGenerator_hls": (
            "replaced by finn.custom_op.fpgadataflow.rtl.ConvolutionInputGenerator_rtl"
        ),
        "DownSampler_hls": (
            "replaced by finn.custom_op.fpgadataflow.rtl.ConvolutionInputGenerator_rtl "
            "with a 1x1 kernel and stride"
        ),
        "FMPadding_hls": "replaced by finn.custom_op.fpgadataflow.rtl.FMPadding_rtl",
        "StreamingEltwise_hls": (
            "replaced by ElementwiseAdd_hls, ElementwiseSub_hls and ElementwiseAbsDiff_hls"
        ),
        "StreamingMaxPool_hls": "replaced by Pool_hls",
        "TLastMarker_hls": "replaced by finn.custom_op.fpgadataflow.rtl.TLastMarker_rtl",
    },
}


def retire(registration: Registration) -> None:
    """Declare retired op types of this domain without registering its current ops."""
    for op_type, reason in _RETIRED[registration.domain].items():
        registration.retire(reason, op_type=op_type)
