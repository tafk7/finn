# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Scoped AXIS declarations retain concrete occurrence and field/view types."""

from typing_extensions import assert_type

from finn.kernels.datatypes.values import QONNXDataType
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.physical.axi_stream import AxiStream, AxiStreamInterface, AxiStreamScope
from finn.kernels.physical.layout import PackedBeatLayout
from finn.kernels.space._next import ValueRef, View, ViewAssessment


def check(point: DotpAxiKernel) -> None:
    assert_type(DotpAxiKernel.activation, AxiStreamInterface)
    assert_type(DotpAxiKernel.activation.dtype, ValueRef[QONNXDataType])
    assert_type(DotpAxiKernel.activation.lanes, ValueRef[int])
    assert_type(DotpAxiKernel.activation.payload_bits, ValueRef[int])
    assert_type(DotpAxiKernel.activation.accepted_stream, ValueRef[AxiStream])
    assert_type(DotpAxiKernel.activation.view(), View[AxiStream])
    assert_type(point.activation, AxiStreamScope)
    assert_type(point.activation.dtype, QONNXDataType)
    assert_type(point.activation.payload_bits, int)
    assert_type(point.activation.payload, PackedBeatLayout)
    assert_type(point.activation.assess(DotpAxiKernel.activation.view()), ViewAssessment[AxiStream])
