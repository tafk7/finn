# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source-operation execution for MVAU nodes.

The reusable mathematical profile and its derivation live in
``finn.dataflow.kernels.matmul.base`` because both the implementation family
and source adapter consume them. This module retains the source execution
reference that interprets that profile against ONNX/QONNX tensor values.

**The execution reference is the oracle's, deliberately.**  ``execute_node``
exists so a FINN model containing these nodes can be run in ONNX, and its
answers have to be the same ones the previous implementation gave, bit for bit
-- otherwise a regression in the *design space* is indistinguishable from a
change in what MVAU means.  So the bipolar special case, the four-dimensional
transpose around ``multithreshold``, and the ``BIPOLAR`` output scale and bias
are ported as behaviour rather than reasoned about again.
"""

from __future__ import annotations

from typing import Any

from finn.dataflow.kernels.matmul.base import AccumulationMode, MvauComputationProfile
from finn.kernels.datatypes.values import QONNXDataType


def execute_mvau(
    *,
    activation: Any,
    weight: Any,
    thresholds: Any | None,
    profile: MvauComputationProfile,
    output_type: QONNXDataType | None,
    activation_bias: int,
) -> Any:
    """One MVAU node's numerical result, before it is reshaped to the output.

    Kept apart from the CustomOp method so the reference can be tested against
    ``numpy.matmul``, ``xnorpopcountmatmul`` and ``multithreshold`` directly,
    without a graph -- execution is the piece most able to be silently wrong,
    and a test that has to build a model to check it tests the model too.
    """

    import numpy  # type: ignore[import-not-found]  # noqa: PLC0415 - heavy import
    import qonnx.custom_op.general.xnorpopcount as xnor  # type: ignore[import-not-found] # noqa: PLC0415
    from qonnx.core.datatype import DataType  # type: ignore[import-not-found]  # noqa: PLC0415
    from qonnx.custom_op.general.multithreshold import (  # type: ignore[import-not-found] # noqa: PLC0415
        multithreshold,
    )

    # Two axes, applied in order: accumulate, then activate.  Written as two
    # independent statements because they *are* independent -- an XNOR popcount
    # followed by a multithreshold is a real node, and the shape that made it
    # unrepresentable was one exclusive enum.
    if profile.accumulation is AccumulationMode.XNOR_POPCOUNT:
        result = xnor.xnorpopcountmatmul(activation, weight)
    elif profile.accumulation is AccumulationMode.BIPOLAR_POPCOUNT:
        # The same popcount, but the operands arrive in {-1, +1} rather than
        # already mapped to {0, 1}.
        result = xnor.xnorpopcountmatmul((activation + 1) / 2, (weight + 1) / 2)
    else:
        result = numpy.matmul(activation, weight)

    if not profile.fuses_activation:
        return result
    if thresholds is None:
        raise ValueError("a fused-threshold MVAU cannot execute without its threshold operand")
    if output_type is None:
        raise ValueError(
            "a fused-threshold MVAU scales and biases by its output datatype, and this node "
            "does not supply one"
        )

    bipolar = DataType["BIPOLAR"]
    scale = 2 if output_type == bipolar else 1
    bias = -1 if output_type == bipolar else activation_bias
    # multithreshold wants channels second; a four-dimensional activation
    # arrives channels-last, so it is transposed there and back.
    if result.ndim == 4:
        result = result.transpose((0, 3, 1, 2))
        return multithreshold(result, thresholds, scale, bias).transpose((0, 2, 3, 1))
    return multithreshold(result, thresholds, scale, bias)


__all__ = [
    "execute_mvau",
]
