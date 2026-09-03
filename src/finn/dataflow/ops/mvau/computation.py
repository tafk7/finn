# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What an MVAU node *computes*, as distinct from what is built for it.

Three things live here and nothing else: the profile that names which of the
three MVAU computations a node describes, the numerical reference for each of
them, and the one initializer analysis MVAU declares.  They are together
because they are the same subject -- the mathematics of the source operation --
and apart from ``op.py`` because that module is about the design space over it.

**The profile is a fact about the node, not a mode switch.**  It is derived
from two attributes FINN's graphs already carry, and every consumer reads the
derivation rather than re-testing the attributes: a reader that asks
``noActivation`` directly is a reader that will disagree with this one the
first time a third profile appears.

**The execution reference is the oracle's, deliberately.**  ``execute_node``
exists so a FINN model containing these nodes can be run in ONNX, and its
answers have to be the same ones the previous implementation gave, bit for bit
-- otherwise a regression in the *design space* is indistinguishable from a
change in what MVAU means.  So the bipolar special case, the four-dimensional
transpose around ``multithreshold``, and the ``BIPOLAR`` output scale and bias
are ported as behaviour rather than reasoned about again.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from finn.dataflow.datatypes import QONNXDataType


class MvauComputationProfile(str, Enum):
    """Which of the three MVAU computations one node describes.

    The values match the previous implementation's, so a correspondence table
    between the two stacks compares strings rather than a mapping somebody
    maintains.
    """

    ACCUMULATOR_INTEGER = "accumulator_integer"
    BIPOLAR_XNOR_ACCUMULATOR = "bipolar_xnor_accumulator"
    FUSED_THRESHOLD = "fused_threshold"


def computation_profile(*, no_activation: bool, binary_xnor: bool) -> MvauComputationProfile:
    """The profile, from the two attributes that determine it.

    A fused threshold outranks the XNOR mode: a thresholded node is thresholded
    whatever its accumulator does, and the threshold is what decides the output
    type and the operand list.
    """

    if not no_activation:
        return MvauComputationProfile.FUSED_THRESHOLD
    if binary_xnor:
        return MvauComputationProfile.BIPOLAR_XNOR_ACCUMULATOR
    return MvauComputationProfile.ACCUMULATOR_INTEGER


def initializer_excludes_minimum(values: Any, datatype: QONNXDataType) -> bool | None:
    """Whether an initializer avoids the minimum value its datatype can hold.

    The one fact MVAU needs from the weights themselves, and the reason a
    narrow-weights encoding is *derivable* rather than an attribute somebody
    sets: a signed weight matrix that never uses its most negative value can be
    stored one bit narrower, and whether it does is a property of the values.

    ``None`` when the question cannot be asked -- an empty array, a datatype
    with no minimum -- because "no answer" and "uses the minimum" lead to
    different builds and must not be spelled the same way.
    """

    import numpy  # type: ignore[import-not-found]  # noqa: PLC0415 - heavy import

    try:
        array = numpy.asarray(values)
        if array.size == 0:
            return None
        return bool(float(array.min()) != float(datatype.min()))
    except (AttributeError, TypeError, ValueError):
        return None


def execute_mvau(
    *,
    activation: Any,
    weight: Any,
    thresholds: Any | None,
    profile: MvauComputationProfile,
    activation_type: QONNXDataType,
    weight_type: QONNXDataType,
    output_type: QONNXDataType,
    activation_bias: int,
) -> Any:
    """One MVAU node's numerical result, before it is reshaped to the output.

    Kept apart from the CustomOp method so the reference can be tested against
    ``numpy.matmul``, ``xnorpopcountmatmul`` and ``multithreshold`` directly,
    without a graph -- execution is the piece most able to be silently wrong,
    and a test that has to build a model to check it tests the model too.
    """

    import numpy  # noqa: PLC0415 - heavy import
    import qonnx.custom_op.general.xnorpopcount as xnor  # type: ignore[import-not-found] # noqa: PLC0415
    from qonnx.core.datatype import DataType  # type: ignore[import-not-found] # noqa: PLC0415
    from qonnx.custom_op.general.multithreshold import (  # type: ignore[import-not-found] # noqa: PLC0415
        multithreshold,
    )

    bipolar = DataType["BIPOLAR"]
    if profile is MvauComputationProfile.BIPOLAR_XNOR_ACCUMULATOR:
        result = xnor.xnorpopcountmatmul(activation, weight)
    elif activation_type == bipolar and weight_type == bipolar:
        # The same XNOR popcount, but the operands arrive in {-1, +1} rather
        # than already mapped to {0, 1}.
        result = xnor.xnorpopcountmatmul((activation + 1) / 2, (weight + 1) / 2)
    else:
        result = numpy.matmul(activation, weight)

    if profile is not MvauComputationProfile.FUSED_THRESHOLD:
        return result
    if thresholds is None:
        raise ValueError("a fused-threshold MVAU cannot execute without its threshold operand")

    scale = 2 if output_type == bipolar else 1
    bias = -1 if output_type == bipolar else activation_bias
    # multithreshold wants channels second; a four-dimensional activation
    # arrives channels-last, so it is transposed there and back.
    if result.ndim == 4:
        result = result.transpose((0, 3, 1, 2))
        return multithreshold(result, thresholds, scale, bias).transpose((0, 2, 3, 1))
    return multithreshold(result, thresholds, scale, bias)


__all__ = [
    "MvauComputationProfile",
    "computation_profile",
    "execute_mvau",
    "initializer_excludes_minimum",
]
