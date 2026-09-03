# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What an MVAU node *computes*, as distinct from what is built for it.

Three things live here and nothing else: the profile that says what a node
computes, the numerical reference for each combination, and the one initializer
analysis MVAU declares.  They are together because they are the same subject --
the mathematics of the source operation -- and apart from ``op.py`` because
that module is about the design space over it.

**The profile is two axes, not one enum.**  How the products are accumulated
and what happens to the accumulator afterwards are independent: an XNOR
popcount followed by a multithreshold is an ordinary node, and collapsing the
two into one exclusive value makes it unrepresentable -- silently, as a plain
matrix product that is then thresholded.  So the profile carries both, and
every consumer reads the axis it is about.

**It is a fact about the node, not a mode switch.**  Every consumer reads the
derivation rather than re-testing the attributes, because the accumulation also
depends on the operand *datatypes* -- two BIPOLAR operands mean a popcount
whatever ``binaryXnorMode`` says -- and a reader that checked only the
attribute would disagree with this one.

**The execution reference is the oracle's, deliberately.**  ``execute_node``
exists so a FINN model containing these nodes can be run in ONNX, and its
answers have to be the same ones the previous implementation gave, bit for bit
-- otherwise a regression in the *design space* is indistinguishable from a
change in what MVAU means.  So the bipolar special case, the four-dimensional
transpose around ``multithreshold``, and the ``BIPOLAR`` output scale and bias
are ported as behaviour rather than reasoned about again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from finn.dataflow.datatypes import QONNXDataType


class AccumulationMode(str, Enum):
    """How the products are accumulated."""

    #: An ordinary integer matrix product.
    INTEGER = "integer"
    #: ``binaryXnorMode``: an XNOR popcount over operands already in {0, 1}.
    XNOR_POPCOUNT = "xnor_popcount"
    #: The same popcount, reached because *both operands are BIPOLAR* rather
    #: than because an attribute said so.  The operands arrive in {-1, +1} and
    #: are mapped before the popcount, which is why it is a third value and not
    #: a flag on the second.
    BIPOLAR_POPCOUNT = "bipolar_popcount"


class ActivationMode(str, Enum):
    """What happens to the accumulator afterwards."""

    NONE = "none"
    MULTITHRESHOLD = "multithreshold"


@dataclass(frozen=True, slots=True)
class MvauComputationProfile:
    """What one MVAU node computes, on the two axes that are independent.

    These *are* two axes and were briefly modelled as one three-valued enum,
    which silently lost a real combination: ``binaryXnorMode`` with a fused
    threshold is an XNOR popcount **and then** a multithreshold, and an
    exclusive enum forced the threshold to win and the popcount to be dropped.
    The node computed a plain matrix product and thresholded that, which is a
    different function of the same graph.

    So the accumulation and the post-operation are carried separately, and
    every consumer reads whichever axis it is about.
    """

    accumulation: AccumulationMode
    activation: ActivationMode

    @property
    def fuses_activation(self) -> bool:
        return self.activation is ActivationMode.MULTITHRESHOLD

    @property
    def name(self) -> str:
        """A stable label, for a message or a parity record."""

        return f"{self.accumulation.value}+{self.activation.value}"


def computation_profile(
    *,
    no_activation: bool,
    binary_xnor: bool,
    activation_type: QONNXDataType | None = None,
    weight_type: QONNXDataType | None = None,
) -> MvauComputationProfile:
    """The profile, from the attributes and the operand datatypes that decide it.

    The accumulation is decided here rather than at execution time so that one
    derived fact answers "what does this node compute" for every consumer --
    the Designs' applicability, the execution, and any later parity record.
    The implicit bipolar case is part of it precisely because it is *not* an
    attribute: two BIPOLAR operands mean a popcount whatever the attributes
    say, and a consumer testing only ``binaryXnorMode`` would miss it.
    """

    from qonnx.core.datatype import DataType  # type: ignore[import-not-found]  # noqa: PLC0415

    bipolar = DataType["BIPOLAR"]
    if binary_xnor:
        accumulation = AccumulationMode.XNOR_POPCOUNT
    elif activation_type == bipolar and weight_type == bipolar:
        accumulation = AccumulationMode.BIPOLAR_POPCOUNT
    else:
        accumulation = AccumulationMode.INTEGER
    return MvauComputationProfile(
        accumulation,
        ActivationMode.NONE if no_activation else ActivationMode.MULTITHRESHOLD,
    )


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
    output_type: QONNXDataType | None,
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
    from qonnx.core.datatype import DataType  # noqa: PLC0415
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
    "AccumulationMode",
    "ActivationMode",
    "MvauComputationProfile",
    "computation_profile",
    "execute_mvau",
    "initializer_excludes_minimum",
]
