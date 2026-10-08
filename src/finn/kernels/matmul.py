# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A matrix-multiply unit on channels: activations and weights in, results out.

The facts are the extents in canonical GEMM notation, ``m`` rows, ``n``
outputs and the reduction ``k``, and the ``form`` (``finn.dataflow.gemm``):

- ``DENSE``: Y[m, n] = sum over k of X[m, k] * W[k, n]. Every output reads the
  whole activation row, so the row is replayed once per output fold.
- ``DEPTHWISE``: Y[m, n] = sum over k of X[m, k, n] * W[k, n]. Each output
  channel reads its own activations, so nothing is replayed.

Weights are read ``(k, n)``, as ONNX ``MatMul`` stores them. A depthwise
operation runs natively (one channel per PE lane, INT8 DSP58 only) or, by the
``realization`` Decision, on the dense datapath with block-diagonal weights
(``block_diagonal``), reading its (M, K, N) activations as (M, K * N).

``MatMulKernel`` is a kernel with children: the compute cores that sit on the
channels its parent supplies, ``x_channel`` (the activations), ``w_channel`` (the
weights) and ``y_channel`` (the results). Each channel, its adapter, its FIFO
and its source are its parent's: the parent (a test harness, a KernelOp's node
root, or a shell root) declares each channel and either binds its tensor to
MatMul's view of it (``activation_tensor``, ``weight_tensor``,
``result_tensor``), which reads only MatMul's facts and ``realization``, never
a port, or states it; ``carried`` refuses a stated tensor of another shape, or
whose values do not fit (``matmul-tensor``): MatMul's results must fit their
channel, and an input channel's values must fit what MatMul reads. A boundary
channel there presents its ``port`` name (``in0_V``).

MatMul consumes its weights; it never holds them. Known weights are the weight
channel's value: its ``contents``, which its ``source`` stores, bound by
whoever declares the channel (the value's owner), who states on the channel's
tensor the range it promises (``INT4 over [-7, 7]``). With several weight sets
the channel's ``sets`` and ``index`` select one per row. A consumer derives
from what the channel carries what it may (every core's weight width, the
packed core's ``NARROW_WEIGHTS``); the owner states known weights narrowed to
their values (``stored_element``: INT8-typed ternary weights are ``INT2``); a
channel without a value carries the datatype's range. A dense realization of a
depthwise operation reads block-diagonal weights, so it needs a value on its
weight channel (``matmul-realization``).

The result's range is derived from what the weight channel carries: with a
value, the exact range of each output column's dot products over the
activation datatype (``column_range``), unioned over the columns; without one,
every dot product of the two datatypes' values. The result's type is the
range's smallest encoding, ``UINT`` when no value is negative (``range_dtype``,
``result_type``), whatever core computes it; the result element states the
range (``result_tensor``). Each core derives its accumulator from the range
(the core's ``result_range``).

- ``compute`` is a Decision over the dot-product cores. They share the facts
  and channels. Each core owns its folding factors (``compute.<core>.pe``,
  ``.simd``, ``.compute_pumping``) and derives every channel's beat sequence
  from its schedule.
- Where the weights come from is the weight channel's ``source``, not MatMul's.
- The activation channel's plan replays each dense row and frames each
  reduction; the channel's adapter carries that out.

Its ``module`` (``finn.kernels.base``) merges its children's netlists; placed
alone, without its channels, its interfaces are idle.
"""

from __future__ import annotations

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    requires,
    view,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import require_positive
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.target import Platform
from finn.kernels.values.domains import admit_element, range_dtype, values_within
from finn.kernels.values.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerTensorValue,
    integer_columns,
    integer_shape,
    integers,
)

_CARRIED = (
    ("x_channel", "activation_tensor"),
    ("w_channel", "weight_tensor"),
    ("y_channel", "result_tensor"),
)
"""Each channel MatMul sits on, and its view of the tensor the channel carries."""


def datatype_range(
    vector_length: int, activation_dtype: QONNXDataType, weights_dtype: QONNXDataType
) -> tuple[int, int]:
    """The least and greatest dot product of ``vector_length`` values of each datatype."""
    require_positive(vector_length, "vector_length")
    activation = ordinary_integer_bounds(canonical_qonnx_datatype(activation_dtype))
    weights = ordinary_integer_bounds(canonical_qonnx_datatype(weights_dtype))
    products = tuple(a * w for a in activation for w in weights)
    return vector_length * min(products), vector_length * max(products)


def exact_result_dtype(
    vector_length: int, activation_dtype: QONNXDataType, weights_dtype: QONNXDataType
) -> QONNXDataType:
    """The smallest encoding of every full-range integer dot product (``range_dtype``)."""
    return range_dtype(*datatype_range(vector_length, activation_dtype, weights_dtype))


def column_range(activation_dtype: QONNXDataType, weights: IntegerTensor) -> tuple[int, int]:
    """The least and greatest dot product of any column of ``weights`` (``(..., k, n)``,
    reduced over ``k``) with activations of ``activation_dtype``.

    A column whose positive weights sum to P and negative ones to N meets activations
    in [lo, hi] (an ordinary integer type: lo <= 0 <= hi) over [lo * P + hi * N,
    hi * P + lo * N], each term taking its extremes independently. Every partial sum
    lies within it too, since each term's range holds 0.
    """
    low, high = ordinary_integer_bounds(canonical_qonnx_datatype(activation_dtype))
    columns = integer_columns(weights)
    return (
        min(low * positive + high * negative for positive, negative in columns),
        max(high * positive + low * negative for positive, negative in columns),
    )


def block_diagonal(weights: IntegerTensor) -> IntegerTensorValue:
    """Depthwise weights ``(k, n)``, or one operand a set ``(sets, k, n)``, as the dense
    datapath reads them: ``(k * n, n)`` a set.

    W'[k * N + c, n] = W[k, n] when c = n, and 0 otherwise: the densely read
    activation row (k, c) meets only its own channel's weights. The value owner
    states it for a dense realization; it reads the integers.
    """
    shape = integer_shape(weights)
    if shape is None or len(shape) not in (2, 3):
        raise ValueError("depthwise weights are (k, n), or (sets, k, n)")
    channels, flat = shape[-1], integers(weights)
    dense = [
        value if channel == output else 0
        for row in range(0, len(flat), channels)
        for channel in range(channels)
        for output, value in enumerate(flat[row : row + channels])
    ]
    return IntegerTensorValue.flat((*shape[:-2], shape[-2] * channels, channels), dense)


class MatMulKernel(Kernel):
    """Operation facts, and the kernels and Decisions over kernels on its channels."""

    id = "finn.matmul"
    version = 1

    m: int = Param()
    n: int = Param()
    k: int = Param()
    form: Form = Param(default=Form.DENSE)
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    platform: Platform = Param()

    @derived
    def depthwise(self) -> bool:
        return self.form is Form.DEPTHWISE

    @derived
    def reads_depthwise(self) -> bool:
        """Whether a core reads depthwise operands natively: only the INT8 DSP58 core
        does, within its own bounds on the platform's DSP and the datatypes."""
        return (
            Int8Dsp58DotpKernel.operand_refusal(
                self.platform.dsp, self.activation_dtype, self.weights_dtype
            )
            is None
        )

    # A depthwise operation runs natively (one channel per PE lane, a core that reads
    # it) or on the dense datapath with block-diagonal weights, on any core.
    realization: str = Decision(
        values=("native", "dense"),
        when=depthwise,
        requires=(
            requires(
                reads_depthwise,
                "matmul-native: no core reads these depthwise operands natively "
                "(the INT8 DSP58 core alone does, within its bounds)",
                cases=("native",),
            ),
        ),
    )

    @derived
    def datapath(self) -> Form:
        """The form the datapath computes: densely realized, a depthwise one is dense."""
        if self.form is Form.DENSE or self.realization == "dense":
            return Form.DENSE
        return Form.DEPTHWISE

    @derived
    def dense_view(self) -> bool:
        """Whether the datapath reads depthwise activations (M, K, N) as (M, K * N)."""
        return self.depthwise and self.datapath is Form.DENSE

    @derived
    def datapath_k(self) -> int:
        """The datapath's K: the window times the channels when densely realized."""
        return self.k * self.n if self.dense_view else self.k

    @derived
    def result_range(self) -> tuple[int, int] | Rejected:
        """The results' least and greatest value: with a value on the weight channel, its
        columns' over the activation datatype (``column_range``, per output column); without
        one, every dot product of the two datatypes' values (``datatype_range``)."""
        try:
            if self.stored:
                return column_range(self.activation_dtype, self.w_channel.contents)
            return datatype_range(self.k, self.activation_dtype, self.weights_dtype)
        except ValueError as error:
            return reject("matmul-arithmetic", str(error))

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_type(self) -> QONNXDataType:
        """The result range's smallest encoding: ``INT<b>``, or ``UINT<b>`` when no result
        is negative (``range_dtype``). No core's accumulator enters it."""
        least, greatest = self.result_range
        return range_dtype(least, greatest)

    @constraint
    def extents_supported(self) -> bool | Rejected:
        if any(type(value) is not int or value < 1 for value in (self.m, self.n, self.k)):
            return reject("matmul-extents", "the extents m, n and k must be positive integers")
        return True

    # The tensors the channels carry.

    def _tensor(
        self,
        shape: tuple[int, ...],
        dtype: QONNXDataType,
        value_range: tuple[int, int] | None = None,
    ) -> Tensor | Rejected:
        element = admit_element(dtype, value_range)
        if isinstance(element, Rejected):
            return element
        try:
            return Tensor(shape, element)
        except ValueError as error:
            return reject("matmul-extents", str(error))

    @view
    def activation_tensor(self) -> Tensor | Rejected:
        """(M, K), or (M, K, N) depthwise, however the datapath reads it."""
        shape = (self.m, self.k, self.n) if self.depthwise else (self.m, self.k)
        return self._tensor(shape, self.activation_dtype)

    @view
    def weight_tensor(self) -> Tensor | Rejected:
        """(K, N) as the datapath reads it, in the weights' datatype: the weight channel's
        values must be values of it, in whatever encoding its owner states."""
        return self._tensor((self.datapath_k, self.n), self.weights_dtype)

    @view
    def result_tensor(self) -> Tensor | Rejected:
        """(M, N), the result type over the result range."""
        return self._tensor((self.m, self.n), self.result_type, self.result_range)

    # The channels it sits on, supplied by its parent.
    x_channel: Channel = Param(required=False)
    w_channel: Channel = Param(required=False)
    y_channel: Channel = Param(required=False)

    @derived
    def stored(self) -> bool:
        """Whether the weight channel carries a known value, which its source stores."""
        return self.present(MatMulKernel.w_channel) and self.w_channel.valued

    @constraint
    def carried(self) -> bool | Rejected:
        """Each supplied channel carries a tensor of the shape MatMul derives for it.

        On the results channel MatMul's values fit the channel's element; on a
        channel it consumes, activations or weights, the channel's values are
        values of MatMul's, in any integer encoding (``values_within``): the owner of
        known weights states them narrowed to their values.
        """
        for reference, tensor in _CARRIED:
            if not self.present(getattr(MatMulKernel, reference)):
                continue
            supplied, derived_ = getattr(self, reference).tensor, getattr(self, tensor)
            produced = reference == "y_channel"
            if produced:
                fits = derived_.element.fits(supplied.element)
            else:
                fits = values_within(supplied.element, derived_.element)
            if supplied.shape != derived_.shape or not fits:
                return reject(
                    "matmul-tensor",
                    f"{reference} carries {supplied.shape} {supplied.element}; "
                    f"MatMul's {tensor} is {derived_.shape} {derived_.element}, "
                    f"which {'must fit it' if produced else 'it must fit'}",
                )
        return True

    # The compute cores. Each refuses what its core cannot build and owns its
    # folding factors; ``packed`` names the entry, so references reach through it.
    packed = PackedDotpKernel()
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Decision(
        {"packed": packed, "int8_dsp58": Int8Dsp58DotpKernel},
        form=datapath,
        reshape_activations=dense_view,
        result_range=result_range,
        platform=platform,
        x_channel=x_channel,
        w_channel=w_channel,
        y_channel=y_channel,
    )

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.dense_view and not self.stored:
            return reject(
                "matmul-realization",
                "a dense realization reads block-diagonal weights, so it needs a value on "
                "its weight channel",
            )
        return True

    admission = ConstraintGroup(extents_supported, realization_supported, carried)

    def stem(self) -> str:
        return "finn_matmul"


__all__ = [
    "MatMulKernel",
    "block_diagonal",
    "column_range",
    "datatype_range",
    "exact_result_dtype",
]
