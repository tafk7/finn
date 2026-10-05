# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A matrix-multiply unit on streams, with external or stored weights.

The facts are the extents in canonical GEMM notation, ``m`` rows, ``n``
outputs and the reduction ``k``, and the ``form`` (``finn.dataflow.gemm``):

- ``DENSE``: Y[m, n] = sum over k of X[m, k] * W[k, n]. Every output reads the
  whole activation row, so the row is replayed once per output fold.
- ``DEPTHWISE``: Y[m, n] = sum over k of X[m, k, n] * W[k, n]. Each output
  channel reads its own activations, so nothing is replayed.

Weights are stored ``(k, n)``, as ONNX ``MatMul`` stores them. A depthwise
operation runs natively (one channel per PE lane, INT8 DSP58 only) or, by the
``realization`` Decision, on the dense datapath with block-diagonal weights,
reading its (M, K, N) activations as (M, K * N).

``MatMulKernel`` is a kernel with children: the compute cores that sit on the
streams its parent supplies, ``x_channel`` (the activations), ``w_channel`` (the
weights) and ``y_channel`` (the results). Each stream, its adapter, its FIFO
and its source are its parent's: the parent (a test harness, a KernelOp's node
root, a partition root) declares each stream and either binds its tensor to
MatMul's view of it (``activation_tensor``, ``weight_tensor``,
``result_tensor``, ``set_tensor``), which reads only MatMul's facts and
``realization``, never a port, or states it; ``carried`` refuses a stated
tensor of another shape, or whose values do not fit (``matmul-tensor``):
MatMul's values must fit a stream that carries them, and a stream's values
must fit what MatMul consumes. A boundary stream there presents its ``port``
name (``in0_V``).

Known ``weights`` are values MatMul owns: its ``weight_tensor`` states their
range (``INT4 over [-7, 7]``), and its ``weight_values`` view is the value the
weight stream carries (the datapath's weights, block-diagonal when densely
realized), which the parent binds to the stream's ``contents``. Whether the
weights are known is their presence, the view's guard, so the stream's
``source`` applies before the realization is chosen. With several weight
sets, the parent's set channel (bound to ``set_tensor``) is the weight stream's
``index``. A consumer derives from the range what it may (the packed core's
``NARROW_WEIGHTS``). Unknown weights carry the datatype's range.

- ``compute`` is a Decision over the dot-product cores. They share the facts
  and streams. Each core owns its folding factors (``compute.<core>.pe``,
  ``.simd``, ``.compute_pumping``) and derives every stream's beat sequence
  from its schedule.
- Where the weights come from is the weight stream's ``source``, not MatMul's.
- The activation stream's plan replays each dense row and frames each
  reduction; the stream's adapter carries that out.

Its ``module`` (``finn.kernels.base``) merges its children's netlists; placed
alone, without its streams, its interfaces are idle.
"""

from __future__ import annotations

from typing import cast

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.datatypes.domains import set_index_dtype
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    integer_range,
)
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.target import Platform

_CARRIED = (
    ("x_channel", "activation_tensor"),
    ("w_channel", "weight_tensor"),
    ("y_channel", "result_tensor"),
)
"""Each stream MatMul sits on, and its view of the tensor the stream carries."""


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def exact_result_dtype(
    vector_length: int, activation_dtype: QONNXDataType, weights_dtype: QONNXDataType
) -> QONNXDataType:
    """Smallest signed INT covering every full-range integer dot product."""
    _positive(vector_length, "vector_length")
    activation = ordinary_integer_bounds(canonical_qonnx_datatype(activation_dtype))
    weights = ordinary_integer_bounds(canonical_qonnx_datatype(weights_dtype))
    products = tuple(a * w for a in activation for w in weights)
    lower, upper = vector_length * min(products), vector_length * max(products)
    bits = max(1, upper.bit_length() + 1, (~lower).bit_length() + 1 if lower < 0 else 1)
    return resolve_qonnx_datatype_name(f"INT{bits}")


class MatMulKernel(Kernel):
    """Operation facts, and the kernels and Decisions over kernels on its streams."""

    id = "finn.matmul"
    version = 1

    m: int = Param()
    n: int = Param()
    k: int = Param()
    form: Form = Param(default=Form.DENSE)
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    # Several weight sets, one selected per row by an index on ``in2_V``;
    # ``weights`` then holds one operand per set.
    weight_sets: int = Param(default=1)
    platform: Platform = Param()

    @derived
    def known(self) -> bool:
        """Whether the weights are known: MatMul owns them, and its weight stream's
        source stores them."""
        return self.present(MatMulKernel.weights)

    @derived
    def depthwise(self) -> bool:
        return self.form is Form.DEPTHWISE

    @derived
    def multi_set(self) -> bool:
        return self.weight_sets > 1

    # A depthwise operation runs natively (one channel per PE lane, INT8 DSP58
    # only) or on the dense datapath with block-diagonal weights, on any core.
    realization: str = Decision(values=("native", "dense"), when=depthwise)

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

    @derived(semantics=INTEGER_TENSOR)
    def datapath_weights(self) -> IntegerTensor:
        """The weights the datapath reads, ``(k, n)``: block-diagonal when densely realized.

        W'[k * N + c, n] = W[k, n] when c = n, and 0 otherwise: the densely read
        activation row (k, c) meets only its own channel's weights.
        """
        weights = self.weights
        if not self.dense_view:
            return weights
        channels = self.n

        def blocks(operand: object) -> IntegerTensor:
            return tuple(
                tuple(value if channel == output else 0 for output, value in enumerate(row))
                for row in cast("tuple[tuple[int, ...], ...]", operand)
                for channel in range(channels)
            )

        return tuple(blocks(item) for item in weights) if self.multi_set else blocks(weights)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_type(self) -> QONNXDataType | Rejected:
        try:
            return exact_result_dtype(self.k, self.activation_dtype, self.weights_dtype)
        except ValueError as error:
            return reject("matmul-arithmetic", str(error))

    @constraint
    def extents_supported(self) -> bool | Rejected:
        if any(type(value) is not int or value < 1 for value in (self.m, self.n, self.k)):
            return reject("matmul-extents", "the extents m, n and k must be positive integers")
        return True

    # The tensors the streams carry.

    def _tensor(self, shape: tuple[int, ...], dtype: QONNXDataType) -> Tensor | Rejected:
        element = ScalarEncoding.admit(dtype)
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
        """(K, N) as the datapath reads it, over the range of known weights."""
        shape = (self.datapath_k, self.n)
        if not self.present(MatMulKernel.weights):
            return self._tensor(shape, self.weights_dtype)
        least, greatest = integer_range(self.datapath_weights)
        low, high = ordinary_integer_bounds(self.weights_dtype)
        if not low <= least <= greatest <= high:
            return reject(
                "memstream-values",
                f"every value must be an integer admitted by {self.weights_dtype.name}",
            )
        element = ScalarEncoding.admit(self.weights_dtype, (least, greatest))
        if isinstance(element, Rejected):
            return element
        return Tensor(shape, element)

    @view
    def result_tensor(self) -> Tensor | Rejected:
        return self._tensor((self.m, self.n), self.result_type)

    @view(when=known, semantics=INTEGER_TENSOR)
    def weight_values(self) -> IntegerTensor:
        """The value the weight stream carries: the datapath's weights, one operand a set."""
        return self.datapath_weights

    @view
    def set_tensor(self) -> Tensor | Rejected:
        """One set index per row, as wide as the weight source's selector."""
        return self._tensor((self.m,), set_index_dtype(self.weight_sets))

    # The streams it sits on, supplied by its parent.
    x_channel: Channel = Param(required=False)
    w_channel: Channel = Param(required=False)
    y_channel: Channel = Param(required=False)

    @constraint
    def carried(self) -> bool | Rejected:
        """Each supplied stream carries a tensor of the shape MatMul derives for it.

        On a stream that carries MatMul's own values (the results, and the
        weights when known: MatMul owns them, the stream's source streams them)
        MatMul's values fit the stream's element; on a stream it consumes, the
        stream's values fit MatMul's.
        """
        for reference, tensor in _CARRIED:
            if not self.present(getattr(MatMulKernel, reference)):
                continue
            supplied, derived_ = getattr(self, reference).tensor, getattr(self, tensor)
            produced = reference == "y_channel" or (reference == "w_channel" and self.known)
            inner, outer = (derived_, supplied) if produced else (supplied, derived_)
            if supplied.shape != derived_.shape or not inner.element.fits(outer.element):
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
        result_dtype=result_type,
        platform=platform,
        x_channel=x_channel,
        w_channel=w_channel,
        y_channel=y_channel,
    )

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.dense_view and not self.known:
            return reject(
                "matmul-realization",
                "a dense realization builds block-diagonal weights, so it needs known weights",
            )
        return True

    admission = ConstraintGroup(extents_supported, realization_supported, carried)

    def stem(self) -> str:
        return "finn_matmul"


__all__ = ["MatMulKernel", "exact_result_dtype"]
