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

``MatMulKernel`` is a kernel with children: the kernels and Decisions over
kernels that sit on the streams its parent supplies, ``x_stream`` (the
activations), ``w_stream`` (the weights), ``y_stream`` (the results) and, with
several weight sets, ``set_stream`` (the set index). Each stream, its adapter
and its FIFO are its parent's: the parent (a test harness, the graph front end)
declares each stream and either binds its tensor to MatMul's view of it
(``activation_tensor``, ``weight_tensor``, ``result_tensor``, ``set_tensor``),
which reads only MatMul's facts and ``realization``, never a port, or states
it; ``carried`` refuses a stated tensor of another shape, or one MatMul's
values do not fit (``matmul-tensor``). A boundary stream there presents its
``port`` name (``in0_V``).

Known ``weights`` are values MatMul owns: its ``weight_tensor`` states their
range (``INT4 over [-7, 7]``), and so does the memory that streams them, from
the same contents; a consumer derives from the range what it may (the packed
core's ``NARROW_WEIGHTS``). Unknown weights carry the datatype's range.

- ``compute`` is a Decision over the dot-product cores. They share the facts
  and streams. Each core owns its folding factors (``compute.<core>.pe``,
  ``.simd``, ``.compute_pumping``) and derives every stream's beat sequence
  from its schedule.
- ``memory`` is an optional Decision over the weight memories: none (the
  weight stream's producer is its parent's, a boundary for instance) or a
  ``memstream``, which drives the weight stream with one period of the order
  the core reads, per weight set, fixed at build time. Known weights are
  stored by a memory, and only they (``matmul-memory``).
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
    selected,
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
from finn.dataflow.traversal import Traversal, period
from finn.kernels.artifacts.module import ProducerIdentity
from finn.kernels.base import Kernel
from finn.kernels.datatypes.domains import set_index_dtype
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    integers,
)
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock


FinnAttributes = tuple[tuple[str, int | str | tuple[int, ...]], ...]

_CARRIED = (
    ("x_stream", "activation_tensor"),
    ("w_stream", "weight_tensor"),
    ("y_stream", "result_tensor"),
    ("set_stream", "set_tensor"),
)
"""Each stream MatMul sits on, and the tensor it must carry."""


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
    version = "1"

    m: int = Param()
    n: int = Param()
    k: int = Param()
    form: Form = Param(default=Form.DENSE)
    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp: DspBlock = Param()
    target_period_ns: float = Param()
    weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    # Several weight sets, one selected per row by an index on ``in2_V``;
    # ``weights`` then holds one operand per set.
    weight_sets: int = Param(default=1)

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
        values = integers(self.datapath_weights)
        low, high = ordinary_integer_bounds(self.weights_dtype)
        if not low <= min(values) <= max(values) <= high:
            return reject(
                "memstream-values",
                f"every value must be an integer admitted by {self.weights_dtype.name}",
            )
        element = ScalarEncoding.admit(self.weights_dtype, (min(values), max(values)))
        if isinstance(element, Rejected):
            return element
        return Tensor(shape, element)

    @view
    def result_tensor(self) -> Tensor | Rejected:
        return self._tensor((self.m, self.n), self.result_type)

    @view
    def set_tensor(self) -> Tensor | Rejected:
        """One set index per row, as wide as the memory's selector."""
        return self._tensor((self.m,), set_index_dtype(self.weight_sets))

    # The streams it sits on, supplied by its parent.
    x_stream: Stream = Param(required=False)
    w_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)
    set_stream: Stream = Param(required=False)

    @constraint
    def carried(self) -> bool | Rejected:
        """Each supplied stream carries a tensor of the shape MatMul derives for it, and
        MatMul's values fit its element."""
        for reference, tensor in _CARRIED:
            if not self.present(getattr(MatMulKernel, reference)):
                continue
            supplied, derived_ = getattr(self, reference).tensor, getattr(self, tensor)
            if supplied.shape != derived_.shape or not derived_.element.fits(supplied.element):
                return reject(
                    "matmul-tensor",
                    f"{reference} carries {supplied.shape} {supplied.element}; "
                    f"MatMul's {tensor} is {derived_.shape} {derived_.element}",
                )
        return True

    # The compute cores. Each refuses what its core cannot build and owns its
    # folding factors; ``packed`` names the entry, so references reach through it.
    packed = PackedDotpKernel()
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Decision(
        {"packed": packed, "int8_dsp58": Int8Dsp58DotpKernel},
        form=datapath,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        reshape_activations=dense_view,
        result_dtype=result_type,
        x_stream=x_stream,
        w_stream=w_stream,
        y_stream=y_stream,
    )

    @derived
    def weight_period(self) -> Traversal:
        """One pass of the weights in the order the core reads them: what a memory stores."""
        return period(self.compute.w.presented.form)

    # The weight memories. Each drives w_stream.
    memory: MemStreamKernel | None = Decision(
        {"memstream": MemStreamKernel(set_stream=set_stream)},
        optional=True,
        dtype=weights_dtype,
        form=weight_period,
        contents=datapath_weights,
        sets=weight_sets,
        output_stream=w_stream,
    )
    supplied = selected(memory)

    @constraint
    def supply_supported(self) -> bool | Rejected:
        if self.present(MatMulKernel.weights) != (self.supplied != "none"):
            return reject("matmul-memory", "known weights are stored by a memory, and only they")
        if self.supplied == "none" and self.multi_set:
            return reject("matmul-sets", "several weight sets need a memory")
        return True

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.dense_view and self.supplied == "none":
            return reject(
                "matmul-realization",
                "a dense realization builds block-diagonal weights, so it needs known weights",
            )
        return True

    admission = ConstraintGroup(extents_supported, realization_supported, supply_supported, carried)

    @view(requires=(admission,))
    def finn_attributes(self) -> FinnAttributes | Rejected:
        """FINN's ``MVAU`` node attributes, from this configuration alone.

        The memory maps onto FINN's ``mem_mode``: none is ``external``,
        memstream ``internal_decoupled`` (FINN's memstream); its memory style is
        ``ram_style``. A depthwise MatMul is FINN's ``VVAU``, not mapped yet.
        """
        if self.depthwise:
            return reject("finn-attributes", "a depthwise MatMul is FINN's VVAU, not mapped yet")
        memory = self.memory
        mode = {"none": "external", "memstream": "internal_decoupled"}
        style = "auto" if memory is None else memory.ram_style
        pumped = memory is not None and memory.pumped_memory
        compute = self.compute
        attributes: dict[str, int | str | tuple[int, ...]] = {
            "MW": self.k,
            "MH": self.n,
            "SIMD": compute.simd,
            "PE": compute.pe,
            "numInputVectors": (self.m,),
            "inputDataType": self.activation_dtype.name,
            "weightDataType": self.weights_dtype.name,
            "outputDataType": self.result_type.name,
            "accDataType": self.result_type.name,
            "mem_mode": mode[self.supplied],
            "ram_style": style,
            "pumpedMemory": int(pumped),
            "resType": "dsp",
            "noActivation": 1,
            "binaryXnorMode": 0,
            "backend": "fpgadataflow",
            "preferred_impl_style": "rtl",
        }
        return tuple(sorted(attributes.items()))

    def stem(self) -> str:
        return "finn_matmul_" + self.supplied

    def producer_identity(self) -> ProducerIdentity:
        return ProducerIdentity("finn.matmul." + self.supplied, type(self).version)


__all__ = ["FinnAttributes", "MatMulKernel", "exact_result_dtype"]
