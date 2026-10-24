# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A matrix-multiply unit on tensors: activations and weights in, results out.

Its facts are its tensors and the ``form`` (``finn.dataflow.gemm``). Its extents in
canonical GEMM notation, ``m`` rows, ``n`` outputs and the reduction ``k``, and the
activations' datatype are read from the tensors, each stated once: ``m``, ``k`` and
``activation_dtype`` from the activations (``m`` every leading axis together, at any
rank), ``n`` from the weights:

- ``DENSE``: Y[m, n] = sum over k of X[m, k] * W[k, n]. Every output reads the
  whole activation row, so the row is replayed once per output fold.
- ``DEPTHWISE``: Y[m, n] = sum over k of X[m, k, n] * W[k, n]. Each output
  channel reads its own activations, so nothing is replayed.

A dense operation may read its activations through a ``window``
(``finn.dataflow.gemm.Window``): a convolution lowered to a matrix multiplication,
``m`` the output pixels (OH * OW) and ``k`` the window ``(kh, kw, c)``, KH * KW * C.
Its activations are then the image, ``(H * W, C)`` or ``(1, H, W, C)`` at the graph's
rank, which the activation channel's adapter reads through the window (an
``input_gen``); its weights and results are the dense form's. Its results are stated at
the activations' rank (``leading``): ``(..., N)`` from ``(..., K)``, ``(1, OH, OW, N)``
from an image ``(1, H, W, C)``; what it requires of a given activation tensor is its
element count and last axis (``carried``).

Weights are read ``(k, n)``, as ONNX ``MatMul`` stores them. A depthwise
operation runs natively (one channel per PE lane, INT8 DSP58 only) or, by the
``realization`` Decision, on the dense datapath with block-diagonal weights
(``block_diagonal``), reading its (M, K, N) activations as (M, K * N).

``MatMulKernel`` is a kernel with children: the compute cores, which read the
activations' tensor it is given (``x_tensor``) and the weights' tensor it carries
(``carried_weights``), its folding factors (``pe``, ``simd``), and N and the widest K
(``widest_reduction``),
stated from its facts alone, and present its ports (``x``, ``w``, ``y``) as the selected
core's (``finn.kernels.port.served``): ``x_stream`` is the selected core's stream, and
``x_contract`` its contract, labelled ``compute.<core>.x``
(``finn.kernels.base.PortContract``). Each channel, its adapter, its FIFO and its source
are its network's: the network (a test harness's, or the graph's, a KernelOp's alone or
a partition's) declares each channel, gives it the tensor MatMul's port reads or
writes, and gives it MatMul's ports' streams, labels and widths. MatMul's own view of each tensor
(``activation_tensor``, ``weight_tensor``, and its results, ``y_tensor``) reads only its
facts and ``realization``, never a port; ``carried`` refuses a given input tensor that is
not what MatMul requires of it (the activations' element count and last axis, the
weights' shape), or whose values do not fit what MatMul reads (``matmul-tensor``). A
boundary channel there presents its ``port`` name (``in0_V``).

MatMul consumes its weights; it never holds them. Its weights are one fact,
``weights``: known at build (``KnownWeights``, given by the value's owner: their values
with the datatype the graph declares them in), or not (their ``Tensor``, which streams;
an initializer a run may write is read as one too). Known weights are its weight
channel's value: MatMul states what that channel carries, downstream of it.
``w_stored`` is the contents its source stores, block-diagonal for a dense realization
of a depthwise form (it may read ``realization``), and ``w_carried`` the tensor it
carries, over the stored values' range, narrowed to it (``stored_element``:
INT8-typed ternary weights are ``INT2``); without known weights, ``w_carried``
is the weights' tensor. Its network binds the weight channel's ``tensor`` and
``contents`` to the values the two present (``carried_weights``, ``stored_weights``),
whatever MatMul's validity: what a kernel states for the graph from its facts is a
value. With several weight sets the contents hold one operand a set, and the
channel's ``sets`` and ``index_tensor`` select one per row. Each core reads its
weights as ``w_carried`` presents them, the value ``carried_weights`` (a rule reads
values; a view is a product for its holder): its weight width, and the packed core's
``NARROW_WEIGHTS``. A dense realization
of a depthwise operation reads block-diagonal weights, so it needs known weights
(``matmul-realization``).

The result's range is derived from the weights and the activations' range (their
tensor's element: their datatype's, or the narrower range their producer states,
``operand_range``): known weights, the exact range of each output column's dot
products over the activations' range (``column_range``), unioned over the columns;
otherwise, every dot product of the two tensors' values. The result's type is the
range's smallest encoding, ``UINT`` when no value is negative (``range_dtype``,
``result_type``), whatever core computes it; the result element states the range
(``y_tensor``, presenting the value ``result_tensor``), which inference writes on the
graph with the type, so the next kernel reads it. Each core derives its accumulator
from the range (the core's ``result_range``) and states its results' tensor over it,
which its ``y`` port writes.

- ``pe`` and ``simd``, its folding factors, are MatMul's, the role's: declared before
  ``compute``, over the divisors of N and of the widest SIMD extent
  (``widest_simd_extent``), from its facts alone. Every core takes them (a switch of
  cores keeps them), and states what it requires of them as its rules. A core's own
  choices stay its own: pumping (``compute.<core>.compute_pumping``) and the packed
  core's reducer.
- ``compute`` is a selector over the dot-product cores (``Select``). They share
  the facts, the tensors and the folding, and each derives every port's beat sequence
  from its schedule.
- Where the weights come from is the weight channel's ``source``, not MatMul's.
- The steps between the activation channel's two streams replay each dense row and
  frame each reduction; the channel's adapter carries that out.

It binds no module of its own: its netlist (``finn.kernels.base.KernelSpace``) merges its
children's.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import cast

from qonnx.core.ranges import TensorRange

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Select,
    View,
    constraint,
    derived,
    divisors_of,
    reject,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    is_ordinary_integer,
    ordinary_integer_bounds,
)
from finn.dataflow.gemm import Form, Window
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import require_positive
from finn.kernels.base import KernelSpace
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.port import served
from finn.kernels.values.domains import (
    admit_element,
    integer_bounds,
    range_dtype,
    stored_element,
    values_within,
)
from finn.kernels.values.semantics import (
    IntegerTensor,
    IntegerTensorValue,
    integer_columns,
    integer_range,
    integer_shape,
    integers,
)
from finn.platform.capabilities import Platform


def operand_range(element: ScalarEncoding) -> tuple[int, int]:
    """The least and greatest value an operand of ``element`` takes in a dot product: its
    range (``integer_bounds``), widened to hold 0, so that every partial sum lies within
    the range of the whole (a term's range holds 0). Every integer datatype's own range
    holds 0 already; a narrower stated range may not (``[1, 3]``). An encoding that is not
    an ordinary integer is refused, as the cores refuse it (``ordinary_integer_bounds``)."""
    ordinary_integer_bounds(element.dtype)
    least, greatest = integer_bounds(element)
    return min(least, 0), max(greatest, 0)


def datatype_range(
    vector_length: int, activations: ScalarEncoding, weights: ScalarEncoding
) -> tuple[int, int]:
    """The least and greatest dot product of ``vector_length`` values of each element (its
    datatype over its range, ``operand_range``)."""
    require_positive(vector_length, "vector_length")
    activation, weight = operand_range(activations), operand_range(weights)
    products = tuple(a * w for a in activation for w in weight)
    return vector_length * min(products), vector_length * max(products)


def column_range(activations: ScalarEncoding, weights: IntegerTensor) -> tuple[int, int]:
    """The least and greatest dot product of any column of ``weights`` (``(..., k, n)``,
    reduced over ``k``) with activations of ``activations`` (an element: its datatype
    over its range, ``operand_range``).

    A column whose positive weights sum to P and negative ones to N meets activations
    in [lo, hi] (lo <= 0 <= hi, ``operand_range``) over [lo * P + hi * N,
    hi * P + lo * N], each term taking its extremes independently. Every partial sum
    lies within it too, since each term's range holds 0.
    """
    low, high = operand_range(activations)
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


@dataclass(frozen=True)
class KnownWeights:
    """Weights known at build: their values (``contents``), which the weight channel's
    source stores, and the datatype the graph declares them in (``dtype``), which the
    values narrow (``MatMulKernel.carried_weights``). A field of another type raises
    ``TypeError``, where the value is made."""

    contents: IntegerTensorValue
    dtype: QONNXDataType

    def __post_init__(self) -> None:
        for name, kind, label in (
            ("contents", IntegerTensorValue, "IntegerTensorValue"),
            ("dtype", QONNXDataType, "QONNX datatype"),
        ):
            found = getattr(self, name)
            if not isinstance(found, kind):
                raise TypeError(
                    f"known weights' {name}: expected {label}, given {type(found).__name__}"
                )


class MatMulKernel(KernelSpace):
    """Operation facts, and the kernels and selectors over kernels on its tensors."""

    id = "finn.matmul"

    form: Form = Param(default=Form.DENSE)
    # A dense form read through a window: a lowered convolution (module docstring).
    window: Window | None = Param(default=None)
    platform: Platform = Param()
    # The activations' tensor its port reads, and its weights, known or not: facts it is
    # given.
    x_tensor: Tensor = Param()
    weights: Tensor | KnownWeights = Param()

    @derived
    def stored(self) -> bool:
        """Whether the weights are known, which the weight channel's source stores."""
        return isinstance(self.weights, KnownWeights)

    @derived
    def depthwise(self) -> bool:
        return self.form is Form.DEPTHWISE

    @derived
    def windowed(self) -> bool:
        """Whether it reads its activations through a window."""
        return self.window is not None

    # Its extents and datatypes, read from its tensors: each is stated once, as a tensor.

    @derived
    def m(self) -> int:
        """M, the rows: a window's output pixels; otherwise every leading axis of the
        activations together, before their last (dense) or their last two (depthwise)."""
        if self.windowed:
            rows, columns = cast(Window, self.window).output
            return rows * columns
        return prod(self.x_tensor.shape[: -2 if self.depthwise else -1])

    @derived
    def k(self) -> int | Rejected:
        """K, the reduction: a window's taps times the image's channels, the activations'
        second last axis depthwise, else their last."""
        shape = self.x_tensor.shape
        if self.windowed:
            return cast(Window, self.window).tap_count * shape[-1]
        if self.depthwise:
            if len(shape) != 3:
                return reject("matmul-tensor", f"depthwise activations are (M, K, N), not {shape}")
            return shape[-2]
        return shape[-1]

    @derived
    def n(self) -> int | Rejected:
        """N, the outputs: the weights' last axis, known or streamed."""
        weights = self.weights
        if isinstance(weights, KnownWeights):
            shape = integer_shape(weights.contents)
            if not shape:
                return reject("matmul-tensor", "the known weights state no shape")
            return shape[-1]
        found: int = weights.shape[-1]
        return found

    @derived
    def activation_dtype(self) -> QONNXDataType:
        """The activations' datatype: their tensor's."""
        found: QONNXDataType = self.x_tensor.element.dtype
        return found

    @derived
    def weights_type(self) -> QONNXDataType:
        """The weights' datatype: known weights' as the graph declares them
        (``KnownWeights.dtype``), streamed weights' their tensor's."""
        weights = self.weights
        found: QONNXDataType = (
            weights.dtype if isinstance(weights, KnownWeights) else weights.element.dtype
        )
        return found

    @derived
    def reads_depthwise(self) -> bool:
        """Whether a core reads depthwise operands natively: only the INT8 DSP58 core
        does, within its own bounds on the platform's DSP and the datatypes."""
        return (
            Int8Dsp58DotpKernel.operand_refusal(
                self.platform.dsp, self.activation_dtype, self.weights_type
            )
            is None
        )

    # A depthwise operation runs natively (one channel per PE lane, a core that reads
    # it) or on the dense datapath with block-diagonal weights, on any core.
    realization: str = Decision(values=("native", "dense"), when=depthwise)

    @constraint(when=depthwise)
    def native_readable(self) -> bool | Rejected:
        if self.realization == "native" and not self.reads_depthwise:
            return reject(
                "matmul-native",
                "no core reads these depthwise operands natively "
                "(the INT8 DSP58 core alone does, within its bounds)",
            )
        return True

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
    def widest_reduction(self) -> int:
        """The widest K a datapath of the operation reads, from its facts alone: the window
        times the channels when depthwise (a dense realization reads that), else K. SIMD's
        domain reads it (``widest_simd_extent``), which reads no choice."""
        return self.k * self.n if self.depthwise else self.k

    @derived
    def widest_simd_extent(self) -> int:
        """The widest extent SIMD may divide, from facts alone: a window's channels (the
        activations' last axis), else the widest reduction. The datapath's extent, a core's
        ``simd_extent``, divides it."""
        found: int = self.x_tensor.shape[-1] if self.windowed else self.widest_reduction
        return found

    @derived
    def datapath_k(self) -> int:
        """The datapath's K: the window times the channels when densely realized."""
        return self.k * self.n if self.dense_view else self.k

    @derived
    def result_range(self) -> tuple[int, int] | Rejected:
        """The results' least and greatest value: with known weights, their columns' over
        the activations' range (``column_range``, per output column, of what the weight
        channel stores); without, every dot product of the two tensors' values
        (``datatype_range``). Each operand's range is its tensor's element's: its
        datatype's, or the narrower one its producer states."""
        try:
            if self.stored:
                return column_range(self.x_tensor.element, self.stored_weights)
            weights = cast(Tensor, self.weights)  # not stored: their tensor
            return datatype_range(self.k, self.x_tensor.element, weights.element)
        except ValueError as error:
            return reject("matmul-arithmetic", str(error))

    @derived
    def result_type(self) -> QONNXDataType:
        """The result range's smallest encoding: ``INT<b>``, or ``UINT<b>`` when no result
        is negative (``range_dtype``). No core's accumulator enters it."""
        least, greatest = self.result_range
        return range_dtype(least, greatest)

    @constraint
    def extents_supported(self) -> bool | Rejected:
        """A window reads one image of its own extent, from a dense form's activations:
        its ``(H * W, C)`` rows, or the image ``(..., H, W, C)`` at the graph's rank, every
        leading axis one."""
        if not self.windowed:
            return True
        window, shape = cast(Window, self.window), self.x_tensor.shape
        if self.depthwise:
            return reject("matmul-window", "a window reads a dense form's activations")
        (height, width) = window.image
        if len(shape) == 2:
            fits = shape[0] == height * width
        else:
            fits = len(shape) >= 3 and shape[-3:-1] == (height, width)
            fits = fits and all(extent == 1 for extent in shape[:-3])
        if not fits:
            return reject(
                "matmul-window",
                f"a {window.size} window over {window.image} reads one image of those "
                f"pixels, as (H * W, C) rows or (..., H, W, C); the activations are {shape}",
            )
        return True

    # MatMul's own views of its tensors: its facts and realization, never a port.

    def _tensor(
        self,
        shape: tuple[int, ...],
        dtype: QONNXDataType,
        value_range: TensorRange | None = None,
    ) -> Tensor | Rejected:
        element = admit_element(dtype, value_range)
        if isinstance(element, Rejected):
            return element
        try:
            return Tensor(shape, element)
        except ValueError as error:
            return reject("matmul-extents", str(error))

    @derived
    def required_activations(self) -> Tensor | Rejected:
        """(M, K), or (M, K, N) depthwise, however the datapath reads it; through a window,
        the image's (H * W, C) rows. What it requires of the activations' tensor: its
        element count and last axis (``carried``)."""
        shape: tuple[int, ...]
        if self.windowed:
            window = cast(Window, self.window)
            (height, width), taps = window.image, window.tap_count
            shape = (height * width, self.k // taps)
        else:
            shape = (self.m, self.k, self.n) if self.depthwise else (self.m, self.k)
        return self._tensor(shape, self.activation_dtype)

    @derived
    def required_weights(self) -> Tensor | Rejected:
        """(K, N) as the datapath reads it, in the weights' datatype: the weights' values
        must be values of it, in whatever encoding their owner states."""
        return self._tensor((self.datapath_k, self.n), self.weights_type)

    # The products its holder reads; ``carried`` reads the values they present.
    activation_tensor = View(required_activations)
    weight_tensor = View(required_weights)

    @derived
    def leading(self) -> tuple[int, ...]:
        """The results' axes before N, at the activations' rank: their leading axes
        (before their last two, depthwise); through a window, the pixels ``(OH * OW,)``,
        or ``(..., OH, OW)`` over an image at the graph's rank."""
        shape = self.x_tensor.shape
        if self.windowed:
            if len(shape) == 2:
                return (self.m,)
            return (*shape[:-3], *cast(Window, self.window).output)
        found: tuple[int, ...] = shape[: -2 if self.depthwise else -1]
        return found

    @derived
    def result_tensor(self) -> Tensor | Rejected:
        """Its results: ``(..., N)``, the activations' leading axes (``leading``) and N, in
        the result type over the result range. A value of its facts and its realization,
        which inference reads whatever MatMul's validity."""
        least, greatest = self.result_range
        return self._tensor((*self.leading, self.n), self.result_type, TensorRange(least, greatest))

    # The product its holder reads; inference reads the value it presents.
    y_tensor = View(result_tensor)

    @derived(when=stored)
    def stored_weights(self) -> IntegerTensorValue:
        """The known weights as the datapath reads them, which the weight channel's source
        stores: block-diagonal when a depthwise operation is realized densely."""
        weights = cast(KnownWeights, self.weights).contents  # stored: known
        return block_diagonal(weights) if self.dense_view else weights

    @derived
    def carried_weights(self) -> Tensor | Rejected:
        """The tensor the weight channel carries, which the cores read: with known weights,
        a set of the stored values, ``(K, N)``, over their range, narrowed to it
        (``stored_element``); otherwise the weights' tensor."""
        if not self.stored:
            return cast(Tensor, self.weights)  # not stored: their tensor
        stored = integer_shape(self.stored_weights)
        if not stored or len(stored) < 2:
            return reject("matmul-tensor", f"known weights are (K, N) a set, not {stored}")
        dtype, shape = self.weights_type, stored[-2:]
        if not is_ordinary_integer(dtype):
            return Tensor(shape, ScalarEncoding(dtype))
        least, greatest = integer_range(self.stored_weights)
        low, high = ordinary_integer_bounds(dtype)
        if not low <= least <= greatest <= high:
            return reject(
                "dtype-storage", f"every weight must be an integer admitted by {dtype.name}"
            )
        return Tensor(shape, stored_element(dtype, (least, greatest)))

    # What the weight channel's source stores and what it carries, the products its
    # holder reads; its network (``Fact(member, "carried_weights")``) and the cores read
    # the values.
    w_stored = View(stored_weights, when=stored)
    w_carried = View(carried_weights)

    @constraint
    def carried(self) -> bool | Rejected:
        """Each given tensor is what MatMul requires of it, and its values are values of
        MatMul's, in any integer encoding (``values_within``): the owner of known weights
        states them narrowed to their values. The activations are the element count and
        the last axis of ``activation_tensor``, at any rank; the weights its shape."""
        given: list[tuple[str, Tensor, str, bool]] = [
            ("x_tensor", self.x_tensor, "activation_tensor", False),
            (
                "w_carried" if self.stored else "weights",
                self.carried_weights,
                "weight_tensor",
                True,
            ),
        ]
        for name, supplied_, tensor, whole in given:
            # The weights are read whole, the activations by their count and last axis.
            derived_ = self.required_weights if whole else self.required_activations
            fits = (
                supplied_.shape == derived_.shape
                if whole
                else supplied_.size == derived_.size and supplied_.shape[-1] == derived_.shape[-1]
            )
            if not fits or not values_within(supplied_.element, derived_.element):
                return reject(
                    "matmul-tensor",
                    f"{name} is {supplied_.shape} {supplied_.element}; "
                    f"MatMul's {tensor} is {derived_.shape} {derived_.element}, "
                    "which it must fit",
                )
        return True

    # Its folding factors, the role's: declared before ``compute``, so that conversion
    # folds first and then picks the core. Their domains read facts only; a core's limit
    # on them is a rule of the core's (``simd_fits``, ``pumping_has_lanes``), since a
    # core's domain cannot read its parent's choice.
    pe: int = Decision(domain=divisors_of(n))
    simd: int = Decision(domain=divisors_of(widest_simd_extent))

    # The compute cores, given the folding. Each refuses what its core cannot build.
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Select(
        packed=PackedDotpKernel,
        int8_dsp58=Int8Dsp58DotpKernel,
        form=datapath,
        reshape_activations=dense_view,
        window=window,
        result_range=result_range,
        platform=platform,
        x_tensor=x_tensor,
        w_tensor=carried_weights,
        columns=n,
        widest_reduction=widest_reduction,
        pe=pe,
        simd=simd,
    )

    # Its ports: the selected core's activations, weights and results.
    x = served(compute, "x")
    w = served(compute, "w")
    y = served(compute, "y")

    @derived
    def takes_doubled_clock(self) -> bool:
        """Whether it takes the doubled clock: its selected core's (a pumped core), a case
        not selected never read."""
        core = self.compute
        return bool(core.takes_doubled_clock)

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.dense_view and not self.stored:
            return reject(
                "matmul-realization",
                "a dense realization reads block-diagonal weights, so it needs known weights",
            )
        return True


__all__ = [
    "KnownWeights",
    "MatMulKernel",
    "block_diagonal",
    "column_range",
    "datatype_range",
    "operand_range",
]
