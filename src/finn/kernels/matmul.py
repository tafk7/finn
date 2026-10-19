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
tensors it is given, ``x_tensor`` (the activations) and ``w_tensor`` (the weights),
and N and the widest K (``widest_reduction``), stated from its facts alone for their
folding factors' domains, and present its interfaces (``x``, ``w``, ``y``): ``x_end``,
``w_end`` and ``y_end`` are the selected core's ports' ends, labelled ``compute.<core>.x``
(``finn.kernels.base.PortEnd``). Each channel, its adapter, its FIFO and its source
are its network's: the network (a test harness's, or the graph's, a KernelOp's alone or
a partition's) declares each channel, gives it the tensor MatMul's port reads or
writes, and gives it MatMul's ends. MatMul's own view of each tensor (``activation_tensor``,
``weight_tensor``, and its results, ``y_tensor``) reads only its facts and
``realization``, never a port; ``carried`` refuses a given input tensor that is not
what MatMul requires of it (the activations' element count and last axis, the
weights' shape), or whose values do not fit what MatMul reads (``matmul-tensor``). A
boundary channel there presents its ``port`` name (``in0_V``).

MatMul consumes its weights; it never holds them. Known weights are a fact
(``w_contents``, given by the value's owner, with the datatype the graph declares
them in, ``weights_dtype``), and its weight channel's value:
MatMul states what that channel carries, downstream of it. ``w_stored`` is the
contents its source stores, block-diagonal for a dense realization of a
depthwise form (it may read ``realization``), and ``w_carried`` the tensor it
carries, over the stored values' range, narrowed to it (``stored_element``:
INT8-typed ternary weights are ``INT2``); without known weights, ``w_carried``
is ``w_tensor``. Its network binds the weight channel's ``tensor`` and
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

- ``compute`` is a Decision over the dot-product cores. They share the facts
  and tensors. Each core owns its folding factors (``compute.<core>.pe``,
  ``.simd``, ``.compute_pumping``) and derives every port's beat sequence
  from its schedule.
- Where the weights come from is the weight channel's ``source``, not MatMul's.
- The activation channel's plan replays each dense row and frames each
  reduction; the channel's adapter carries that out.

Its ``module`` (``finn.kernels.base``) merges its children's netlists.
"""

from __future__ import annotations

from math import prod

from qonnx.core.ranges import TensorRange

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    View,
    constraint,
    derived,
    reject,
    selected,
    supplied,
    view,
)
from finn.dataflow.datatypes import (
    QONNXDataType,
    is_ordinary_integer,
    ordinary_integer_bounds,
)
from finn.dataflow.gemm import Form, Window
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import require_positive
from finn.kernels.base import PORT_END, ComposedKernel, PortEnd
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.values.domains import (
    admit_element,
    integer_bounds,
    range_dtype,
    stored_element,
    values_within,
)
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
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


class MatMulKernel(ComposedKernel):
    """Operation facts, and the kernels and Decisions over kernels on its tensors."""

    id = "finn.matmul"
    version = 1
    interfaces = ("x", "w", "y")

    form: Form = Param(default=Form.DENSE)
    # A dense form read through a window: a lowered convolution (module docstring).
    window: Window = Param(required=False)
    platform: Platform = Param()
    # The tensors its ports read, and the known weights: facts it is given.
    x_tensor: Tensor = Param()
    w_tensor: Tensor = Param(required=False)
    w_contents: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    # Known weights' datatype, the graph's, which their values narrow (``w_carried``);
    # streamed weights' is their tensor's.
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    # Whether the weights are known, which the weight channel's source stores.
    stored = supplied(w_contents)

    @derived
    def depthwise(self) -> bool:
        return self.form is Form.DEPTHWISE

    @derived
    def windowed(self) -> bool:
        """Whether it reads its activations through a window."""
        return self.present(MatMulKernel.window)

    # Its extents and datatypes, read from its tensors: each is stated once, as a tensor.

    @derived
    def m(self) -> int:
        """M, the rows: a window's output pixels; otherwise every leading axis of the
        activations together, before their last (dense) or their last two (depthwise)."""
        if self.windowed:
            rows, columns = self.window.output
            return rows * columns
        return prod(self.x_tensor.shape[: -2 if self.depthwise else -1])

    @derived
    def k(self) -> int | Rejected:
        """K, the reduction: a window's taps times the image's channels, the activations'
        second last axis depthwise, else their last."""
        shape = self.x_tensor.shape
        if self.windowed:
            return self.window.tap_count * shape[-1]
        if self.depthwise:
            if len(shape) != 3:
                return reject("matmul-tensor", f"depthwise activations are (M, K, N), not {shape}")
            return shape[-2]
        return shape[-1]

    @derived
    def n(self) -> int | Rejected:
        """N, the outputs: the weights' last axis, known or streamed."""
        if self.stored:
            shape = integer_shape(self.w_contents)
            if not shape:
                return reject("matmul-tensor", "the known weights state no shape")
            return shape[-1]
        if self.present(MatMulKernel.w_tensor):
            found: int = self.w_tensor.shape[-1]
            return found
        return reject("matmul-tensor", "MatMul reads its weights: known (w_contents) or a tensor")

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def activation_dtype(self) -> QONNXDataType:
        """The activations' datatype: their tensor's."""
        found: QONNXDataType = self.x_tensor.element.dtype
        return found

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def weights_type(self) -> QONNXDataType | Rejected:
        """The weights' datatype: known weights' as the graph declares them
        (``weights_dtype``), streamed weights' their tensor's."""
        given = self.present(MatMulKernel.weights_dtype)
        if self.stored:
            if not given:
                return reject("matmul-tensor", "known weights state their datatype (weights_dtype)")
            declared: QONNXDataType = self.weights_dtype
            return declared
        if given:
            return reject(
                "matmul-tensor",
                "streamed weights' datatype is their tensor's: weights_dtype is known weights'",
            )
        found: QONNXDataType = self.w_tensor.element.dtype
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
        times the channels when depthwise (a dense realization reads that), else K. The
        cores' SIMD domains read it, which read no choice."""
        return self.k * self.n if self.depthwise else self.k

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
            return datatype_range(self.k, self.x_tensor.element, self.w_tensor.element)
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
        """A window reads one image of its own extent, from a dense form's activations:
        its ``(H * W, C)`` rows, or the image ``(..., H, W, C)`` at the graph's rank, every
        leading axis one."""
        if not self.windowed:
            return True
        window, shape = self.window, self.x_tensor.shape
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
            (height, width), taps = self.window.image, self.window.tap_count
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
            return (*shape[:-3], *self.window.output)
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

    @derived
    def weighted(self) -> bool:
        """Whether its weights are placed: known, or a weight tensor given."""
        return self.stored or self.present(MatMulKernel.w_tensor)

    @derived(when=stored, semantics=INTEGER_TENSOR)
    def stored_weights(self) -> IntegerTensor:
        """The known weights as the datapath reads them, which the weight channel's source
        stores: block-diagonal when a depthwise operation is realized densely."""
        weights = self.w_contents
        return block_diagonal(weights) if self.dense_view else weights

    @derived(when=weighted)
    def carried_weights(self) -> Tensor | Rejected:
        """The tensor the weight channel carries, which the cores read: with known weights,
        a set of the stored values, ``(K, N)``, over their range, narrowed to it
        (``stored_element``); otherwise ``w_tensor``."""
        if not self.stored:
            tensor: Tensor = self.w_tensor
            return tensor
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
    w_carried = View(carried_weights, when=weighted)

    @constraint
    def carried(self) -> bool | Rejected:
        """Each given tensor is what MatMul requires of it, and its values are values of
        MatMul's, in any integer encoding (``values_within``): the owner of known weights
        states them narrowed to their values. The activations are the element count and
        the last axis of ``activation_tensor``, at any rank; the weights its shape."""
        given: list[tuple[str, Tensor, str, bool]] = [
            ("x_tensor", self.x_tensor, "activation_tensor", False)
        ]
        if self.weighted:
            weights = "w_carried" if self.stored else "w_tensor"
            given.append((weights, self.carried_weights, "weight_tensor", True))
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

    # The compute cores. Each refuses what its core cannot build and owns its
    # folding factors; ``packed`` names the entry.
    packed = PackedDotpKernel()
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Decision(
        {"packed": packed, "int8_dsp58": Int8Dsp58DotpKernel},
        form=datapath,
        reshape_activations=dense_view,
        window=window,
        result_range=result_range,
        platform=platform,
        x_tensor=x_tensor,
        w_tensor=carried_weights,
        columns=n,
        widest_reduction=widest_reduction,
    )
    compute_case = selected(compute)

    def _served(self, end: PortEnd) -> PortEnd:
        """The selected core's ``end``, labelled below MatMul (``compute.packed.x``)."""
        return end.under(f"compute.{self.compute_case}")

    @view(semantics=PORT_END)
    def x_end(self) -> PortEnd:
        """The activations' interface: the selected core's ``x``."""
        return self._served(self.compute.x_end)

    @view(semantics=PORT_END)
    def w_end(self) -> PortEnd:
        """The weights' interface: the selected core's ``w``."""
        return self._served(self.compute.w_end)

    @view(semantics=PORT_END)
    def y_end(self) -> PortEnd:
        """The results' interface: the selected core's ``y``."""
        return self._served(self.compute.y_end)

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.dense_view and not self.stored:
            return reject(
                "matmul-realization",
                "a dense realization reads block-diagonal weights, so it needs known weights",
            )
        return True


__all__ = [
    "MatMulKernel",
    "block_diagonal",
    "column_range",
    "datatype_range",
    "operand_range",
]
