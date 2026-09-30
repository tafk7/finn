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

``MatMulKernel`` is a graph of design spaces: ``Stream`` nodes (activations,
weights, results, and the set index with several weight sets) and the
kernels that sit on them. A stream with a single user is a boundary of the
kernel and presents its ``port`` name (``in0_V``, ``in1_V``, ``out0_V``,
``in2_V``).

- ``compute`` is a Decision over the dot-product cores. They share the facts
  and streams; the packed core also takes ``narrow_weights``. Each core owns its
  folds (``compute.<core>.pe``, ``.simd``, ``.compute_pumping``) and derives
  every stream's beat sequence from its schedule.
- ``memory`` is an optional Decision over the weight memories: none (the
  weight stream is the boundary ``in1_V``) or a ``memstream``, which stores
  one period of the order the core reads, per weight set, read-only unless
  the weights are writable.
- The activation stream's plan replays each dense row and frames each
  reduction; its adapter carries that out.

``structure`` wires ``Members(MODULE)`` through ``Members(CONNECTION)``; the
module has ``ap_clk2x`` only when compute is pumped. ``matmul_assembly`` is a
convenience adapter: it configures concrete facts, commits the caller's
choices, settles the rest and packs the views into a ``MatMulAssembly``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from finn.core.space import (
    Available,
    ConstraintGroup,
    Decision,
    Param,
    QueryResult,
    Rejected,
    constraint,
    derived,
    design_space,
    reject,
    selected,
    View,
    view,
)
from finn.core.space.settling import compatible_cases
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, period
from finn.kernels.artifacts.build import ModuleBuildRequirements
from finn.kernels.base import PORT
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.configure import admission, commit, describe, settle, undecided
from finn.kernels.control import ControlBus
from finn.kernels.datatypes.domains import set_index_dtype
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    integers,
)
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.physical.structure import PhysicalStructure
from finn.kernels.composite import Composite
from finn.kernels.streams import ADAPTER_RAM_STYLES, BufferedStream, Stream
from finn.kernels.target import DspBlock


FinnAttributes = tuple[tuple[str, int | str | tuple[int, ...]], ...]


class WeightDelivery(Enum):
    """Where the weights come from: the ``memory`` Decision's case for each."""

    EXTERNAL = "none"
    MEMSTREAM = "memstream"


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


@dataclass(frozen=True, slots=True)
class MatMulAssembly:
    activation_beats: int
    weight_beats: int
    result_beats: int
    result_dtype: QONNXDataType
    weight_delivery: WeightDelivery
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements
    initializer: tuple[int, ...]


class MatMulKernel(Composite):
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
    # Software rewrites the weights at run time through AXI-Lite.
    writable_weights: bool = Param(default=False)
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

    @derived
    def activation_tensor(self) -> Tensor | Rejected:
        """(M, K), or (M, K, N) depthwise, however the datapath reads it."""
        shape = (self.m, self.k, self.n) if self.depthwise else (self.m, self.k)
        return self._tensor(shape, self.activation_dtype)

    @derived
    def weight_tensor(self) -> Tensor | Rejected:
        return self._tensor((self.datapath_k, self.n), self.weights_dtype)

    @derived
    def result_tensor(self) -> Tensor | Rejected:
        return self._tensor((self.m, self.n), self.result_type)

    @derived
    def set_tensor(self) -> Tensor | Rejected:
        """One set index per row, as wide as the memory's selector."""
        return self._tensor((self.m,), set_index_dtype(self.weight_sets))

    # Streams: relations between the kernels that reference them. A stream with a
    # single user is a boundary of the kernel and presents its ABI port name.
    # The activations enter at in0_V, each row once; the stream's adapter
    # replays them for the core and closes each reduction with a frame marker.
    activations = Stream(tensor=activation_tensor, port="in0_V")
    weight_stream = BufferedStream(tensor=weight_tensor, port="in1_V")
    results = Stream(tensor=result_tensor, port="out0_V")
    set_index = Stream(tensor=set_tensor, port="in2_V", when=multi_set)

    # Placed in a parent, it sits on the parent's streams through these inputs,
    # each the boundary stream of the same tensor (``finn.kernels.composite``).
    x_stream: Stream = Param(required=False)
    w_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)
    set_stream: Stream = Param(required=False)
    boundaries = {
        "x_stream": "activations",
        "w_stream": "weight_stream",
        "y_stream": "results",
        "set_stream": "set_index",
    }
    x_port = View(activations.boundary, requires=(Composite.seated,))
    w_port = View(weight_stream.boundary, requires=(Composite.seated,))
    y_port = View(results.boundary, requires=(Composite.seated,))
    set_port = View(set_index.boundary, requires=(Composite.seated,))

    @derived
    def narrow_weights(self) -> bool:
        """Known weights that avoid their type's most negative value let the packed core
        pack more lanes (NARROW_WEIGHTS). Provisional: the user means to revisit it."""
        read_only = self.supplied != WeightDelivery.EXTERNAL.value and not self.writable_weights
        if not read_only:
            return False  # weights arriving or rewritten at run time promise nothing
        low, _ = ordinary_integer_bounds(self.weights_dtype)
        return all(value > low for value in integers(self.weights))

    # The compute cores. Each refuses what its core cannot build and owns its
    # folds; ``packed`` names the entry for its own binding.
    packed = PackedDotpKernel(narrow_weights=narrow_weights)
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Decision(
        {"packed": packed, "int8_dsp58": Int8Dsp58DotpKernel},
        form=datapath,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        reshape_activations=dense_view,
        x_stream=activations,
        w_stream=weight_stream,
        y_stream=results,
    )

    @derived
    def weight_period(self) -> Traversal:
        """One pass of the weights in the order the core reads them: what a memory stores."""
        return period(self.compute.w.presented.form)

    # The weight memories. Each references weight_stream as its producer, so only
    # when one is selected is the stream internal; a writable memstream exports
    # its AXI-Lite port through ``config`` (s_axilite).
    config = ControlBus(port="s_axilite")
    memory: MemStreamKernel | None = Decision(
        {"memstream": MemStreamKernel(set_stream=set_index, control=config)},
        optional=True,
        dtype=weights_dtype,
        form=weight_period,
        contents=datapath_weights,
        writable=writable_weights,
        sets=weight_sets,
        output_stream=weight_stream,
    )
    supplied = selected(memory)

    @constraint
    def supply_supported(self) -> bool | Rejected:
        if self.supplied == WeightDelivery.EXTERNAL.value:
            if self.writable_weights:
                return reject("matmul-writable", "runtime-writable weights need a memory")
            if self.multi_set:
                return reject("matmul-sets", "several weight sets need a memory")
        return True

    @constraint
    def realization_supported(self) -> bool | Rejected:
        if self.dense_view and self.supplied == WeightDelivery.EXTERNAL.value:
            return reject(
                "matmul-realization",
                "a dense realization builds block-diagonal weights, so it needs known weights",
            )
        return True

    admission = ConstraintGroup(extents_supported, realization_supported, supply_supported)

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
            "runtime_writeable_weights": int(self.writable_weights),
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
        return ProducerIdentity("finn.matmul." + self.supplied, "1")

    exports = {
        **Composite.exports,
        PORT: {x_stream: x_port, w_stream: w_port, y_stream: y_port, set_stream: set_port},
    }


def _frozen(values: object) -> object:
    """Nested sequences as nested tuples."""
    if isinstance(values, Sequence):
        return tuple(_frozen(item) for item in values)
    return values


def _realizes(base: MatMulKernel, choices: dict[str, object]) -> QueryResult[bool]:
    """Accepted when the choices commit, the realization's own rule holds, and some
    core can compute it."""
    try:
        point = commit(base, choices)
    except ValueError as error:
        return reject("matmul-realization", str(error))
    rule = point.inspect(MatMulKernel.realization_supported).result
    if not isinstance(rule, Available):
        return rule
    cores = compatible_cases(point, "compute", admission)
    return Available(True) if cores else reject("matmul-realization", "no core computes it")


def matmul_assembly(
    *,
    m: int,
    n: int,
    k: int,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    pe: int,
    simd: int,
    target_dsp: DspBlock,
    form: Form = Form.DENSE,
    target_period_ns: float = 5.0,
    compute_pumping: bool = False,
    core: str | None = None,
    realization: str | None = None,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[object] | None = None,
    ram_style: str = "auto",
    pumped_memory: bool = False,
    writable_weights: bool = False,
    weight_sets: int = 1,
    weight_fifo_depth: int | None = None,
) -> MatMulAssembly:
    """Bind operation facts, commit the caller's choices, settle the rest, then assemble.

    ``m`` rows, ``n`` outputs and the reduction ``k``; for a depthwise ``form``,
    ``k`` is the window and ``n`` the channels. ``weights`` is stored (K, N),
    and is required by, and only accepted with, a memory. The ``auto``
    ``ram_style`` default leaves memory inference to synthesis.
    ``weight_fifo_depth`` places a FIFO on the weight stream; ``None`` connects
    it directly. ``target_period_ns`` is the clock the module must meet (5 ns:
    200 MHz); it sets dotp's DSP58 chain segmentation. ``core`` names the
    compute core (``packed`` or ``int8_dsp58``); left out, the one core
    compatible with the configuration is settled, and several compatible cores
    must be chosen from. PE, SIMD and pumping are the core's.
    """
    if not isinstance(weight_delivery, WeightDelivery):
        raise ValueError("weight_delivery must be a WeightDelivery value")
    known = weight_delivery is not WeightDelivery.EXTERNAL
    if known != (weights is not None):
        raise ValueError("stored delivery requires weights; external delivery has no initializer")
    facts: dict[str, Any] = dict(
        m=m,
        n=n,
        k=k,
        form=form,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        writable_weights=writable_weights,
        weight_sets=weight_sets,
    )
    if weights is not None:
        facts["weights"] = _frozen(weights)
    case = weight_delivery.value
    buffered = weight_fifo_depth is not None
    choices: dict[str, object] = {
        "memory": case,
        "weight_stream.transport": "fifo" if buffered else "direct",
    }
    if weight_delivery is WeightDelivery.MEMSTREAM:
        choices["memory.memstream.ram_style"] = ram_style
        choices["memory.memstream.pumped_memory"] = pumped_memory
    if buffered:
        choices["weight_stream.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["weight_stream.transport.fifo.buffer.ram_style"] = "auto"
    base = design_space(MatMulKernel(**facts))
    if form is Form.DEPTHWISE:
        # The realization sets the datapath's reduction, so it is committed with
        # the other choices before the core's folds.
        if realization is None:
            viable = [
                case
                for case in ("native", "dense")
                if isinstance(_realizes(base, {**choices, "realization": case}), Available)
            ]
            if len(viable) != 1:
                named = ", ".join(viable) or "none"
                raise ValueError(f"realizations compatible with this configuration: {named}")
            realization = viable[0]
        choices["realization"] = realization
    point = commit(base, choices)
    if core is None:
        settled = settle(point)
        if "compute" not in settled.committed:
            cores = settled.open.get("compute", ())
            if cores:
                raise ValueError(f"compute cores {', '.join(cores)} are all compatible; choose one")
            refusals = (
                admission(commit(point, {"compute": case}).compute)
                for case in ("packed", "int8_dsp58")
            )
            found = describe(result for result in refusals if result is not None)
            raise ValueError(f"no compute core is compatible: {found}")
        point, core = settled.point, settled.committed["compute"]
    else:
        point = commit(point, {"compute": core})
    point = commit(
        point,
        {
            f"compute.{core}.pe": pe,
            f"compute.{core}.simd": simd,
            f"compute.{core}.compute_pumping": compute_pumping,
        },
    )
    # Each stream's one compatible adapter; an input_gen's memory is inferred.
    point = settle(point).point
    styles = undecided(point, ADAPTER_RAM_STYLES)
    if styles:
        point = commit(point, dict.fromkeys(styles, "auto"))
    composed = point.query(MatMulKernel.structure)
    if not isinstance(composed, Available):
        raise ValueError(f"MatMul assembly is not accepted: {describe([composed])}")
    compute = point.compute
    return MatMulAssembly(
        point.activations.ends.source.sequence.form.beats,
        compute.w.presented.form.beats,
        compute.y.presented.form.beats,
        point.result_type,
        weight_delivery,
        composed.value.structure,
        composed.value.requirements,
        point.memory.image if point.memory is not None else (),
    )


__all__ = [
    "FinnAttributes",
    "MatMulAssembly",
    "MatMulKernel",
    "WeightDelivery",
    "exact_result_dtype",
    "matmul_assembly",
]
