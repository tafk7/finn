# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test construction: configure a kernel node from its formals, then commit choices by key.

Facts are the root node's typed formals; a missing required one is refused at
the node call. Choices use the stable decision keys ``inspection`` reports.

A kernel with children sits on channels its parent supplies, so a test places
it in a ``Root``, the top of what is emitted, which declares them (as
``placed_dotp`` does for a dot-product core). ``placed_matmul`` places a
MatMul (``matmul``) on the root's ``x`` (``in0_V``), ``w`` (``in1_V``,
buffered), ``y`` (``out0_V``) and, with several weight sets, ``set``
(``in2_V``). The root declares the channels and binds each one's tensor to
MatMul's view of it (``activation_tensor``, ``weight_tensor``,
``result_tensor``), which reads only MatMul's facts and its ``realization``:
one root serves every realization, a depthwise MatMul's left open until
committed. Known weights are the root's: the weight channel's ``contents``
(block-diagonal for a dense realization), its tensor over their range, with
``set`` its ``index`` (one index per row), so the channel's ``source`` stores
them. The edge choices are the root's (``x.adapter``,
``w.transport``, ``w.source.memstream.ram_style``), the MatMul's below it
(``matmul.compute.packed.pe``). ``matmul_assembly`` configures one from
concrete facts and choices.

What a simulation runs on (FinnLib, Vivado, the run's identity) is
``finn.harness.toolchain``'s: this module only constructs."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any, TypeVar, cast

from finn.core.space import (
    Constraint,
    Param,
    Rejected,
    Space,
    composite,
    derived,
    design_space,
    inspection,
    reject,
    supplied,
)
from finn.core.space.results import Available, QueryResult
from finn.dataflow.datatypes import (
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import commit, describe, undecided
from finn.kernels.control import ControlBus
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.explore import Choice
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.matmul import MatMulKernel, block_diagonal
from finn.kernels.target import DspBlock, Fabric, Platform
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.domains import set_index_dtype, stored_element
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerVector,
    ThresholdTable,
    integer_range,
)

T = TypeVar("T")
S = TypeVar("S", bound=Space)

XSIM_KEY = "construction"
"""How ``scripts/emitted_text.py`` keys an XSim job that imports this module: by what
it constructs, not by its code. What it builds reaches a simulation only through the
captured designs and, for a numeric sweep, the results of the calls the sweep makes
here, which the capture records. Code whose effect reaches a verdict otherwise is
harness (``finn.harness``)."""


def full_platform(dsp: DspBlock, *, period_ns: float = 5.0) -> Platform:
    """The platform a bare-kernel test means when it is not about the platform: ``dsp``
    its DSP block, a ``period_ns`` clock (5 ns: 200 MHz), and every capability (UltraRAM
    that takes initial contents, a doubled clock) on an UltraScale fabric, its
    resources not stated (no kernel reads them)."""
    return Platform(
        period_ns=period_ns,
        dsp=dsp,
        fabric=Fabric.ULTRASCALE,
        uram=True,
        uram_init=True,
        resources=None,
    )


FULL_DSP48E2 = full_platform(DspBlock.DSP48E2)
FULL_DSP58 = full_platform(DspBlock.DSP58)


def point_for(kernel: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    return commit(design_space(kernel(**facts)), choices)


def value(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> QueryResult[bool]:
    return point.inspect(condition).result


def placed_dotp(
    space_type: Callable[..., S],
    *,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    result_dtype: QONNXDataType,
    result_range: tuple[int, int] | None = None,
    weights_range: tuple[int, int] | None = None,
    pe: object = None,
    simd: object = None,
    compute_pumping: object = False,
    reducer: object = "tree",
    rows: int = 1,
    outputs: int | None = None,
    reduction: int | None = None,
    **facts: object,
) -> S:
    """A dot-product core between three boundary channels, its folding factors committed.

    The core takes its extents from the channels: ``outputs`` (N) defaults to PE
    and ``reduction`` (K) to SIMD, one fold each. A folding factor left ``None``
    stays open, as does ``reducer``, which only a core that declares it commits.
    ``weights_range`` is the range of values the weight channel carries; the
    datatype's own by default. ``result_range`` is the results' range, from which the core
    derives its accumulator; ``result_dtype``'s own by default. The results channel
    carries ``result_dtype`` over it.
    """
    form = facts.get("form", Form.DENSE)
    n = outputs if outputs is not None else (pe if isinstance(pe, int) and pe > 0 else 1)
    k = reduction if reduction is not None else (simd if isinstance(simd, int) and simd > 0 else 1)
    x_shape = (rows, k, n) if form is Form.DEPTHWISE else (rows, k)
    # The channels are on the core's platform; a test that omits it gets FULL_DSP48E2's.
    platform = cast(Platform, facts.get("platform", FULL_DSP48E2))

    class Placed(Space):
        x = Channel(
            tensor=Tensor(x_shape, ScalarEncoding(activation_dtype)),
            port="in0_V",
            platform=platform,
        )
        w = Channel(
            tensor=Tensor((k, n), ScalarEncoding(weights_dtype, weights_range)),
            port="in1_V",
            platform=platform,
        )
        y = Channel(
            tensor=Tensor((rows, n), ScalarEncoding(result_dtype, result_range)),
            port="out0_V",
            platform=platform,
        )
        compute = space_type(
            x_channel=x,
            w_channel=w,
            y_channel=y,
            result_range=result_range or ordinary_integer_bounds(result_dtype),
            **facts,
        )

    choices = {
        key: value
        for key, value in (
            ("compute.pe", pe),
            ("compute.simd", simd),
            ("compute.compute_pumping", compute_pumping),
            ("compute.reducer", reducer if hasattr(space_type, "reducer") else None),
        )
        if value is not None
    }
    point = design_space(Placed())
    placed = commit(point, choices)
    return placed.compute


class Root(Kernel):
    """The top of what a test emits: the channels it declares and the kernels on them."""

    id = "test.root"
    version = 1


def rooted(name: str, members: Mapping[str, object]) -> Root:
    """A root named ``name`` (its module's stem ``finn_<name>``) of ``members``."""
    space_type: Any = composite(name, dict(members), base=Root)
    root: Root = space_type()
    return root


def stated(tensor: Tensor, value: IntegerTensor) -> Tensor | Rejected:
    """``tensor`` as the owner of ``value`` states it for the channel carrying it: over the
    value's range, in the encoding it needs (``stored_element``), which its datatype must
    admit."""
    dtype = tensor.element.dtype
    low, high = ordinary_integer_bounds(dtype)
    least, greatest = integer_range(value)
    if not low <= least <= greatest <= high:
        return reject("dtype-storage", f"every value must be an integer admitted by {dtype.name}")
    return Tensor(tensor.shape, stored_element(dtype, (least, greatest)))


@cache
def matmul_root(space_type: type[MatMulKernel]) -> type[Root]:
    """A root placing a MatMul of ``space_type`` (``matmul``) on the channels it declares, its
    facts its own formals, the channels' tensors MatMul's views and the weights its own;
    see the module docstring."""

    class MatMul(Root):
        m: int = Param()
        n: int = Param()
        k: int = Param()
        form: Form = Param(default=Form.DENSE)
        activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        platform: Platform = Param()
        # The weights are the root's (the value owner's), one operand a set.
        weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
        weight_sets: int = Param(default=1)
        known = supplied(weights)

        @derived
        def several(self) -> bool:
            return self.weight_sets > 1

        # Each channel's tensor is MatMul's view of it: its facts, never its ports.
        @derived
        def x_tensor(self) -> Tensor:
            return self.matmul.activation_tensor

        @derived(when=known, semantics=INTEGER_TENSOR)
        def w_contents(self) -> IntegerTensor:
            """The weights as the datapath reads them: block-diagonal when a depthwise
            operation is realized densely."""
            return block_diagonal(self.weights) if self.matmul.dense_view else self.weights

        @derived
        def w_tensor(self) -> Tensor | Rejected:
            """MatMul's view of its weights, over the range of the value the root states."""
            tensor = self.matmul.weight_tensor
            return stated(tensor, self.w_contents) if self.present(MatMul.weights) else tensor

        @derived
        def y_tensor(self) -> Tensor:
            return self.matmul.result_tensor

        @derived
        def set_tensor(self) -> Tensor:
            """One set index per row, as wide as the weight source's selector."""
            return Tensor((self.m,), ScalarEncoding(set_index_dtype(self.weight_sets)))

        x = Channel(tensor=x_tensor, port="in0_V", platform=platform)
        set = Channel(tensor=set_tensor, port="in2_V", when=several, platform=platform)
        w = Channel(
            tensor=w_tensor,
            contents=w_contents,
            sets=weight_sets,
            index=set,
            port="in1_V",
            platform=platform,
        )
        y = Channel(tensor=y_tensor, port="out0_V", platform=platform)
        matmul = space_type(
            m=m,
            n=n,
            k=k,
            form=form,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
            platform=platform,
            x_channel=x,
            w_channel=w,
            y_channel=y,
        )

    return MatMul


def placed_matmul(**facts: object) -> Root:
    """A MatMul (``matmul``) in a root declaring its channels; see the module docstring."""
    space_type: Any = matmul_root(MatMulKernel)
    root: Root = space_type(**facts)
    return root


def matmul_point(*, realization: str | None = None, **facts: object) -> Any:
    """``placed_matmul`` as a design space, its ``realization`` committed when given."""
    point = design_space(placed_matmul(**facts))
    return commit(point, {"matmul.realization": realization}) if realization else point


def controlled(space_type: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    """A kernel (``kernel``) presenting its configuration bus through a ``ControlBus``, its
    choices committed by their keys below it; as placed alone it holds the bus."""

    class Controlled(Space):
        config = ControlBus(port="s_axilite")
        kernel = space_type(**facts, control=config)

    point = commit(design_space(Controlled()), {f"kernel.{key}": v for key, v in choices.items()})
    kernel: S = point.kernel
    return kernel


THRESHOLD_TABLE: ThresholdTable = (((-2, 0, 3), (-1, 1, 4)),)
"""One threshold table: two channels of three thresholds."""


def threshold_base(
    *,
    table: ThresholdTable = THRESHOLD_TABLE,
    bias: int = -1,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT5",
) -> ThresholdingAxiKernel:
    """A thresholding's design space, its choices open."""
    return design_space(
        ThresholdingAxiKernel(
            input_dtype=resolve_qonnx_datatype_name(input_dtype),
            threshold_dtype=resolve_qonnx_datatype_name(threshold_dtype),
            thresholds=table,
            bias=bias,
            platform=FULL_DSP48E2,
        )
    )


def generator(
    *,
    bits: int = 13,
    frame: int = 6,
    dims: IntegerVector = (3, 6),
    strides: IntegerVector = (0, 1),
) -> InputGeneratorKernel:
    """An input generator's design space, its memory Vivado's."""
    return design_space(
        InputGeneratorKernel(
            word_bits=bits, frame_words=frame, dims=dims, strides=strides, platform=FULL_DSP48E2
        )
    ).with_choices(ram_style="auto")


def eltwise(
    *,
    operation: str = "ADD",
    pe: int = 2,
    lhs: str = "INT3",
    rhs: str = "INT3",
    scale: float = 1.0,
    target: DspBlock = DspBlock.DSP58,
) -> EltwiseKernel:
    """An elementwise kernel's design space, on a platform of ``target``."""
    return design_space(
        EltwiseKernel(
            operation=operation,
            pe=pe,
            lhs_dtype=resolve_qonnx_datatype_name(lhs),
            rhs_dtype=resolve_qonnx_datatype_name(rhs),
            b_scale=scale,
            platform=full_platform(target),
        )
    )


def codes(result: object) -> set[str]:
    """The codes of the findings of ``result``, which must be refused."""
    assert isinstance(result, Rejected), result
    return {finding.code for finding in result.findings}


def labels(module: Composed) -> list[str]:
    """A composed module's instance labels, in netlist order."""
    return [label for label, _ in module.fragment.instances]


def placed(module: Composed, label: str) -> Leaf:
    """The leaf a composed module places at ``label``."""
    return dict(module.fragment.instances)[label]


def pin_names(module: Composed | Leaf) -> set[str]:
    return {port.name for port in module.abi.pins}


ADAPTER_RAM_STYLES = "*.adapter.*.ram_style"
"""The keys (``fnmatch``) of every adapter stage's memory choice, an ``input_gen``'s."""


def with_adapter_memories(point: S, ram_style: str = "auto") -> S:
    """Each open adapter input_gen's memory takes ``ram_style``: the flow's choice. Each
    channel's adapter, its one viable chain, is forced."""
    styles = undecided(point, ADAPTER_RAM_STYLES)
    return commit(point, dict.fromkeys(styles, ram_style)) if styles else point


TRANSPORTS = "*.transport"
"""The keys (``fnmatch``) of every channel's transport choice."""


def with_direct_transports(point: S) -> S:
    """Each channel's open transport direct: its baseline, the first case."""
    open_ = undecided(point, TRANSPORTS)
    return commit(point, dict.fromkeys(open_, "direct")) if open_ else point


class Lanes:
    """A rank policy for tests (``finn.kernels.explore.Ranked``) that folds by hand: an
    ordered choice of integers at ``lanes`` where that case is viable, otherwise at its
    largest viable case; every other choice, and every choice with no ``lanes``, in its
    domain's order, the kernel's baseline first. A test that needs a choice made on
    purpose states it so; ``Lanes()`` ranks as the baseline completion takes."""

    def __init__(self, lanes: int | None = None) -> None:
        self.lanes = lanes

    def rank(self, choice: Choice) -> Sequence[object]:
        cases = choice.cases or ()
        if self.lanes is None or not choice.ordered:
            return cases
        factors = sorted((case for case in cases if isinstance(case, int)), reverse=True)
        return sorted(factors, key=lambda factor: factor != self.lanes)


class WeightDelivery(Enum):
    """Where the weights come from: the weight channel's ``source`` case, or the
    boundary (external)."""

    EXTERNAL = "none"
    MEMSTREAM = "memstream"


@dataclass(frozen=True, slots=True)
class MatMulAssembly:
    activation_beats: int
    weight_beats: int
    result_beats: int
    result_dtype: QONNXDataType
    weight_delivery: WeightDelivery
    module: Composed
    initializer: tuple[int, ...]
    point: Any


def _frozen(values: object) -> object:
    """Nested sequences as nested tuples."""
    if isinstance(values, Sequence):
        return tuple(_frozen(item) for item in values)
    return values


def _realizes(point: Any, realization: str) -> QueryResult[bool]:
    """Accepted when the realization commits on ``point``, its own rule holds, and some
    core can compute it."""
    try:
        point = commit(point, {"matmul.realization": realization})
    except ValueError as error:
        return reject("matmul-realization", str(error))
    rule: QueryResult[bool] = point.matmul.inspect(MatMulKernel.realization_supported).result
    if not isinstance(rule, Available):
        return rule
    cores = point.matmul.query(MatMulKernel.compute)  # refused when no core is viable
    if isinstance(cores, Rejected):
        return reject("matmul-realization", "no core computes it")
    return Available(True)


def matmul_assembly(
    *,
    m: int,
    n: int,
    k: int,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    pe: int,
    simd: int,
    platform: Platform,
    form: Form = Form.DENSE,
    compute_pumping: bool = False,
    reducer: str = "tree",
    core: str | None = None,
    realization: str | None = None,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[object] | None = None,
    ram_style: str = "auto",
    pumped_memory: bool = False,
    weight_sets: int = 1,
    weight_fifo_depth: int | None = None,
) -> MatMulAssembly:
    """Bind operation facts, commit the caller's choices and the flow's, then assemble.

    ``m`` rows, ``n`` outputs and the reduction ``k``; for a depthwise ``form``,
    ``k`` is the window and ``n`` the channels. ``weights`` is stored (K, N),
    and is required by, and only accepted with, a memory: the weight channel's
    source, forced when its one candidate is viable. The ``auto``
    ``ram_style`` default leaves memory inference to synthesis.
    ``weight_fifo_depth`` places a FIFO on the weight channel; ``None`` connects
    it directly. The ``platform``'s clock period is the clock the module must
    meet; it sets dotp's DSP58 chain segmentation. ``core`` names the
    compute core (``packed`` or ``int8_dsp58``); left out, the one core
    compatible with the configuration is forced, and several compatible cores
    must be chosen from. PE, SIMD and pumping are the core's, and the
    packed core's ``reducer``.
    Every Decision with one viable case (the source, the adapters, the core
    on DSP48E2) is forced, not committed.
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
        platform=platform,
        weight_sets=weight_sets,
    )
    if weights is not None:
        facts["weights"] = _frozen(weights)
    buffered = weight_fifo_depth is not None
    choices: dict[str, object] = {"w.transport": "fifo" if buffered else "direct"}
    if weight_delivery is WeightDelivery.MEMSTREAM:
        choices["w.source.memstream.ram_style"] = ram_style
        choices["w.source.memstream.pumped_memory"] = pumped_memory
    if buffered:
        choices["w.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["w.transport.fifo.buffer.ram_style"] = "auto"
    point = commit(matmul_point(realization=realization, **facts), choices)
    if form is Form.DEPTHWISE and realization is None:
        # The realization sets the datapath's reduction, and so the weight channel's
        # tensor: each is tried on the one point.
        viable = [
            case for case in ("native", "dense") if isinstance(_realizes(point, case), Available)
        ]
        if len(viable) != 1:
            named = ", ".join(viable) or "none"
            raise ValueError(f"realizations compatible with this configuration: {named}")
        point = commit(point, {"matmul.realization": viable[0]})
    if core is None:
        forced = {item.key: item.value for item in inspection.forced(point)}
        if "matmul.compute" not in forced:
            refusals = {
                case: inspection.admission(commit(point, {"matmul.compute": case}).matmul.compute)
                for case in ("packed", "int8_dsp58")
            }
            cores = [
                case for case, refusal in refusals.items() if not isinstance(refusal, Rejected)
            ]
            if cores:
                raise ValueError(f"compute cores {', '.join(cores)} are all compatible; choose one")
            found = describe(result for result in refusals.values() if result is not None)
            raise ValueError(f"no compute core is compatible: {found}")
        core = str(forced["matmul.compute"])
    else:
        point = commit(point, {"matmul.compute": core})
    point = commit(
        point,
        {
            f"matmul.compute.{core}.pe": pe,
            f"matmul.compute.{core}.simd": simd,
            f"matmul.compute.{core}.compute_pumping": compute_pumping,
            **({"matmul.compute.packed.reducer": reducer} if core == "packed" else {}),
        },
    )
    # Each channel's adapter is forced; an input_gen's memory is inferred; the other
    # channels connect directly.
    point = with_direct_transports(with_adapter_memories(point))
    built = point.query(Kernel.module)
    if not isinstance(built, Available):
        raise ValueError(f"MatMul assembly is not accepted: {describe([built])}")
    matmul = point.matmul
    return MatMulAssembly(
        point.x.ends.source.sequence.form.beats,
        matmul.compute.w.presented.form.beats,
        matmul.compute.y.presented.form.beats,
        matmul.result_type,
        weight_delivery,
        built.value,
        point.w.source.image if point.w.valued else (),
        point,
    )
