# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test construction: configure a kernel node from its formals, then commit choices by key.

Facts are the root node's typed formals; a missing required one is refused at
the node call. Choices use the stable decision keys ``inspection`` reports.

A kernel reads tensors and presents its ends on channels, so a test places it in a
network (``finn.kernels.network``): its kernels and channels by name, each its own
space, each channel taking the ends its kernels present (``Interface``) and, for a
parameter channel, what its kernel states for it (``Fact``). A test names a choice as
``member.key`` (``matmul.compute.packed.pe``, ``w.transport``): ``keyed`` makes it the
network's pair, ``choose`` commits a batch. ``placed_dotp`` places a dot-product core
between three boundary channels. ``matmul_network`` places a MatMul (``matmul``) on
``x`` (``in0_V``), ``w`` (``in1_V``, buffered), ``y`` (``out0_V``) and, with several
weight sets, ``set`` (``in2_V``). Its tensors are MatMul's own views of them
(``activation_tensor``, ``weight_tensor``, and its results, ``y_tensor``), read from its facts
before it is placed: they read only its facts and its ``realization``, and the
result's range is the same in either realization, so one network serves every
realization, a depthwise MatMul's left open until committed. Known weights are a fact
given to MatMul (``w_contents``); the weight channel carries what MatMul states for it
(``w_carried``, ``w_stored``: block-diagonal for a dense realization, over their
range), with ``set``'s tensor its ``index_tensor`` (one index per row) and ``set``
taking its ``set_end``, so the channel's ``source`` stores them. The edge choices are
the channels' (``x.adapter``, ``w.transport``, ``w.source.memstream.ram_style``), the
MatMul's its own (``matmul.compute.packed.pe``). ``matmul_assembly`` configures one
from concrete facts and choices. A network's module is named ``finn_<stem>``
(``module_of``).

A kernel presents its configuration bus itself, at the port it is given (``control``);
a network presents it on the module.

What a simulation runs on (FinnLib, Vivado, the run's identity) is
``finn.harness.toolchain``'s: this module only constructs."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any, TypeVar, cast

from finn.core.space import (
    Constraint,
    Decision,
    Rejected,
    Space,
    design_space,
    inspection,
    reject,
)
from finn.core.space.results import Available, QueryResult
from finn.dataflow.datatypes import (
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Loop, Traversal, vector_major
from finn.kernels.artifacts.module import Composed, Leaf, ProducerIdentity
from finn.kernels.base import CYCLES
from finn.kernels.configure import commit, describe, undecided
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.explore import Choice
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.matmul import MatMulKernel, datatype_range
from finn.kernels.network import Fact, Interface, Key, Network, Wire, dotted, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.domains import range_dtype, set_index_dtype
from finn.kernels.values.semantics import (
    IntegerTensor,
    IntegerTensorValue,
    IntegerVector,
    integer_shape,
)
from finn.platform.capabilities import DspBlock, Fabric, Platform

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


def exact_result_dtype(
    vector_length: int, activation_dtype: QONNXDataType, weights_dtype: QONNXDataType
) -> QONNXDataType:
    """The smallest encoding of every full-range integer dot product of ``vector_length``
    values of each datatype (``range_dtype`` of ``matmul.datatype_range``): the result
    type a test states for a MatMul whose weights are not known."""
    return range_dtype(*datatype_range(vector_length, activation_dtype, weights_dtype))


def point_for(kernel: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    return commit(design_space(kernel(**facts)), choices)


def value(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> QueryResult[bool]:
    return point.inspect(condition).result


PRODUCER = ProducerIdentity("test.root", "1")
"""Who derives a test network's module: the tests' harness, each module named by its stem."""


def keyed(choices: Mapping[str, object]) -> dict[Key, object]:
    """Choices named as a test names them, ``member.key``, by the network's key: the
    member, and the key in its space."""
    found: dict[Key, object] = {}
    for name, value in choices.items():
        owner, _, key = name.partition(".")
        found[(owner, key)] = value
    return found


N = TypeVar("N", bound=Network)


def choose(network: N, choices: Mapping[str, object]) -> N:
    """``network`` with ``choices`` (``member.key``) committed, atomically; a refusal, or a
    stored choice the change would drop, is a ``NetworkError`` (a ``ValueError``)."""
    return network.with_choices(keyed(choices)) if choices else network


def module_of(network: Network, stem: str) -> QueryResult[Composed]:
    """``network``'s module, named ``finn_<stem>``."""
    return network.module(PRODUCER, f"finn_{stem}")


def built(network: Network, stem: str) -> Composed:
    """``network``'s module, named ``finn_<stem>``, which must be accepted."""
    module = module_of(network, stem)
    assert isinstance(module, Available), describe([module])
    return module.value


def cycles_of(network: Network) -> QueryResult[int]:
    """The most cycles a frame any member of ``network`` takes, once each states its own."""
    found = [network.query(name, CYCLES) for name in network.members]
    waiting = [answer for answer in found if not isinstance(answer, Available)]
    if waiting:
        return waiting[0]
    return Available(max(cast(Available[int], answer).value for answer in found))


def is_decision(space_type: object, name: str) -> bool:
    """Whether ``space_type``'s member ``name`` is a Decision (the packed core's
    ``reducer``; the INT8 core states its own)."""
    return isinstance(getattr(space_type, name, None), Decision)


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

    The core takes its extents from its tensors: ``outputs`` (N) defaults to PE
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

    x_tensor = Tensor(x_shape, ScalarEncoding(activation_dtype))
    w_tensor = Tensor((k, n), ScalarEncoding(weights_dtype, weights_range))
    # The results' channel's tensor: the graph's statement of what the core produces.
    y_tensor = Tensor((rows, n), ScalarEncoding(result_dtype, result_range))
    network = Network(
        {
            "x": wire(
                tensor=x_tensor,
                port="in0_V",
                consumer=Interface("compute", "x_end"),
                platform=platform,
            ),
            "w": wire(
                tensor=w_tensor,
                port="in1_V",
                consumer=Interface("compute", "w_end"),
                platform=platform,
            ),
            "y": wire(
                tensor=y_tensor,
                port="out0_V",
                producer=Interface("compute", "y_end"),
                platform=platform,
            ),
            "compute": space_type(
                x_tensor=x_tensor,
                w_tensor=w_tensor,
                result_range=result_range or ordinary_integer_bounds(result_dtype),
                **facts,
            ),
        }
    )
    choices = {
        key: value
        for key, value in (
            ("compute.pe", pe),
            ("compute.simd", simd),
            ("compute.compute_pumping", compute_pumping),
            ("compute.reducer", reducer if is_decision(space_type, "reducer") else None),
        )
        if value is not None
    }
    placed: S = choose(network, choices).member("compute")
    return placed


def _views(
    space_type: type[MatMulKernel], facts: Mapping[str, Any], realization: str | None
) -> Any:
    """A MatMul of ``space_type`` on ``facts`` alone, its ``realization`` committed where
    it is depthwise (as named; else ``dense`` with known weights, ``native`` without, or
    whichever commits): what its views of its activations and results read. The result's
    range is the same in either realization."""
    probe = design_space(space_type(**facts))
    if facts.get("form") is not Form.DEPTHWISE:
        return probe
    preferred = realization or ("dense" if "w_contents" in facts else "native")
    for case in (preferred, *(other for other in ("native", "dense") if other != preferred)):
        try:
            return commit(probe, {"realization": case})
        except ValueError:
            if realization is not None:
                raise
    return probe


def _view(probe: Any, name: str) -> Tensor:
    """MatMul's view ``name`` of a tensor, or a ``ValueError`` naming why it states none."""
    found = probe.query(getattr(type(probe), name))
    if not isinstance(found, Available):
        raise ValueError(f"MatMul's {name} is refused: {describe([found])}")
    tensor: Tensor = found.value
    return tensor


def matmul_network(
    space_type: type[MatMulKernel] | None = None,
    *,
    m: int,
    n: int,
    k: int,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    platform: Platform,
    form: Form = Form.DENSE,
    weights: IntegerTensor | None = None,
    weight_sets: int = 1,
    realization: str | None = None,
) -> Network:
    """A MatMul of ``space_type`` (``matmul``; a ``MatMulKernel`` by default) on the
    channels of a network: its activations ``(m, k)`` (``(m, k, n)`` depthwise) of
    ``activation_dtype``, its weights known (``weights``, declared ``weights_dtype``) or
    streamed (``(k, n)`` of ``weights_dtype``), its results its own statement; see the
    module docstring. ``realization`` is committed where given."""
    space_type = space_type or MatMulKernel
    shape = (m, k, n) if form is Form.DEPTHWISE else (m, k)
    facts: dict[str, Any] = dict(
        form=form, platform=platform, x_tensor=Tensor(shape, ScalarEncoding(activation_dtype))
    )
    if weights is not None:
        facts |= {"w_contents": weights, "weights_dtype": weights_dtype}
    else:
        facts["w_tensor"] = Tensor((k, n), ScalarEncoding(weights_dtype))
    views = _views(space_type, facts, realization)
    tensors: dict[str, Any] = {
        "x_tensor": facts.pop("x_tensor"),
        "y_tensor": _view(views, "y_tensor"),
    }
    if weights is None:
        # Unknown weights are MatMul's view of them at the realization committed; with
        # none, as stored, (K, N) in their datatype.
        stated = views.query(type(views).weight_tensor)
        given = facts.pop("w_tensor")
        tensors["w_tensor"] = stated.value if isinstance(stated, Available) else given
    several = weight_sets > 1
    set_tensor = Tensor((m,), ScalarEncoding(set_index_dtype(weight_sets)))
    members: dict[str, Space | Wire] = {
        "x": wire(
            tensor=tensors["x_tensor"],
            port="in0_V",
            consumer=Interface("matmul", "x_end"),
            platform=platform,
        )
    }
    if several:
        members["set"] = wire(
            tensor=set_tensor,
            port="in2_V",
            consumer=Interface("w", "set_end"),
            platform=platform,
        )
    weight: dict[str, object] = dict(
        tensor=Fact("matmul", "w_carried"),
        contents=Fact("matmul", "w_stored"),
        sets=weight_sets,
        port="in1_V",
        consumer=Interface("matmul", "w_end"),
        platform=platform,
    )
    if several:
        weight["index_tensor"] = set_tensor
    members["w"] = wire(**weight)
    members["y"] = wire(
        tensor=tensors["y_tensor"],
        port="out0_V",
        producer=Interface("matmul", "y_end"),
        platform=platform,
    )
    members["matmul"] = space_type(
        **facts, **{name: tensor for name, tensor in tensors.items() if name != "y_tensor"}
    )
    network = Network(members)
    return choose(network, {"matmul.realization": realization}) if realization else network


def matmul_point(*, realization: str | None = None, **facts: Any) -> Network:
    """``matmul_network`` of a ``MatMulKernel``, its ``realization`` committed when given."""
    return matmul_network(realization=realization, **facts)


def controlled(space_type: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    """A kernel given a port for its configuration bus (``s_axilite``), its ``choices``
    committed; as placed alone it holds the bus."""
    return commit(design_space(space_type(**facts, control="s_axilite")), choices)


THRESHOLDS = IntegerTensorValue.of((((-2, 0, 3), (-1, 1, 4)),))
"""One threshold table, one set of two channels of three thresholds."""


def threshold_base(
    *,
    table: IntegerTensor = THRESHOLDS,
    bias: int = -1,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT5",
    channels: int | None = None,
) -> ThresholdingAxiKernel:
    """A thresholding's design space on one row of ``channels`` inputs of ``input_dtype``
    (the table's rows by default), its choices open."""
    shape = integer_shape(table)
    count = channels if channels is not None else (shape[1] if shape and len(shape) == 3 else 1)
    return design_space(
        ThresholdingAxiKernel(
            input_tensor=Tensor(
                (1, count), ScalarEncoding(resolve_qonnx_datatype_name(input_dtype))
            ),
            threshold_dtype=resolve_qonnx_datatype_name(threshold_dtype),
            thresholds=table,
            bias=bias,
            platform=FULL_DSP48E2,
        )
    )


def stage_tensor(word_bits: int, words: int = 6, lanes: int = 1) -> Tensor:
    """A channel stage's tensor alone: ``words`` words of ``lanes`` lanes, ``word_bits``
    bits a word (an ``INT`` element of ``word_bits // lanes`` bits)."""
    element = ScalarEncoding(resolve_qonnx_datatype_name(f"INT{word_bits // lanes}"))
    return Tensor((words, lanes), element)


def in_order(tensor: Tensor) -> BeatSequence:
    """A stage tensor's words in order, a word a beat."""
    return BeatSequence(vector_major(tensor.shape, tensor.shape[-1]))


def fifo_node(
    *, word_bits: int = 13, depth: int = 8, platform: Platform | None = None, words: int = 6
) -> FifoKernel:
    """A FIFO alone, of ``depth`` words of ``word_bits`` bits: its tensor ``words`` words
    in order (``stage_tensor``), on ``platform`` (``FULL_DSP48E2``)."""
    tensor = stage_tensor(word_bits, words)
    return FifoKernel(
        input_tensor=tensor,
        arriving=in_order(tensor),
        depth=depth,
        platform=FULL_DSP48E2 if platform is None else platform,
    )


def nested(
    frame: int, dims: IntegerVector, strides: IntegerVector, lanes: int = 1
) -> tuple[BeatSequence, BeatSequence]:
    """What an input generator that walks the nest ``dims``/``strides`` over frames of
    ``frame`` words is given: one frame's words in order, and its order of them."""
    shape = (frame, lanes)
    loops = [Loop(extent, stride * lanes) for extent, stride in zip(dims, strides)]
    leaving = Traversal(shape, loops, (Loop(lanes, 1),))
    return BeatSequence(vector_major(shape, lanes)), BeatSequence(leaving)


def generator_node(
    *,
    bits: int = 13,
    frame: int = 6,
    dims: IntegerVector = (3, 6),
    strides: IntegerVector = (0, 1),
    platform: Platform | None = None,
) -> InputGeneratorKernel:
    """An input generator alone, walking ``dims``/``strides`` over frames of ``frame``
    words of ``bits`` bits (``nested``)."""
    arriving, leaving = nested(frame, dims, strides)
    return InputGeneratorKernel(
        input_tensor=stage_tensor(bits, frame),
        arriving=arriving,
        leaving=leaving,
        platform=FULL_DSP48E2 if platform is None else platform,
    )


def generator(
    *,
    bits: int = 13,
    frame: int = 6,
    dims: IntegerVector = (3, 6),
    strides: IntegerVector = (0, 1),
) -> InputGeneratorKernel:
    """An input generator's design space (``generator_node``), its memory Vivado's."""
    node = generator_node(bits=bits, frame=frame, dims=dims, strides=strides)
    return design_space(node).with_choices(ram_style="auto")


def eltwise(
    *,
    operation: str = "ADD",
    pe: int = 2,
    lhs: str = "INT3",
    rhs: str = "INT3",
    scale: float = 1.0,
    target: DspBlock = DspBlock.DSP58,
    shape: tuple[int, ...] = (2, 6),
) -> EltwiseKernel:
    """An elementwise kernel's design space on two ``shape`` operands, on a platform of
    ``target``, its ``pe`` committed."""
    return commit(
        design_space(
            EltwiseKernel(
                operation=operation,
                lhs_tensor=Tensor(shape, ScalarEncoding(resolve_qonnx_datatype_name(lhs))),
                rhs_tensor=Tensor(shape, ScalarEncoding(resolve_qonnx_datatype_name(rhs))),
                b_scale=scale,
                platform=full_platform(target),
            )
        ),
        {"pe": pe},
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

P = TypeVar("P", bound=Space | Network)


def open_choices(network: Network, pattern: str) -> list[str]:
    """The open choices of ``network`` matching ``pattern`` (``fnmatch``), as
    ``member.key``."""
    return _undecided(network, pattern)


def _undecided(subject: Space | Network, pattern: str) -> list[str]:
    """The open choices of a point or a network matching ``pattern``, by key (a network's
    as ``member.key``)."""
    if isinstance(subject, Network):
        return [dotted(key) for key in subject.undecided() if fnmatchcase(dotted(key), pattern)]
    return undecided(subject, pattern)


def _committed(subject: P, choices: Mapping[str, object]) -> P:
    if isinstance(subject, Network):
        return cast(P, choose(subject, choices))
    return cast(P, commit(subject, choices))


def with_adapter_memories(subject: P, ram_style: str | None = None) -> P:
    """Each open adapter input_gen's memory takes ``ram_style``, or where None its
    baseline, the first viable case (an explicit style, by the size the buffer holds),
    as ``Seam.baseline`` takes it, on a network or a point (a shell root's). Each
    channel's adapter, its one viable chain, is forced."""
    styles = _undecided(subject, ADAPTER_RAM_STYLES)
    if not styles:
        return subject
    if ram_style is None:
        if isinstance(subject, Network):
            viable = {dotted(key): item for key, item in subject.viable()}
        else:
            viable = {item.key: item for item in inspection.viable(subject)}
        first = {key: item.cases[0] for key, item in viable.items() if item.cases}
        return _committed(subject, {key: first[key] for key in styles})
    return _committed(subject, dict.fromkeys(styles, ram_style))


TRANSPORTS = "*.transport"
"""The keys (``fnmatch``) of every channel's transport choice."""


def with_direct_transports(subject: P) -> P:
    """Each channel's open transport direct: its baseline, the first case."""
    open_ = _undecided(subject, TRANSPORTS)
    return _committed(subject, dict.fromkeys(open_, "direct")) if open_ else subject


class Lanes:
    """A rank policy for tests (``finn.kernels.explore.Ranked``) that folds by hand: an
    ordered choice of integers at ``lanes`` where that case is viable, otherwise at its
    largest viable case; every other choice, and every choice with no ``lanes``, in its
    domain's order, the kernel's baseline first. A test that needs a choice made on
    purpose states it so; ``Lanes()`` ranks as the baseline takes."""

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
    network: Network


def _frozen(values: object) -> object:
    """Nested sequences as nested tuples."""
    if isinstance(values, Sequence):
        return tuple(_frozen(item) for item in values)
    return values


def _realizes(network: Network, realization: str) -> QueryResult[bool]:
    """Accepted when the realization commits on ``network``, its own rule holds, and some
    core can compute it."""
    try:
        network = choose(network, {"matmul.realization": realization})
    except ValueError as error:
        return reject("matmul-realization", str(error))
    matmul = network.member("matmul")
    rule: QueryResult[bool] = matmul.inspect(MatMulKernel.realization_supported).result
    if not isinstance(rule, Available):
        return rule
    cores = matmul.query(MatMulKernel.compute)  # refused when no core is viable
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
    on DSP48E2) is forced, not committed. The kernel's choices are committed first,
    then its channels', whose ends are its.
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
    channel: dict[str, object] = {"w.transport": "fifo" if buffered else "direct"}
    # The memory's style is ordered by the bits its table holds, which a depthwise
    # weight channel's realization sets: it is committed after the realization.
    if weight_delivery is WeightDelivery.MEMSTREAM:
        channel["w.source.memstream.ram_style"] = ram_style
        channel["w.source.memstream.pumped_memory"] = pumped_memory
    if buffered:
        channel["w.transport.fifo.buffer.depth"] = weight_fifo_depth
        channel["w.transport.fifo.buffer.ram_style"] = "auto"
    network = matmul_point(realization=realization, **facts)
    if form is Form.DEPTHWISE and realization is None:
        # The realization sets the datapath's reduction, and so the weight channel's
        # tensor: each is tried on the one network.
        viable = [
            case for case in ("native", "dense") if isinstance(_realizes(network, case), Available)
        ]
        if len(viable) != 1:
            named = ", ".join(viable) or "none"
            raise ValueError(f"realizations compatible with this configuration: {named}")
        network = choose(network, {"matmul.realization": viable[0]})
    if core is None:
        forced = network.forced(("matmul",))
        if ("matmul", "compute") not in forced:
            refusals = {
                case: inspection.admission(
                    choose(network, {"matmul.compute": case}).member("matmul").compute
                )
                for case in ("packed", "int8_dsp58")
            }
            cores = [
                case for case, refusal in refusals.items() if not isinstance(refusal, Rejected)
            ]
            if cores:
                raise ValueError(f"compute cores {', '.join(cores)} are all compatible; choose one")
            found = describe(result for result in refusals.values() if result is not None)
            raise ValueError(f"no compute core is compatible: {found}")
        core = str(forced[("matmul", "compute")])
    else:
        network = choose(network, {"matmul.compute": core})
    network = choose(
        network,
        {
            f"matmul.compute.{core}.pe": pe,
            f"matmul.compute.{core}.simd": simd,
            f"matmul.compute.{core}.compute_pumping": compute_pumping,
            **({"matmul.compute.packed.reducer": reducer} if core == "packed" else {}),
        },
    )
    network = choose(network, channel)
    # Each channel's adapter is forced; an input_gen's memory is inferred; the other
    # channels connect directly.
    network = with_direct_transports(with_adapter_memories(network))
    module = module_of(network, "matmul")
    if not isinstance(module, Available):
        raise ValueError(f"MatMul assembly is not accepted: {describe([module])}")
    matmul = network.member("matmul")
    w = network.member("w")
    return MatMulAssembly(
        network.member("x").ends.source.sequence.form.beats,
        matmul.compute.w.presented.form.beats,
        matmul.compute.y.presented.form.beats,
        matmul.result_type,
        weight_delivery,
        module.value,
        w.source.image if w.valued else (),
        network,
    )
