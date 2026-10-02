# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test harness: configure a kernel node from its formals, then commit choices by key.

Facts are the root node's typed formals; a missing required one is refused at
the node call. Choices use the stable decision keys ``inspection`` reports.

A kernel with children sits on streams its parent supplies, so a test places
it in a ``Root``, the top of what is emitted, which declares them (as
``placed_dotp`` does for a dot-product core). ``placed_matmul`` places a
MatMul (``matmul``) on the root's ``x`` (``in0_V``), ``w`` (``in1_V``,
buffered), ``y`` (``out0_V``) and, with several weight sets, ``set``
(``in2_V``); the edge choices are the root's (``x.adapter``, ``w.transport``),
the MatMul's below it (``matmul.memory``, ``matmul.compute.packed.pe``).
``matmul_assembly`` configures one from concrete facts and choices."""

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from functools import cache
from typing import Any, TypeVar

from finn import resources

from finn.core.space import Constraint, Param, Space, composite, derived, design_space, reject
from finn.core.space.settling import compatible_cases
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.base import Kernel
from finn.kernels.streams import ADAPTER_RAM_STYLES, BufferedStream, Stream
from finn.core.space.results import Available, QueryResult
from finn.kernels.configure import admission, commit, describe, settle, undecided
from finn.kernels.control import ControlBus
from finn.kernels.matmul import MatMulKernel
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
)
from finn.kernels.target import DspBlock

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def point_for(kernel: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    return commit(design_space(kernel(**facts)), choices)


def value(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> QueryResult[bool]:
    return point.inspect(condition).result


def placed_dotp(
    family: Callable[..., S],
    *,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    result_dtype: QONNXDataType,
    pe: object = None,
    simd: object = None,
    compute_pumping: object = False,
    rows: int = 1,
    outputs: int | None = None,
    reduction: int | None = None,
    **facts: object,
) -> S:
    """A dot-product core between three boundary streams, its folding factors committed.

    The core takes its extents from the streams: ``outputs`` (N) defaults to PE
    and ``reduction`` (K) to SIMD, one fold each. A folding factor left ``None``
    stays open.
    """
    form = facts.get("form", Form.DENSE)
    n = outputs if outputs is not None else (pe if isinstance(pe, int) and pe > 0 else 1)
    k = reduction if reduction is not None else (simd if isinstance(simd, int) and simd > 0 else 1)
    x_shape = (rows, k, n) if form is Form.DEPTHWISE else (rows, k)

    class Placed(Space):
        x = Stream(tensor=Tensor(x_shape, ScalarEncoding(activation_dtype)), port="in0_V")
        w = Stream(tensor=Tensor((k, n), ScalarEncoding(weights_dtype)), port="in1_V")
        y = Stream(tensor=Tensor((rows, n), ScalarEncoding(result_dtype)), port="out0_V")
        compute = family(x_stream=x, w_stream=w, y_stream=y, result_dtype=result_dtype, **facts)

    choices = {
        key: value
        for key, value in (
            ("compute.pe", pe),
            ("compute.simd", simd),
            ("compute.compute_pumping", compute_pumping),
        )
        if value is not None
    }
    point = design_space(Placed())
    placed = commit(point, choices)
    return placed.compute


class Root(Kernel):
    """The top of what a test emits: the streams it declares and the kernels on them."""

    id = "test.root"
    version = "1"


def rooted(name: str, members: Mapping[str, object]) -> Root:
    """A root named ``name`` (its module's stem ``finn_<name>``) of ``members``."""
    family: Any = composite(name, dict(members), base=Root)
    root: Root = family()
    return root


MATMUL_STREAMS = (
    ("x", MatMulKernel.activation_tensor),
    ("w", MatMulKernel.weight_tensor),
    ("y", MatMulKernel.result_tensor),
    ("set", MatMulKernel.set_tensor),
)


def matmul_tensors(
    facts: Mapping[str, object], realization: str | None = None
) -> dict[str, Tensor]:
    """The tensors of the streams a MatMul sits on, read from the MatMul itself.

    A depthwise MatMul's weights follow its ``realization``; several weight sets
    add the set index.
    """
    point: Any = design_space(MatMulKernel(**facts))  # type: ignore[arg-type]
    if facts.get("form", Form.DENSE) is Form.DEPTHWISE:
        if realization is None:
            raise ValueError("a depthwise MatMul's weight tensor follows its realization")
        point = commit(point, {"realization": realization})
    sets = facts.get("weight_sets", 1)
    found: dict[str, Tensor] = {}
    for stream, view in MATMUL_STREAMS:
        if stream == "set" and not (isinstance(sets, int) and sets > 1):
            continue
        result = point.query(view)
        if not isinstance(result, Available):
            raise ValueError(f"MatMul is not accepted: {describe([result])}")
        found[stream] = result.value
    return found


@cache
def matmul_root(family: type[MatMulKernel]) -> type[Root]:
    """A root placing a MatMul of ``family`` (``matmul``) on the streams it declares, its
    facts and the streams' tensors its own formals; see the module docstring."""

    class MatMul(Root):
        x_tensor: Tensor = Param()
        w_tensor: Tensor = Param()
        y_tensor: Tensor = Param()
        set_tensor: Tensor = Param(required=False)
        m: int = Param()
        n: int = Param()
        k: int = Param()
        form: Form = Param(default=Form.DENSE)
        activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        target_dsp: DspBlock = Param()
        target_period_ns: float = Param()
        weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
        writable_weights: bool = Param(default=False)
        weight_sets: int = Param(default=1)

        @derived
        def several(self) -> bool:
            return self.weight_sets > 1

        x = Stream(tensor=x_tensor, port="in0_V")
        w = BufferedStream(tensor=w_tensor, port="in1_V")
        y = Stream(tensor=y_tensor, port="out0_V")
        set = Stream(tensor=set_tensor, port="in2_V", when=several)
        matmul = family(
            m=m,
            n=n,
            k=k,
            form=form,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
            target_dsp=target_dsp,
            target_period_ns=target_period_ns,
            weights=weights,
            writable_weights=writable_weights,
            weight_sets=weight_sets,
            x_stream=x,
            w_stream=w,
            y_stream=y,
            set_stream=set,
        )

    return MatMul


def placed_matmul(*, realization: str | None = None, **facts: object) -> Root:
    """A MatMul (``matmul``) in a root declaring its streams; see the module docstring."""
    tensors = {
        f"{name}_tensor": tensor for name, tensor in matmul_tensors(facts, realization).items()
    }
    family: Any = matmul_root(MatMulKernel)
    root: Root = family(**facts, **tensors)
    return root


def matmul_point(*, realization: str | None = None, **facts: object) -> Any:
    """``placed_matmul`` as a design space, its ``realization`` committed when given."""
    point = design_space(placed_matmul(realization=realization, **facts))
    return commit(point, {"matmul.realization": realization}) if realization else point


def controlled(family: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    """A kernel (``kernel``) presenting its configuration bus through a ``ControlBus``, its
    choices committed by their keys below it; as placed alone it holds the bus."""

    class Controlled(Space):
        config = ControlBus(port="s_axilite")
        kernel = family(**facts, control=config)

    point = commit(design_space(Controlled()), {f"kernel.{key}": v for key, v in choices.items()})
    kernel: S = point.kernel
    return kernel


def labels(module: Composed) -> list[str]:
    """A composed module's instance labels, in netlist order."""
    return [label for label, _ in module.fragment.instances]


def placed(module: Composed, label: str) -> Leaf:
    """The leaf a composed module places at ``label``."""
    return dict(module.fragment.instances)[label]


def pin_names(module: Composed | Leaf) -> set[str]:
    return {port.name for port in module.pins.ports}


def settled(point: S, ram_style: str = "auto") -> S:
    """Settle every Decision over kernels; each adapter input_gen's memory takes ``ram_style``."""
    point = settle(point).point
    styles = undecided(point, ADAPTER_RAM_STYLES)
    return commit(point, dict.fromkeys(styles, ram_style)) if styles else point


def finnlib_root() -> Path:
    """FinnLib as FINN resolves it: FINN_RESOURCES_FINNLIB, a cached copy, or a fetch."""
    return Path(resources.path("finnlib"))


def vivado_simulator() -> bool:
    """Whether a selected Vivado provides xvlog, xelab and xsim.

    FINN images put tool shims on PATH, so a command being found does not mean
    a Vivado installation is selected.
    """
    tools = ("xvlog", "xelab", "xsim")
    return bool(os.environ.get("XILINX_VIVADO")) and all(shutil.which(tool) for tool in tools)


class WeightDelivery(Enum):
    """Where the weights come from: the ``memory`` Decision's case for each."""

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


def _realizes(
    facts: Mapping[str, object], realization: str, choices: dict[str, object]
) -> QueryResult[bool]:
    """Accepted when the choices commit on a root carrying the realization's weights, its
    own rule holds, and some core can compute it."""
    try:
        point = commit(matmul_point(realization=realization, **facts), choices)
    except ValueError as error:
        return reject("matmul-realization", str(error))
    rule: QueryResult[bool] = point.matmul.inspect(MatMulKernel.realization_supported).result
    if not isinstance(rule, Available):
        return rule
    cores = compatible_cases(point, "matmul.compute", admission)
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
        "matmul.memory": case,
        "w.transport": "fifo" if buffered else "direct",
    }
    if weight_delivery is WeightDelivery.MEMSTREAM:
        choices["matmul.memory.memstream.ram_style"] = ram_style
        choices["matmul.memory.memstream.pumped_memory"] = pumped_memory
    if buffered:
        choices["w.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["w.transport.fifo.buffer.ram_style"] = "auto"
    if form is Form.DEPTHWISE and realization is None:
        # The realization sets the datapath's reduction, and so the weight stream's
        # tensor: each is tried on a root of its own.
        viable = [
            case
            for case in ("native", "dense")
            if isinstance(_realizes(facts, case, choices), Available)
        ]
        if len(viable) != 1:
            named = ", ".join(viable) or "none"
            raise ValueError(f"realizations compatible with this configuration: {named}")
        realization = viable[0]
    point = commit(matmul_point(realization=realization, **facts), choices)
    if core is None:
        settlement = settle(point)
        if "matmul.compute" not in settlement.committed:
            cores = settlement.open.get("matmul.compute", ())
            if cores:
                raise ValueError(f"compute cores {', '.join(cores)} are all compatible; choose one")
            refusals = (
                admission(commit(point, {"matmul.compute": case}).matmul.compute)
                for case in ("packed", "int8_dsp58")
            )
            found = describe(result for result in refusals if result is not None)
            raise ValueError(f"no compute core is compatible: {found}")
        point, core = settlement.point, settlement.committed["matmul.compute"]
    else:
        point = commit(point, {"matmul.compute": core})
    point = commit(
        point,
        {
            f"matmul.compute.{core}.pe": pe,
            f"matmul.compute.{core}.simd": simd,
            f"matmul.compute.{core}.compute_pumping": compute_pumping,
        },
    )
    # Each stream's one compatible adapter; an input_gen's memory is inferred.
    point = settled(point)
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
        matmul.memory.image if matmul.memory is not None else (),
        point,
    )
