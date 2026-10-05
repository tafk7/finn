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
(``in2_V``). The root declares the streams and binds each one's tensor to
MatMul's view of it (``activation_tensor``, ``weight_tensor``,
``result_tensor``, ``set_tensor``), which reads only MatMul's facts and its
``realization``: one root serves every realization, a depthwise MatMul's left
open until committed. Known weights are the weight stream's ``contents``
(MatMul's ``weight_values``), with ``set`` its ``index``, so the stream's
``source`` stores them. The edge choices are the root's (``x.adapter``,
``w.transport``, ``w.source.memstream.ram_style``), the MatMul's below it
(``matmul.compute.packed.pe``). ``matmul_assembly`` configures one from
concrete facts and choices."""

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from functools import cache
from typing import Any, TypeVar, cast

from finn import resources

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
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.base import Kernel
from finn.kernels.streams import ADAPTER_RAM_STYLES, BufferedStream, Stream
from finn.core.space.results import Available, QueryResult
from finn.kernels.configure import admission, commit, describe, undecided
from finn.kernels.control import ControlBus
from finn.kernels.matmul import MatMulKernel
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
)
from finn.kernels.target import DspBlock, Platform

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def full_platform(dsp: DspBlock, *, period_ns: float = 5.0) -> Platform:
    """The platform a bare-kernel test means when it is not about the platform: ``dsp``
    its DSP block, a ``period_ns`` clock (5 ns: 200 MHz), and every capability (UltraRAM
    that takes initial contents, a doubled clock, a control port; no memory port and
    no AI Engine, which no kernel reads)."""
    return Platform(
        period_ns=period_ns,
        dsp=dsp,
        uram=True,
        uram_init=True,
        clk2x=True,
        control_ports=1,
        memory_ports=0,
        aie=False,
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
    family: Callable[..., S],
    *,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    result_dtype: QONNXDataType,
    weights_range: tuple[int, int] | None = None,
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
    stays open. ``weights_range`` is the range of values the weight stream
    carries; the datatype's own by default.
    """
    form = facts.get("form", Form.DENSE)
    n = outputs if outputs is not None else (pe if isinstance(pe, int) and pe > 0 else 1)
    k = reduction if reduction is not None else (simd if isinstance(simd, int) and simd > 0 else 1)
    x_shape = (rows, k, n) if form is Form.DEPTHWISE else (rows, k)
    # The streams are on the core's platform; a test that omits it gets FULL_DSP48E2's.
    platform = cast(Platform, facts.get("platform", FULL_DSP48E2))

    class Placed(Space):
        x = Stream(
            tensor=Tensor(x_shape, ScalarEncoding(activation_dtype)),
            port="in0_V",
            platform=platform,
        )
        w = Stream(
            tensor=Tensor((k, n), ScalarEncoding(weights_dtype, weights_range)),
            port="in1_V",
            platform=platform,
        )
        y = Stream(
            tensor=Tensor((rows, n), ScalarEncoding(result_dtype)),
            port="out0_V",
            platform=platform,
        )
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
    version = 1


def rooted(name: str, members: Mapping[str, object]) -> Root:
    """A root named ``name`` (its module's stem ``finn_<name>``) of ``members``."""
    family: Any = composite(name, dict(members), base=Root)
    root: Root = family()
    return root


@cache
def matmul_root(family: type[MatMulKernel]) -> type[Root]:
    """A root placing a MatMul of ``family`` (``matmul``) on the streams it declares, its
    facts its own formals and the streams' tensors MatMul's views; see the module
    docstring."""

    class MatMul(Root):
        m: int = Param()
        n: int = Param()
        k: int = Param()
        form: Form = Param(default=Form.DENSE)
        activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        platform: Platform = Param()
        weights: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
        weight_sets: int = Param(default=1)

        @derived
        def several(self) -> bool:
            return self.weight_sets > 1

        # Each stream's tensor is MatMul's view of it: its facts, never its ports.
        @derived
        def x_tensor(self) -> Tensor:
            return self.matmul.activation_tensor

        @derived
        def w_tensor(self) -> Tensor:
            return self.matmul.weight_tensor

        @derived
        def y_tensor(self) -> Tensor:
            return self.matmul.result_tensor

        @derived
        def set_tensor(self) -> Tensor:
            return self.matmul.set_tensor

        x = Stream(tensor=x_tensor, port="in0_V", platform=platform)
        set = Stream(tensor=set_tensor, port="in2_V", when=several, platform=platform)
        w = BufferedStream(
            tensor=w_tensor, sets=weight_sets, index=set, port="in1_V", platform=platform
        )
        y = Stream(tensor=y_tensor, port="out0_V", platform=platform)
        matmul = family(
            m=m,
            n=n,
            k=k,
            form=form,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
            platform=platform,
            weights=weights,
            weight_sets=weight_sets,
            x_stream=x,
            w_stream=w,
            y_stream=y,
        )
        w.contents = matmul.weight_values

    return MatMul


def placed_matmul(**facts: object) -> Root:
    """A MatMul (``matmul``) in a root declaring its streams; see the module docstring."""
    family: Any = matmul_root(MatMulKernel)
    root: Root = family(**facts)
    return root


def matmul_point(*, realization: str | None = None, **facts: object) -> Any:
    """``placed_matmul`` as a design space, its ``realization`` committed when given."""
    point = design_space(placed_matmul(**facts))
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


def with_adapter_memories(point: S, ram_style: str = "auto") -> S:
    """Each open adapter input_gen's memory takes ``ram_style``: the flow's choice. Each
    stream's adapter, its one viable chain, is forced."""
    styles = undecided(point, ADAPTER_RAM_STYLES)
    return commit(point, dict.fromkeys(styles, ram_style)) if styles else point


def finnlib_root() -> Path:
    """FinnLib as FINN resolves it: FINN_RESOURCES_FINNLIB, a cached copy, or a fetch."""
    return Path(resources.path("finnlib"))


def print_identity() -> None:
    """Print the FINN and FinnLib revisions a run compiles, first in its log.

    ``finn <sha>[ dirty]`` and ``finnlib <revision> <path>``: a log that does not
    say what it compiled is not evidence about a commit. A working clone of
    FinnLib is named by its commit; the cached pin, which is not a repository,
    by the digest it was verified against.
    """

    def revision(directory: Path) -> str:
        def git(*args: str) -> str:
            done = subprocess.run(
                ["git", "-C", str(directory), *args], capture_output=True, text=True
            )
            return done.stdout.strip() if done.returncode == 0 else ""

        sha = git("rev-parse", "--short", "HEAD")
        if not sha:
            marker = directory / ".finn-resource"
            return f"pinned {marker.read_text().strip()[:19]}" if marker.exists() else "unknown"
        return f"{sha} dirty" if git("status", "--porcelain", "--untracked-files=no") else sha

    finnlib = finnlib_root()
    print(f"finn {revision(Path(__file__).resolve().parent)}", flush=True)
    print(f"finnlib {revision(finnlib)} {finnlib}", flush=True)


def vivado_simulator() -> bool:
    """Whether a selected Vivado provides xvlog, xelab and xsim.

    FINN images put tool shims on PATH, so a command being found does not mean
    a Vivado installation is selected.
    """
    tools = ("xvlog", "xelab", "xsim")
    return bool(os.environ.get("XILINX_VIVADO")) and all(shutil.which(tool) for tool in tools)


class WeightDelivery(Enum):
    """Where the weights come from: the weight stream's ``source`` case, or the
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
    and is required by, and only accepted with, a memory: the weight stream's
    source, forced when its one candidate is viable. The ``auto``
    ``ram_style`` default leaves memory inference to synthesis.
    ``weight_fifo_depth`` places a FIFO on the weight stream; ``None`` connects
    it directly. The ``platform``'s clock period is the clock the module must
    meet; it sets dotp's DSP58 chain segmentation. ``core`` names the
    compute core (``packed`` or ``int8_dsp58``); left out, the one core
    compatible with the configuration is forced, and several compatible cores
    must be chosen from. PE, SIMD and pumping are the core's.
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
        # The realization sets the datapath's reduction, and so the weight stream's
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
                case: admission(commit(point, {"matmul.compute": case}).matmul.compute)
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
        },
    )
    # Each stream's adapter is forced; an input_gen's memory is inferred.
    point = with_adapter_memories(point)
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
