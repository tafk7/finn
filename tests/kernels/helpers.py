# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test harness: configure a kernel node from its formals, then commit choices by key.

Facts are the root node's typed formals; a missing required one is refused at
the node call. Choices use the stable decision keys ``inspection`` reports.
``matmul_assembly`` configures a MatMul from concrete facts and choices."""

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from finn import resources

from finn.core.space import Constraint, Space, design_space, reject
from finn.core.space.settling import compatible_cases
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.streams import ADAPTER_RAM_STYLES, Stream
from finn.core.space.results import Available, QueryResult
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.configure import admission, commit, describe, settle, undecided
from finn.kernels.matmul import MatMulKernel
from finn.kernels.physical.structure import PhysicalStructure
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
    structure: PhysicalStructure
    requirements: ModuleBuildRequirements
    initializer: tuple[int, ...]


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
        # the other choices before the core's folding factors.
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
        settlement = settle(point)
        if "compute" not in settlement.committed:
            cores = settlement.open.get("compute", ())
            if cores:
                raise ValueError(f"compute cores {', '.join(cores)} are all compatible; choose one")
            refusals = (
                admission(commit(point, {"compute": case}).compute)
                for case in ("packed", "int8_dsp58")
            )
            found = describe(result for result in refusals if result is not None)
            raise ValueError(f"no compute core is compatible: {found}")
        point, core = settlement.point, settlement.committed["compute"]
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
    point = settled(point)
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
