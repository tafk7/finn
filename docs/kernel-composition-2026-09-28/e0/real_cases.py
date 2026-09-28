# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Cases 1-4 in the refined form, over the real kernels, beside today's declarations.

Each refined family subclasses today's and replaces only the Decision under
test, under the same name, so every other member (and every key) is the one
today's family declares:

- ``RefinedStream`` / ``RefinedBufferedStream``: the stream's ``adapter``
  (case 3, option A of Q3: ``ram_style`` written on each entry that takes it)
  and ``transport`` (case 4);
- ``RefinedMatMul``: ``compute`` over the two dot-product cores (case 1) and
  the weight memory ``delivery`` (case 2, ``optional``), with its streams
  replaced by the refined streams;
- ``OwnedRamStyleStream``: case 3 under option C of Q3, each buffering
  adapter owning its ``ram_style`` (keys ``adapter.<case>.ram_style``).

``refined_assembly`` is ``matmul_assembly`` for any MatMul family, with
``settle`` in place of ``compatible`` and ``commit_adapters``.
"""

# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from refined import Decision, Settlement, admission_by_family, compatible_cases, settle

from finn.core.space import Available, Space, design_space, inspection
from finn.kernels.adapters import (
    INPUT_GEN_RAM_STYLES,
    InputGenAdapter,
    RegroupAdapter,
    RegroupMarkersAdapter,
    ReorderWidthAdapter,
    ReorderWidthMarkersAdapter,
    StreamAdapter,
    WidthAdapter,
    WidthReorderAdapter,
)
from finn.kernels.configure import commit, describe
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import Contraction, DotpAxiKernel, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.matmul import MatMulAssembly, MatMulKernel, WeightDelivery, _frozen
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.streams import BufferedStream, Stream, StreamFifo, _Direct
from finn.kernels.target import DspBlock

# Today's kernels name their admission differently (F9 unifies it as ``admission``).
ADMISSION = admission_by_family(
    {DotpAxiKernel: DotpAxiKernel.support, StreamAdapter: StreamAdapter.admitted}
)

T = Stream  # the base whose members the refined streams bind


class RefinedStream(Stream):
    """Case 3: seven entries, shared ``tensor`` and ``plan``, guarded; ``ram_style`` per entry."""

    adapter: StreamAdapter = Decision(
        {
            "input_gen": InputGenAdapter(ram_style=T.adapter_ram_style),
            "vpc": WidthAdapter,
            "vpc_input_gen": WidthReorderAdapter(ram_style=T.adapter_ram_style),
            "input_gen_vpc": ReorderWidthAdapter(ram_style=T.adapter_ram_style),
            "input_gen_vpc_input_gen": ReorderWidthMarkersAdapter(ram_style=T.adapter_ram_style),
            "vpc_input_gen_vpc": RegroupAdapter(ram_style=T.adapter_ram_style),
            "vpc_input_gen_vpc_input_gen": RegroupMarkersAdapter(ram_style=T.adapter_ram_style),
        },
        tensor=T.tensor,
        plan=T.plan,
        when=T.adapting,
    )


class RefinedBufferedStream(RefinedStream, BufferedStream):
    """Case 4: the transport, ``direct`` (a bare class) or a FIFO with its own bindings."""

    transport: _Direct | StreamFifo = Decision(
        {"direct": _Direct, "fifo": StreamFifo(tensor=T.tensor, arriving=T.arriving)}
    )


def _owned() -> Any:
    return Decision(values=INPUT_GEN_RAM_STYLES)


class OwnedRamStyleStream(Stream):
    """Case 3, option C: each buffering adapter owns its ``ram_style`` choice.

    Written here as an inline Decision per entry, which is keyed by the
    candidate's formal (``adapter.input_gen.ram_style``): the keys K2 gets by
    declaring ``ram_style`` as a Decision on the adapters that have an
    ``input_gen``. The stream-level ``adapter_ram_style`` is then never bound.
    """

    adapter: StreamAdapter = Decision(
        {
            "input_gen": InputGenAdapter(ram_style=_owned()),
            "vpc": WidthAdapter,
            "vpc_input_gen": WidthReorderAdapter(ram_style=_owned()),
            "input_gen_vpc": ReorderWidthAdapter(ram_style=_owned()),
            "input_gen_vpc_input_gen": ReorderWidthMarkersAdapter(ram_style=_owned()),
            "vpc_input_gen_vpc": RegroupAdapter(ram_style=_owned()),
            "vpc_input_gen_vpc_input_gen": RegroupMarkersAdapter(ram_style=_owned()),
        },
        tensor=T.tensor,
        plan=T.plan,
        when=T.adapting,
    )


M = MatMulKernel  # the base whose facts and derived values the refined Decisions bind


class RefinedMatMul(MatMulKernel):
    """Cases 1 and 2 over the real cores and memories, on the refined streams."""

    activations = RefinedStream(tensor=M.activation_tensor, port="in0_V")
    weight_stream = RefinedBufferedStream(tensor=M.weight_tensor, port="in1_V")
    results = RefinedStream(tensor=M.result_tensor, port="out0_V")
    set_index = RefinedStream(tensor=M.set_tensor, port="in2_V", when=M.multi_set)

    # Case 1: thirteen shared bindings; narrow_weights is the packed core's own.
    compute: PackedDotpKernel | Int8Dsp58DotpKernel = Decision(
        {
            "packed": PackedDotpKernel(narrow_weights=M.narrow_weights),
            "int8_dsp58": Int8Dsp58DotpKernel,
        },
        activation_dtype=M.activation_dtype,
        weights_dtype=M.weights_dtype,
        result_dtype=M.result_type,
        pe=M.pe,
        simd=M.simd,
        target_dsp=M.target_dsp,
        target_period_ns=M.target_period_ns,
        compute_pumping=M.compute_pumping,
        contraction=M.datapath,
        activation_stream=activations,
        weights_stream=weight_stream,
        result_stream=results,
        iteration=M.iteration,
    )
    # Case 2: optional; `values` collides with the scalar form's argument (Q1), so it
    # is written on each entry here. The None case keeps today's key "external".
    delivery: CyclicDelivery | MemStreamKernel | None = Decision(
        {
            "cyclic": CyclicDelivery(values=M.datapath_weights),
            "memstream": MemStreamKernel(
                values=M.datapath_weights,
                writable=M.writable_weights,
                sets=M.weight_sets,
                set_stream=set_index,
                control=M.config,
            ),
        },
        optional=WeightDelivery.EXTERNAL.value,
        dtype=M.weights_dtype,
        form=M.weight_period,
        output_stream=weight_stream,
    )


def _realizes(base: Any, choices: dict[str, object]) -> bool:
    try:
        point = commit(base, choices)
    except ValueError:
        return False
    rule = point.inspect(MatMulKernel.realization_supported).result
    return isinstance(rule, Available) and bool(compatible_cases(point, "compute", ADMISSION))


def refined_assembly(
    family: type[MatMulKernel] = RefinedMatMul,
    *,
    rows: int,
    reduction: int,
    outputs: int,
    activation_dtype: Any,
    weights_dtype: Any,
    pe: int,
    simd: int,
    target_dsp: DspBlock,
    contraction: Contraction = Contraction.DENSE,
    target_period_ns: float = 5.0,
    compute_pumping: bool = False,
    core: str | None = None,
    realization: str | None = None,
    weight_delivery: WeightDelivery = WeightDelivery.EXTERNAL,
    weights: Sequence[object] | None = None,
    rom_style: str = "auto",
    ram_style: str = "auto",
    pumped_memory: bool = False,
    writable_weights: bool = False,
    weight_sets: int = 1,
    weight_fifo_depth: int | None = None,
) -> tuple[MatMulAssembly, Settlement[Any]]:
    """``matmul_assembly`` over ``family``: facts, then ``commit``, then ``settle``."""
    facts: dict[str, Any] = dict(
        rows=rows,
        reduction=reduction,
        outputs=outputs,
        contraction=contraction,
        activation_dtype=activation_dtype,
        weights_dtype=weights_dtype,
        target_dsp=target_dsp,
        target_period_ns=target_period_ns,
        writable_weights=writable_weights,
        weight_sets=weight_sets,
    )
    if weights is not None:
        facts["weights"] = _frozen(weights)
    buffered = weight_fifo_depth is not None
    choices: dict[str, object] = {
        "delivery": weight_delivery.value,
        "weight_stream.transport": "fifo" if buffered else "direct",
        "compute_pumping": compute_pumping,
        "pe": pe,
        "simd": simd,
    }
    if weight_delivery is WeightDelivery.CYCLIC:
        choices["delivery.cyclic.rom_style"] = rom_style
    if weight_delivery is WeightDelivery.MEMSTREAM:
        choices["delivery.memstream.ram_style"] = ram_style
        choices["delivery.memstream.pumped_memory"] = pumped_memory
    if buffered:
        choices["weight_stream.transport.fifo.buffer.depth"] = weight_fifo_depth
        choices["weight_stream.transport.fifo.buffer.ram_style"] = "auto"
    base = design_space(family(**facts))
    if contraction is Contraction.PER_CHANNEL:
        if realization is None:
            viable = [
                case
                for case in ("native", "dense")
                if _realizes(base, {**choices, "realization": case})
            ]
            if len(viable) != 1:
                raise ValueError(f"realizations compatible with this configuration: {viable}")
            realization = viable[0]
        choices["realization"] = realization
    if core is not None:
        choices["compute"] = core
    settled = settle(commit(base, choices), admission=ADMISSION)
    if "compute" in settled.open:
        found = settled.open["compute"]
        raise ValueError(f"compute cores compatible with this configuration: {list(found)}")
    point = settled.point
    # The adapters' memory: the stream-level Decision, applicable when the chain buffers.
    styles = {
        item.key: "auto"
        for item in inspection.decisions(point)
        if item.key.endswith("adapter_ram_style")
        and isinstance(point.field(item.reference).state, Available)
        and point.field(item.reference).state.value.status == "unassigned"  # type: ignore[union-attr]
    }
    if styles:
        point = commit(point, styles)
    composed = point.query(MatMulKernel.structure)
    if not isinstance(composed, Available):
        raise ValueError(f"MatMul assembly is not accepted: {describe([composed])}")
    folding = point.folding
    delivered: Space | None = point.delivery
    return (
        MatMulAssembly(
            folding.activation_beats,
            folding.weight_beats,
            folding.result_beats,
            point.result_type,
            weight_delivery,
            composed.value.structure,
            composed.value.requirements,
            () if delivered is None else delivered.image,  # type: ignore[attr-defined]
        ),
        settled,
    )


__all__ = [
    "ADMISSION",
    "OwnedRamStyleStream",
    "RefinedBufferedStream",
    "RefinedMatMul",
    "RefinedStream",
    "refined_assembly",
]
