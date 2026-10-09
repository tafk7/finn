# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The channel stages: fifo, vpc and input_gen, each on the two channels it moves words
between.

No KernelOp reaches a channel stage: a channel's adapter places it (``finn.kernels.
adapters.realize``). Its ports are opaque words, and its realization states what they
carry; its case places it on two channels (``StagedFifo``, ``StagedVpc``,
``StagedInputGenerator``: its module, its native pins, ports presenting what its
realization gives them), so XSim checks that the RTL walks the realization. Its
test-side reference is the identity: a stage moves elements, it computes none.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from finn.core.space import Param, derived
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.plan import plan
from finn.dataflow.traversal import BeatSequence, Traversal, vector_major
from finn.kernels.adapters import Convert, Generate, RealizedStage, realize
from finn.kernels.artifacts.abi import Direction, Endpoint, Signal
from finn.kernels.artifacts.module import Held
from finn.kernels.base import NATIVE_CLOCKING
from finn.kernels.channels import Channel
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.values.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.vpc import VpcKernel
from kernels.adapted import columns_first
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed
from kernels.specs.thresholding import tensor


def stage_ports(
    input_channel: Any, output_channel: Any, arriving: Any, leaving: Any, moved: Any
) -> tuple[AxiStreamPort, AxiStreamPort]:
    """A channel stage's native pins on the two channels the harness places, presenting the
    beat sequences its realization gives them (``arriving``, ``leaving``).

    On a channel a stage's ports are opaque words (``WordPort``) and its adapter states
    what they carry (``finn.kernels.adapters.realize``); here the ports sit on channels
    and present exactly that, so XSim checks that the RTL walks it.
    """
    input = AxiStreamPort(
        name="input",
        endpoint=Endpoint.TARGET,
        channel=input_channel,
        sequence=arriving,
        signals=("idat", "ivld", "irdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    output = AxiStreamPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        channel=output_channel,
        sequence=leaving,
        dtype=moved,
        signals=("odat", "ovld", "ordy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    return input, output


class StagedFifo(FifoKernel):
    """FifoKernel between two channels: what arrives leaves unchanged."""

    id = "test.staged.fifo"
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def moved(self) -> QONNXDataType:
        return self.input_channel.tensor.element.dtype

    # On channels, in place of the stage's opaque word ports (WordPort).
    input, output = stage_ports(  # type: ignore[assignment]
        input_channel, output_channel, arriving, leaving, moved
    )


class StagedVpc(VpcKernel):
    """VpcKernel between two channels: a width conversion."""

    id = "test.staged.vpc"
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def moved(self) -> QONNXDataType:
        return self.input_channel.tensor.element.dtype

    # On channels, in place of the stage's opaque word ports (WordPort).
    input, output = stage_ports(  # type: ignore[assignment]
        input_channel, output_channel, arriving, leaving, moved
    )


class StagedInputGenerator(InputGeneratorKernel):
    """InputGeneratorKernel between two channels: a reorder. ``olst``, its loop-completion
    marker, is left unused: the harness's boundaries carry no marker. The dotp and MatMul
    cases check it, as the TLAST that frames their activations."""

    id = "test.staged.input_generator"
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def moved(self) -> QONNXDataType:
        return self.input_channel.tensor.element.dtype

    # On channels, in place of the stage's opaque word ports (WordPort).
    input, output = stage_ports(  # type: ignore[assignment]
        input_channel, output_channel, arriving, leaving, moved
    )

    def other_pins(self) -> tuple[Signal, ...]:
        return (Signal("olst", Direction.OUT, len(self.dims)),)

    def held(self) -> Held:
        return Held((), ("olst",))


STAGED = (4, 6)


def realized(source: Traversal, sink: Traversal) -> RealizedStage:
    """The one stage a channel's adapter realizes between ``source`` and ``sink``."""
    (stage,) = realize(plan(BeatSequence(source), BeatSequence(sink)))
    return stage


def staged(stage: RealizedStage) -> dict[str, object]:
    """Its two sequences; the markers its output guarantees are left unused here."""
    return {"arriving": stage.source, "leaving": BeatSequence(stage.sink.form)}


def fifo() -> dict[str, Any]:
    """The identity in each storage the RTL implements: shift, LUTRAM, block and UltraRAM."""

    def storing(lanes: int, depth: int, ram_style: str) -> dict[str, object]:
        same = BeatSequence(vector_major(STAGED, lanes))
        return {
            "word_bits": 4 * lanes,
            "depth": depth,
            "ram_style": ram_style,
            "arriving": same,
            "leaving": same,
        }

    return dict(
        space_type=StagedFifo,
        inputs={"input_channel": tensor(STAGED, "INT4")},
        outputs={"output_channel": STAGED},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=(
            storing(1, 2, "auto"),
            storing(3, 100, "auto"),
            storing(6, 600, "auto"),
            storing(2, 64, "ultra"),
        ),
        facts={"platform": FULL_DSP48E2},
    )


def vpc() -> dict[str, Any]:
    """The same elements in the same order, at another lane count."""

    def converting(lanes_in: int, lanes_out: int) -> dict[str, object]:
        stage = realized(vector_major(STAGED, lanes_in), vector_major(STAGED, lanes_out))
        assert stage.module == Convert(lanes_in, lanes_out)
        return {"lanes_in": lanes_in, "lanes_out": lanes_out, **staged(stage)}

    return dict(
        space_type=StagedVpc,
        inputs={"input_channel": tensor(STAGED, "INT4")},
        outputs={"output_channel": STAGED},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=(converting(1, 3), converting(2, 3), converting(3, 2), converting(6, 1)),
        facts={"element_bits": 4},
    )


def input_gen() -> dict[str, Any]:
    """A reorder as the channel's adapter realizes it: a transpose, word by word and two
    lanes a word, and each row replayed."""

    def generating(source: Traversal, sink: Traversal) -> dict[str, object]:
        stage = realized(source, sink)
        module = stage.module
        assert isinstance(module, Generate)
        return {
            "word_bits": 4 * source.lanes,
            "frame_words": module.frame,
            "dims": module.dims,
            "strides": module.coefs,
            **staged(stage),
        }

    rows, cols = STAGED
    return dict(
        space_type=StagedInputGenerator,
        inputs={"input_channel": tensor(STAGED, "INT4")},
        outputs={"output_channel": STAGED},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=(
            generating(vector_major(STAGED, 1), columns_first(rows, cols, 1)),
            generating(vector_major(STAGED, 2), columns_first(rows, cols, 2)),
            generating(vector_major(STAGED, 3), vector_major(STAGED, 3).replayed(2, inner_beats=2)),
        ),
        choices={"ram_style": "auto"},
        facts={"platform": FULL_DSP48E2},
    )


ADAPTERS = SweepCases("kernels.sweeps.adapter_numeric", ("ADAPTED",), ("sweep-adapters",))

FIFO = KernelSpec(
    kernel=FifoKernel,
    reference=TEST_SIDE,
    cases={"fifo": fifo},
    space="fifo",
    probes=(
        Probe(
            "UltraRAM on a platform without it",
            lambda: placed(fifo(), platform=replace(FULL_DSP48E2, uram=False)),
            frozenset({"uram-absent"}),
            {"kernel.ram_style": "ultra"},
        ),
    ),
    sweeps=(
        SweepCases(
            "kernels.sweeps.matmul_numeric",
            ("CASES",),
            ("sweep-fifo-packed", "sweep-fifo-int8-pumped"),
        ),
    ),
    unit=("tests/kernels/test_fifo.py", "tests/kernels/test_fifo_sizing.py"),
)

VPC = KernelSpec(
    kernel=VpcKernel,
    reference=TEST_SIDE,
    cases={"vpc": vpc},
    space="vpc",
    sweeps=(ADAPTERS,),
    unit=("tests/kernels/test_vpc.py", "tests/kernels/test_adapters.py"),
)

INPUT_GEN = KernelSpec(
    kernel=InputGeneratorKernel,
    reference=TEST_SIDE,
    cases={"input_gen": input_gen},
    space="input_gen",
    probes=(
        Probe(
            "UltraRAM on a platform without it",
            lambda: placed(input_gen(), platform=replace(FULL_DSP48E2, uram=False)),
            frozenset({"uram-absent"}),
            {"kernel.ram_style": "ultra"},
        ),
    ),
    sweeps=(ADAPTERS,),
    unit=("tests/kernels/test_input_generator.py", "tests/kernels/test_adapters.py"),
)

__all__ = [
    "FIFO",
    "INPUT_GEN",
    "STAGED",
    "StagedFifo",
    "StagedInputGenerator",
    "StagedVpc",
    "VPC",
    "fifo",
    "input_gen",
    "realized",
    "stage_ports",
    "staged",
    "vpc",
]
