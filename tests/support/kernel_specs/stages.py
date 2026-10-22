# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The channel stages: fifo, vpc and input_gen, each between the two tensors it moves words
between.

No KernelOp reaches a channel stage: a channel's adapter places it (``finn.kernels.
adapters.realize``), giving it its tensor and the beat sequences its ports present. Its
case places the stage itself on two channels, its ports presenting what its realization
gives them, so XSim checks that the RTL walks the realization; each configuration states
the generics the stage derives from them, which the harness checks. Each case's class is
its stage under the harness's own id (``StagedFifo``, ``StagedVpc``,
``StagedInputGenerator``), which seeds its stimulus. Its test-side reference is the
identity: a stage moves elements, it computes none.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from finn.core.space import view
from finn.dataflow.steps import steps
from finn.dataflow.traversal import BeatSequence, Traversal, vector_major
from finn.kernels.adapters import Convert, RealizedStage, realize
from finn.kernels.artifacts.abi import Direction, Endpoint, Pin, Signal
from finn.kernels.artifacts.module import Held
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import Generate, InputGeneratorKernel
from finn.kernels.port import native_stream
from finn.kernels.vpc import VpcKernel
from support.adapted import columns_first
from support.helpers import FULL_DSP48E2, OtherPins
from support.kernel_specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed
from support.kernel_specs.thresholding import tensor


class StagedFifo(FifoKernel):
    """FifoKernel under the harness's id: what arrives leaves unchanged."""

    id = "test.staged.fifo"


class StagedVpc(VpcKernel):
    """VpcKernel under the harness's id: a width conversion."""

    id = "test.staged.vpc"


class StagedInputGenerator(InputGeneratorKernel):
    """InputGeneratorKernel under the harness's id: a reorder. ``olst``, its loop-completion
    marker, is left unused: the harness's boundaries carry no marker, so its output
    presents ``leaving`` on its ready/valid pins alone, and ``olst`` is a pin it holds
    unused. The dotp and MatMul cases check it, as the TLAST that frames their
    activations."""

    id = "test.staged.input_generator"

    output = native_stream(
        "output",
        Endpoint.INITIATOR,
        InputGeneratorKernel.output_tensor,
        InputGeneratorKernel.leaving,
    )

    @view
    def loop_end_pins(self) -> tuple[Pin, ...]:
        return (Signal("olst", Direction.OUT, len(self.dims)),)

    unused = OtherPins(loop_end_pins, Held((), ("olst",)))


STAGED = (4, 6)


def realized(source: Traversal, sink: Traversal) -> RealizedStage:
    """The one stage a channel's adapter realizes between ``source`` and ``sink``."""
    (stage,) = realize(steps(BeatSequence(source), BeatSequence(sink)))
    return stage


def staged(stage: RealizedStage) -> dict[str, object]:
    """Its two sequences; the markers its output guarantees are left unused here."""
    return {"arriving": stage.source, "leaving": BeatSequence(stage.sink.form)}


def fifo() -> dict[str, Any]:
    """The identity in each storage the RTL implements: shift, LUTRAM, block and UltraRAM."""

    def storing(lanes: int, depth: int, ram_style: str) -> dict[str, object]:
        same = BeatSequence(vector_major(STAGED, lanes))
        return {
            "input_data_width": 4 * lanes,
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
    """The same elements in the same order, at another lane count; W, the element's bits,
    is the tensor's (INT4)."""

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
    )


def input_gen() -> dict[str, Any]:
    """A reorder as the channel's adapter realizes it: a transpose, word by word and two
    lanes a word, and each row replayed."""

    def generating(source: Traversal, sink: Traversal) -> dict[str, object]:
        stage = realized(source, sink)
        module = stage.module
        assert isinstance(module, Generate)
        return {
            "input_data_width": 4 * source.lanes,
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
            {"ram_style": "ultra"},
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
            {"ram_style": "ultra"},
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
    "staged",
    "vpc",
]
