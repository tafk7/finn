# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical ready/valid components for opaque words.

These requirements describe transport and initialization only. Payload bits are
preserved without padding, signed interpretation, or element reordering. Both
components use ``clk`` and a synchronous active-high ``rst``; a transfer occurs
on a rising edge with valid and ready asserted outside reset. Output payload and
framing stay stable while a valid transfer is stalled.
"""

from collections.abc import Sequence

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)

REPLAY_BUFFER_SOURCES = (
    CopiedSource("finnlib", "rtl/infra/replay_buffer.sv", provides=("module:replay_buffer",)),
)


def _positive(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _clock_reset() -> tuple[Signal, Signal]:
    return (
        Signal("clk", Direction.IN, 1, Clock(Free())),
        Signal(
            "rst",
            Direction.IN,
            1,
            Reset(active_low=False, synchronous=True, synchronous_to=("clk",)),
        ),
    )


def _output(word_bits: int, *, last: bool = False) -> Bus:
    return Bus(
        "out0",
        StandardProtocol.AXIS,
        (
            Member("tdata", "odat", word_bits),
            Member("tvalid", "ovld"),
            Member("tready", "ordy"),
            *((Member("tlast", "olast"),) if last else ()),
        ),
        endpoint=Endpoint.INITIATOR,
        associated_clock="clk",
        associated_reset="rst",
    )


def replay_buffer_requirements(
    *, word_bits: int, sequence_length: int, replay_count: int
) -> ModuleBuildRequirements:
    """Replay each consecutive ``sequence_length`` input words ``replay_count`` times.

    ``out0.tlast``/``olast`` marks the final word of every output sequence.
    ``ofin`` marks that word only on its last repetition, completing one input
    sequence's replay. Both are valid-qualified levels, held under backpressure;
    neither is a separate completion pulse. Input sequences are counted and
    have no last pin. Reset discards buffered words and restarts both counters.

    A replay count of one selects FinnLib's combinational identity path. Its
    valid/ready signals remain combinational during reset, so endpoints must
    disregard transfers while reset is asserted.
    """
    _positive("word_bits", word_bits)
    _positive("sequence_length", sequence_length)
    _positive("replay_count", replay_count)
    parameters = (("LEN", sequence_length), ("REP", replay_count), ("W", word_bits))
    return ModuleBuildRequirements(
        "replay_buffer",
        "1",
        parameters,
        ModuleABIRequirements(
            FixedModuleName("replay_buffer"),
            (
                *_clock_reset(),
                Bus(
                    "in0",
                    StandardProtocol.AXIS,
                    (
                        Member("tdata", "idat", word_bits),
                        Member("tvalid", "ivld"),
                        Member("tready", "irdy"),
                    ),
                    endpoint=Endpoint.TARGET,
                    associated_clock="clk",
                    associated_reset="rst",
                ),
                _output(word_bits, last=True),
                Signal("ofin", Direction.OUT, 1),
            ),
            tuple((name, str(value)) for name, value in parameters),
        ),
        REPLAY_BUFFER_SOURCES,
    )


def cyclic_stream_requirements(
    *, word_bits: int, depth: int, image: Sequence[int]
) -> ModuleBuildRequirements:
    """Continuously stream an explicitly initialized, read-only image in order.

    ``image`` must contain exactly ``depth`` unsigned raw words, each fitting
    ``word_bits``. Word zero occupies the low bits of the packed ``INIT_DATA``
    RTL parameter, so contents are part of the requirements and concrete ABI.
    No initialization file or runtime reload interface is needed or implied.

    After reset, output starts at image word zero. Accepted transfers advance
    through the image and wrap at ``depth``; there is no last pin or finite-run
    completion. The registered output can sustain one transfer per cycle and
    holds its word until ready. Reset discards a pending output word.
    """
    _positive("word_bits", word_bits)
    _positive("depth", depth)
    words = tuple(image)
    if len(words) != depth:
        raise ValueError("image must contain exactly depth words")
    if any(type(word) is not int or not 0 <= word < (1 << word_bits) for word in words):
        raise ValueError("image words must be unsigned integers fitting word_bits")
    packed = sum(word << (index * word_bits) for index, word in enumerate(words))
    parameters = (
        ("DEPTH", depth),
        ("INIT_DATA", f"{word_bits * depth}'h{packed:x}"),
        ("W", word_bits),
    )
    return ModuleBuildRequirements(
        "cyclic_stream",
        "1",
        parameters,
        ModuleABIRequirements(
            FixedModuleName("cyclic_stream"),
            (*_clock_reset(), _output(word_bits)),
            tuple((name, str(value)) for name, value in parameters),
        ),
        (
            CopiedSource(
                "kernels",
                "cyclic_stream.sv",
                provides=("module:cyclic_stream",),
            ),
        ),
    )


__all__ = ["replay_buffer_requirements", "cyclic_stream_requirements"]
