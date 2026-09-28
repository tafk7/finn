# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A physical ready/valid component for opaque words: the cyclic stream.

These requirements describe transport and initialization only. Payload bits are
preserved without padding, signed interpretation, or element reordering. The
component uses ``clk`` and a synchronous active-high ``rst``; a transfer occurs
on a rising edge with valid and ready asserted outside reset. Output payload is
stable while a valid transfer is stalled.

FinnLib's ``replay_buffer`` is not wrapped: ``input_gen`` realizes every replay
it could, and marker synthesis too, as a stage of the consuming stream
(``finn.kernels.adapters``).
"""

from __future__ import annotations

from collections.abc import Sequence

from finn.kernels.artifacts.abi import (
    Clock,
    Direction,
    Endpoint,
    Free,
    Reset,
    Signal,
)
from finn.kernels.physical.stream import ReadyValidStream
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)


def _positive(name: str, value: int) -> None:
    if type(value) is not int or not 1 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be a positive native unsigned integer")


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


def cyclic_stream_interface(*, word_bits: int) -> ReadyValidStream:
    _positive("word_bits", word_bits)
    return ReadyValidStream(
        "output", word_bits, Endpoint.INITIATOR, "odat", "ovld", "ordy", "clk", "rst"
    )


CYCLIC_ROM_STYLES = ("auto", "distributed", "block")


def cyclic_stream_requirements(
    *, word_bits: int, depth: int, image: Sequence[int], rom_style: str
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

    ``rom_style`` selects the synthesis ``rom_style`` attribute: ``auto`` leaves
    inference to the tool, ``distributed`` requests LUT ROM and ``block`` block
    RAM. UltraRAM is not offered: bitstream initialization of UltraRAM is not
    available on every supported device family.
    """
    _positive("word_bits", word_bits)
    _positive("depth", depth)
    if rom_style not in CYCLIC_ROM_STYLES:
        raise ValueError(f"rom_style must be one of {', '.join(CYCLIC_ROM_STYLES)}")
    words = tuple(image)
    if len(words) != depth:
        raise ValueError("image must contain exactly depth words")
    if any(type(word) is not int or not 0 <= word < (1 << word_bits) for word in words):
        raise ValueError("image words must be unsigned integers fitting word_bits")
    packed = sum(word << (index * word_bits) for index, word in enumerate(words))
    parameters = (
        ("DEPTH", depth),
        ("INIT_DATA", f"{word_bits * depth}'h{packed:x}"),
        ("ROM_STYLE", f'"{rom_style}"'),
        ("W", word_bits),
    )
    return ModuleBuildRequirements(
        "cyclic_stream",
        "1",
        parameters,
        ModuleABIRequirements(
            FixedModuleName("cyclic_stream"),
            (*_clock_reset(), *cyclic_stream_interface(word_bits=word_bits).pins()),
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


__all__ = [
    "CYCLIC_ROM_STYLES",
    "cyclic_stream_requirements",
    "cyclic_stream_interface",
]
