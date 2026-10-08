# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's opaque-word FIFO, with its native unpadded ready/valid pins.

Words have no numerical datatype. DEPTH is the requested capacity; the native
implementation may round its storage up and forces a shift FIFO for shallow
depths. The RTL's ``auto`` selects by depth and word width: a shift register up
to 64 words narrower than 12 bits, LUTRAM up to 257 words, block RAM up to 2028,
then UltraRAM. Reset is synchronous, active-high, and discards pending words.
``ultra`` requires the ``platform``'s UltraRAM; the FIFO starts empty, so no
initial contents are asked of it. ``auto`` never takes UltraRAM the platform
lacks: where the RTL's own selection would, the kernel gives it ``block``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    requires,
    view,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.port import WordPort
from finn.kernels.target import Platform
from finn.kernels.utilization import RESOURCES_SEMANTICS, Resources, lutram, memory


@dataclass(frozen=True)
class FifoStorage:
    """The storage the native RTL selects for ``ram_style`` and DEPTH, and the words it
    accepts; synthesis resource inference is separate."""

    effective_style: str
    capacity: int


def _selected(depth: int, bits: int, style: str) -> str:
    """The storage the native RTL implements for DEPTH, DATA_WIDTH and RAM_STYLE."""
    if depth <= 33:
        return "shift"
    if style != "auto":
        return style
    if depth <= 64 and bits < 12:
        return "shift"
    if depth <= 257:
        return "distributed"
    return "block" if depth <= 2028 else "ultra"


def fifo_storage(depth: int, bits: int, style: str) -> FifoStorage | None:
    """The storage the native RTL implements for DEPTH, DATA_WIDTH and the RAM_STYLE it
    is given, and the words it accepts; ``None`` when its memory size overflows."""
    effective = _selected(depth, bits, style)
    if effective == "shift":
        capacity = max(5, depth)
    elif effective == "distributed":
        # DEPTH - 1 LUTRAM entries behind one output register.
        capacity = depth
    else:
        # Include the BRAM read pipeline or the URAM credit-limited output queue in
        # accepted-word capacity.
        ultra = effective == "ultra"
        lo, hi = _decomposition(depth, ultra)
        if lo >= 32:
            return None
        capacity = (1 << lo) + ((1 << hi) if hi else 0) + (17 if ultra else 2)
    return FifoStorage(effective, capacity)


def _decomposition(depth: int, ultra: bool) -> tuple[int, int]:
    """The address bits of the native memory's ``lo`` and ``hi`` spaces (``hi`` 0: none):
    the words beyond the output register or queue, in a power of two or two."""
    required = depth - (17 if ultra else 1)
    lo, hi = (required - 1).bit_length(), 0
    if lo > (12 if ultra else 9):
        remainder_bits = (required - (1 << (lo - 1)) - 1).bit_length()
        if remainder_bits < lo - 1:
            lo, hi = lo - 1, max(1, remainder_bits)
    return lo, hi


def fifo_resources(depth: int, data_width: int, ram_style: str) -> Resources:
    """FinnLib ``fifo`` for DEPTH, DATA_WIDTH and the RAM_STYLE it is given, in the
    storage it selects (``_selected``). The storage is the RTL's: a shift register of
    ``DEPTH - 1`` words (four at least), a LUT a bit for 32 of them; a LUTRAM of
    ``DEPTH - 1`` words rounded up to a power of two; or block RAM or UltraRAM, its
    ``lo`` space and its ``hi`` one (``_decomposition``), a ``hi`` space shallower than
    the primitive left ``auto``. The control (pointers, the output register, the
    UltraRAM output queue) is FINN's FIFO model (``finn.util.resource_models``, fitted
    against finn-rtllib's ``fifo.sv``, the same design with an occupancy monitor)."""
    bits, effective = data_width, _selected(depth, data_width, ram_style)
    counter = depth.bit_length() + 1
    if effective == "shift":
        stages = -(-max(depth - 1, 4) // 32)
        return Resources(lut=stages * bits + 5 * counter - 7, ff=bits + counter)
    if effective == "distributed":
        words = 1 << (depth - 2).bit_length()
        return Resources(lut=lutram(words, bits) + 8 * counter - 15, ff=bits + 2 * counter)
    ultra = effective == "ultra"
    lo, hi = _decomposition(depth, ultra)
    storage = memory(1 << lo, bits, effective)
    if hi:
        # The RTL relaxes a hi space shallower than its primitive to ``auto``.
        storage = storage + memory(
            1 << hi, bits, effective if hi >= (12 if ultra else 9) else "auto"
        )
    if ultra:
        return storage + Resources(lut=6 * counter + bits, ff=2 * bits)
    return storage + Resources(lut=54, ff=bits + 2 * counter)


def _given(depth: int, bits: int, style: str, uram: bool) -> str:
    """The RAM_STYLE the RTL is given for ``style``: block RAM where ``auto`` would
    resolve to UltraRAM a platform without it lacks."""
    if style == "auto" and not uram and _selected(depth, bits, style) == "ultra":
        return "block"
    return style


def least_depth_holding(words: int, bits: int, style: str, uram: bool) -> int:
    """The least DEPTH (two at least) whose native storage accepts ``words`` words: a
    shallow FIFO is a shift register of five words, so DEPTH 2 holds up to five."""
    depth = 2
    while True:
        found = fifo_storage(depth, bits, _given(depth, bits, style, uram))
        if found is not None and found.capacity >= words:
            return depth
        depth += 1


class FifoKernel(Kernel):
    id = "finnlib.fifo"
    version = 2
    rtl_module = "fifo"

    word_bits: int = Param()
    depth: int = Param()
    platform: Platform = Param()

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        bits = self.word_bits
        depth = self.depth
        if bits < 1 or depth < 2 or max(bits, depth) > 0xFFFFFFFF:
            return reject("fifo-geometry", "word_bits must be positive and depth at least two")
        return True

    ram_style: str = Decision(
        values=("auto", "shift", "distributed", "block", "ultra"),
        requires=(
            requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=("ultra",)),
        ),
    )

    @derived
    def rtl_ram_style(self) -> str:
        """The RAM_STYLE the RTL is given: ``ram_style``, except an ``auto`` the RTL would
        resolve to UltraRAM the platform lacks, which is block RAM."""
        return _given(self.depth, self.word_bits, self.ram_style, self.platform.uram)

    @view(requires=(geometry_supported,))
    def storage(self) -> FifoStorage | Rejected:
        found = fifo_storage(self.depth, self.word_bits, self.rtl_ram_style)
        if found is None:
            return reject("fifo-capacity", "native memory size overflows unsigned int")
        return found

    @constraint
    def capacity_supported(self) -> bool | Rejected:
        """The native memory the storage decomposes into is addressable."""
        _ = self.storage
        return True

    admission = ConstraintGroup(geometry_supported, capacity_supported)

    input = WordPort(name="input", endpoint=Endpoint.TARGET, bits=word_bits)
    output = WordPort(name="output", endpoint=Endpoint.INITIATOR, bits=word_bits)

    @derived
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        return fifo_resources(self.depth, self.word_bits, self.rtl_ram_style)

    def parameters(self) -> Mapping[str, int | str]:
        return {
            "DATA_WIDTH": self.word_bits,
            "DEPTH": self.depth,
            "RAM_STYLE": f'"{self.rtl_ram_style}"',
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),)


__all__ = [
    "FifoKernel",
    "FifoStorage",
    "fifo_resources",
    "fifo_storage",
    "least_depth_holding",
]
