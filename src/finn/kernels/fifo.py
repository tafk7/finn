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

The FIFO's ``auto`` is FinnLib's selection, not Vivado's: it is its first case, and its
statement is the storage the RTL selects, exact. One part is Vivado's: a decomposed
memory's ``hi`` space shallower than its primitive, which the RTL leaves ``auto``; it
is stated by a model of Vivado's placement (``_hi_style``).
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
    view,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.port import WordPort
from finn.kernels.target import Platform, uram_requirements
from finn.kernels.utilization import RESOURCES_SEMANTICS, Fabric, Resources, lutram, memory


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


#: The model of Vivado's placement of a FIFO's shallow ``hi`` space (``_hi_style``), as
#: observed on xczu3eg: a memory deeper than 64 words in block RAM from 8192 bits
#: (LUTRAM up to 6272 bits, 128 x 32 and 32 x 196; between them, not observed), and one
#: of at most 64 words from 65 536 (LUTRAM at 8192, block RAM at 100 352).
_HI_BLOCK_BITS = 8192
_HI_BLOCK_WIDE_BITS = 65536


def _hi_style(words: int, bits: int) -> str:
    """The style Vivado places a FIFO's shallow ``hi`` space of ``words x bits`` in, which
    the RTL leaves ``auto``: a model, observed on UltraScale+ (xczu3eg) and taken on
    every fabric."""
    size = words * bits
    block = (words > 64 and size >= _HI_BLOCK_BITS) or size >= _HI_BLOCK_WIDE_BITS
    return "block" if block else "distributed"


def fifo_resources(depth: int, data_width: int, ram_style: str, *, fabric: Fabric) -> Resources:
    """FinnLib ``fifo`` for DEPTH, DATA_WIDTH and the RAM_STYLE it is given, in the
    storage it selects (``_selected``), on ``fabric``. The storage is the RTL's: a shift register of
    ``DEPTH - 1`` words (four at least), a LUT a bit for 32 of them; a LUTRAM of
    ``DEPTH - 1`` words rounded up to a power of two; or block RAM or UltraRAM, its
    ``lo`` space and its ``hi`` one (``_decomposition``), a ``hi`` space shallower than
    the primitive left ``auto`` (a model, ``_hi_style``). The control (pointers, the
    output register, the UltraRAM output queue) is taken from the HWCustomOp flow's FIFO
    model (``finn.custom_op.fpgadataflow.resource_models._fifo_cost``, fitted against
    finn-rtllib's ``fifo.sv``, the same design with an occupancy monitor; the flow is
    deleted, and the finn-dev oracle's capture ``fifo_cost`` keeps its values); the terms
    it does not carry over are listed below."""
    # Carried over from ``finn.custom_op.fpgadataflow.resource_models._fifo_cost``, the
    # HWCustomOp flow's model of the same RTL. Where the two differ, by what this one does
    # not carry over (FinnLib's ``rtl/infra/fifo.sv``):
    # - the storage is stated by ``finn.kernels.utilization`` from the arrays the RTL
    #   declares, in the fabric's primitives, not by the legacy model's fitted packing. A
    #   LUTRAM is RAM64M8s (``lutram``), as fifo.sv's header sizes its ``distributed``
    #   path ("1 LUT/bit via RAM64M8"; DEPTH 257, "the natural capacity of 4x RAM64M8
    #   per byte"), with a write decode and a read multiplexer over its banks; the legacy
    #   model counts RAM32X2s with a fitted 5/4 and a mux LUT for every two bits a
    #   further 128 rows. At 32 bits on UltraScale the two agree to depth 65 and are 2 to
    #   4 LUTs apart above it (257 x 32 bits: 261 LUTs here, 257 there).
    # - the shift path's cascade mux over SRLC32Es, which the legacy model counts on
    #   Versal only (UltraScale+'s F7/F8 absorb it), and the URAM read pipeline's 20 LUTs
    #   on Versal: measured on the legacy model's FIFO, not on this one.
    # - the block path's cascade decode (3 LUTs a level of the tile plan, 2 a 5 tiles):
    #   Vivado's decode of ``MemLo`` across tiles, which fifo.sv does not write and the
    #   legacy model fits; 2028 x 32 bits is 54 LUTs here, 67 there.
    # - the lo/hi output select of a decomposed memory (fifo.sv's ``genOutMux``, a
    #   DATA_WIDTH-wide 2:1 mux and its pointer compare; 18 + 3W/7 fitted), and under
    #   UltraRAM a LUTRAM ``hi`` space's delay to the URAM read latency (W LUTs);
    #   1500 x 32 bits, a 1024-word and a 512-word space, is 54 LUTs here, 92 there.
    # - the UltraRAM read pipeline's growth: fifo.sv's ``PIPE_DEPTH`` is
    #   3 + (2**lo - 1)/8192, a stage more each 8192 rows, which the legacy model fits as
    #   a further W LUTs each 4 cascaded URAMs past 12; here the output queue is W at
    #   any depth (100000 x 72 bits is 180 LUTs here, 540 there).
    # The last three are not measured against this model: they are under-counts of
    # deep or decomposed FIFOs, kept as R1 carried the control over, not refit.
    bits, effective = data_width, _selected(depth, data_width, ram_style)
    counter = depth.bit_length() + 1
    if effective == "shift":
        stages = -(-max(depth - 1, 4) // 32)
        return Resources(lut=stages * bits + 5 * counter - 7, ff=bits + counter)
    if effective == "distributed":
        words = 1 << (depth - 2).bit_length()
        return Resources(
            lut=lutram(words, bits, fabric=fabric) + 8 * counter - 15, ff=bits + 2 * counter
        )
    ultra = effective == "ultra"
    lo, hi = _decomposition(depth, ultra)
    storage = memory(1 << lo, bits, effective, fabric=fabric)
    if hi:
        # The RTL relaxes a hi space shallower than its primitive to ``auto``: Vivado's.
        style = effective if hi >= (12 if ultra else 9) else _hi_style(1 << hi, bits)
        storage = storage + memory(1 << hi, bits, style, fabric=fabric)
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
        requires=uram_requirements(platform, ("ultra",)),
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
        return fifo_resources(
            self.depth, self.word_bits, self.rtl_ram_style, fabric=self.platform.fabric
        )

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
    "least_depth_holding",
]
