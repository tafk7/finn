# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib's FIFO, on its native unpadded ready/valid pins.

It buffers the words of a tensor (``input_tensor``) and presents on both sides what
arrives (``arriving``, a beat sequence it is given: a channel's transport gives it what
reaches it); its words are opaque to the RTL.
``DATA_WIDTH`` is its input port's word, the lanes a beat of the tensor's element. Its
cycles are its ports' beats, as every leaf's, and its buffering its depth of words.
DEPTH is the requested capacity; the native
implementation may round its storage up and forces a shift FIFO for shallow
depths. The RTL's ``auto`` selects by depth and word width: a shift register up
to 64 words narrower than 12 bits, LUTRAM up to 257 words, block RAM up to 2028,
then UltraRAM. Reset is synchronous, active-high, and discards pending words.
``ultra`` needs the ``platform``'s UltraRAM, a constraint of the FIFO (``uram-absent``); the
FIFO starts empty, so no initial contents are asked of it. ``auto`` never takes UltraRAM the
platform lacks: where the RTL's own selection would, the kernel gives it ``block``.

The FIFO's ``auto`` is FinnLib's selection, not Vivado's: it is listed first (first valid
takes it), and its
statement is the storage the RTL selects, exact. One part is Vivado's: a decomposed
memory's ``hi`` space shallower than its primitive, which the RTL leaves ``auto``; it
is stated by a model of Vivado's placement (``_hi_style``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import BeatSequence
from finn.dataflow.streams import Stream
from finn.kernels.artifacts.abi import Clocking, Endpoint, Pin
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace
from finn.kernels.library import ModuleSpec, _each, assemble_module, module_unprovided
from finn.kernels.port import held, pins
from finn.kernels.port_model import native_port
from finn.kernels.utilization import lutram, memory, uram_constraint
from finn.platform.capabilities import Fabric, Platform, Resources


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
    finn-rtllib's ``fifo.sv``, the same design with an occupancy monitor); the terms it
    does not carry over are listed below."""
    # Carried over from ``finn.custom_op.fpgadataflow.resource_models._fifo_cost``, the
    # HWCustomOp flow's model of the same RTL (the HWCustomOp model below). Where the two
    # differ, by what this one does not carry over (FinnLib's ``rtl/infra/fifo.sv``):
    # - the storage is stated by ``finn.kernels.utilization`` from the arrays the RTL
    #   declares, in the fabric's primitives, not by the HWCustomOp model's fitted packing. A
    #   LUTRAM is RAM64M8s (``lutram``), as fifo.sv's header sizes its ``distributed``
    #   path ("1 LUT/bit via RAM64M8"; DEPTH 257, "the natural capacity of 4x RAM64M8
    #   per byte"), with a write decode and a read multiplexer over its banks; that model
    #   counts RAM32X2s with a fitted 5/4 and a mux LUT for every two bits a
    #   further 128 rows. At 32 bits on UltraScale the two agree to depth 65 and are 2 to
    #   4 LUTs apart above it (257 x 32 bits: 261 LUTs here, 257 there).
    # - the shift path's cascade mux over SRLC32Es, which that model counts on Versal
    #   only (UltraScale+'s F7/F8 absorb it), and the URAM read pipeline's 20 LUTs on
    #   Versal: measured on that model's FIFO, not on this one.
    # - the block path's cascade decode (3 LUTs a level of the tile plan, 2 a 5 tiles):
    #   Vivado's decode of ``MemLo`` across tiles, which fifo.sv does not write and that
    #   model fits; 2028 x 32 bits is 54 LUTs here, 67 there.
    # - the lo/hi output select of a decomposed memory (fifo.sv's ``genOutMux``, a
    #   DATA_WIDTH-wide 2:1 mux and its pointer compare; 18 + 3W/7 fitted), and under
    #   UltraRAM a LUTRAM ``hi`` space's delay to the URAM read latency (W LUTs);
    #   1500 x 32 bits, a 1024-word and a 512-word space, is 54 LUTs here, 92 there.
    # - the UltraRAM read pipeline's growth: fifo.sv's ``PIPE_DEPTH`` is
    #   3 + (2**lo - 1)/8192, a stage more each 8192 rows, which that model fits as
    #   a further W LUTs each 4 cascaded URAMs past 12; here the output queue is W at
    #   any depth (100000 x 72 bits is 180 LUTs here, 540 there).
    # The last three are not measured against this model: they are under-counts of
    # deep or decomposed FIFOs, kept as the control was carried over, not refit.
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


def _geometry_supported(bits: int, depth: int) -> bool:
    """Whether the native FIFO takes DATA_WIDTH ``bits`` and DEPTH ``depth``: a positive
    word, two words at least, each within an unsigned int."""
    return bits >= 1 and depth >= 2 and max(bits, depth) <= 0xFFFFFFFF


#: The finding of a geometry the native FIFO does not take (``_geometry_supported``).
_GEOMETRY = ("fifo-geometry", "input_data_width must be positive and depth at least two")


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


class FifoKernel(KernelSpace):
    """FinnLib ``fifo``: what arrives of its tensor, presented again on its output; see the
    module docstring."""

    id = "finnlib.fifo"
    clock_domain = NATIVE_CLOCKING

    # The tensor whose words it buffers, a fact, and what arrives of it.
    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    depth: int = Param()
    platform: Platform = Param()

    @derived
    def output_tensor(self) -> Tensor:
        """What it moves its input to: the same tensor."""
        tensor: Tensor = self.input_tensor
        return tensor

    @derived
    def leaving(self) -> BeatSequence:
        """What it presents on its output: what arrives."""
        sequence: BeatSequence = self.arriving
        return sequence

    @derived
    def port_clocking(self) -> Clocking:
        """Native physical names follow the class clock_domain, including overrides."""
        return type(self).clock_domain

    input = native_port(
        "input", Endpoint.TARGET, input_tensor, arriving, clocking=port_clocking, label="input"
    )
    output = native_port(
        "output", Endpoint.INITIATOR, output_tensor, leaving, clocking=port_clocking, label="output"
    )

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        if not _geometry_supported(self.input.width, self.depth):
            return reject(*_GEOMETRY)
        return True

    ram_style: str = Decision(
        domain=("auto", "shift", "distributed", "block", "ultra"),
    )
    uram_present = uram_constraint(platform, ram_style, ("ultra",))

    @derived
    def rtl_ram_style(self) -> str:
        """The RAM_STYLE the RTL is given: ``ram_style``, except an ``auto`` the RTL would
        resolve to UltraRAM the platform lacks, which is block RAM."""
        return _given(self.depth, self.input.width, self.ram_style, self.platform.uram)

    @derived
    def planned_storage(self) -> FifoStorage | Rejected:
        """The storage the native RTL selects (``fifo_storage``), of a geometry it takes:
        refused, as ``geometry_supported`` is, for one it does not."""
        if not _geometry_supported(self.input.width, self.depth):
            return reject(*_GEOMETRY)
        found = fifo_storage(self.depth, self.input.width, self.rtl_ram_style)
        if found is None:
            return reject("fifo-capacity", "native memory size overflows unsigned int")
        return found

    # The storage, the product its holder reads; its rule reads the value.
    storage = view(planned_storage)

    @constraint
    def capacity_supported(self) -> bool | Rejected:
        """The native memory the storage decomposes into is addressable: refused, too,
        with the geometry its storage is planned for, when that is refused."""
        _ = self.planned_storage
        return True

    @view
    def resource_use(self) -> Resources:
        return fifo_resources(
            self.depth, self.input.width, self.rtl_ram_style, fabric=self.platform.fabric
        )

    @view
    def held_bits(self) -> int:
        """What it holds: its depth of words."""
        return self.depth * self.input.width

    @view
    def parameters(self) -> dict[str, int | str]:
        return {
            "DATA_WIDTH": self.input.width,
            "DEPTH": self.depth,
            "RAM_STYLE": f'"{self.rtl_ram_style}"',
        }

    module_spec: ClassVar[ModuleSpec] = ModuleSpec(
        "fifo", (CopiedSource("finnlib", "rtl/infra/fifo.sv", provides=("module:fifo",)),)
    )

    def _input_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.input.presenting()
        return pins(
            self.input.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.input.clocking,
        )

    def _output_pins(self) -> tuple[Pin, ...] | Rejected:
        presenting = self.output.presenting()
        return pins(
            self.output.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.output.clocking,
        )

    def _input_held(self) -> Held | Rejected:
        if self.input.placed:
            return Held()
        element = self.input.element
        return held(self.input.physical_spec(idle=True), element, self.input.clocking)

    def _output_held(self) -> Held | Rejected:
        if self.output.placed:
            return Held()
        element = self.output.element
        return held(self.output.physical_spec(idle=True), element, self.output.clocking)

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        pin_parts = _each((self._input_pins(), self._output_pins()))
        if isinstance(pin_parts, Rejected):
            return pin_parts
        spec = type(self).module_spec
        sources = spec.constant_sources
        assert sources is not None  # Native module has constant sources; no dynamic source View.
        refused = module_unprovided(spec, type(self).__qualname__, sources)
        if isinstance(refused, Rejected):
            return refused
        data = ()
        held_parts = _each((self._input_held(), self._output_held()))
        if isinstance(held_parts, Rejected):
            return held_parts
        leaf = assemble_module(
            spec,
            type(self).id,
            type(self).__qualname__,
            clocking,
            parameters,
            tuple(pin_parts),
            sources,
            data,
            tuple(held_parts),
            (True, True),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int:
        return max(self.input.stream.sequence.form.beats, self.output.stream.sequence.form.beats)


__all__ = [
    "FifoKernel",
    "FifoStorage",
    "fifo_resources",
    "least_depth_holding",
]
