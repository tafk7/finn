# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ends a shell offers on its boundary channels: where a partition's stream meets memory.

A boundary channel of a shell root has a side without a user, its **free side**: the
AXIS port the root presents (``Channel.free_side``). A shell with ends offers each
such channel the ends it can place there (``EndOffer``), and the channel's ``end``
Decision chooses one, by kind (``ENDS``), as it places its ``source``; one offered is
forced, so nothing is persisted. An end binds no RTL in the root's module: the
module's pins stay the root's, and the shell integrates the end beside the IP from
its facts (``EndContract``).

**``IODMA_hls``** (kind ``iodma_hls``), the Zynq shell's memory mover, modelled at its
stream side, the free side's contract:

- its memory port is ``gcd(frame bits, cap)``, a frame's bits the stream's padded
  ``tdata`` times its beats, as ``InsertIODMA`` sizes ``intfWidth``; a frame is
  ``words`` memory words, and a width converter sits between memory and stream when
  the two widths differ;
- its cycles a frame are ``max(beats, words) + ceil(c / frames a call)``: a beat or a
  word a cycle, and each call's constant ``c``, which the offer states with and
  without a converter (``iodma_hls``: measured in XSim against a memory model, 4
  cycles a call with a converter, 13 without; the call ends two converter words
  before the stream runs dry, or once its last word left);
- it presents one AXI-Lite bus to the shell's processor (its ``s_axi_control``),
  which the shell root's admission counts against the shell's budget beside the
  partition's (``END_CONTROL``, ``EndContract.control_buses``);
- the memory latency ``L`` is **excluded**: the shell does not know it, and each call
  adds it (``MEMORY_LATENCY``, what the exploration report states). A frame a call is
  the bound that holds for any batch, since the batch is the driver's runtime argument;
- its beat times over a frame (``EndContract.times``), one a cycle, or once the memory
  word that completes the beat arrived where the memory is the narrower; FIFO sizing
  reads them, and its cycles as its span, as a kernel's port's pace
  (``finn.kernels.fifo_sizing.ends``).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, gcd

from finn.core.space import (
    ConstraintGroup,
    Param,
    Rejected,
    Space,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    reject,
)
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.transport import StreamContract

IODMA_HLS = "iodma_hls"
"""The kind of the ``IODMA_hls`` end."""

MEMORY_LATENCY = "memory latency: unmeasured; each call adds L"
"""What an end's cycles leave out, as the exploration report states it."""

END_CONTROL = ViewKey("end_control", default_semantics(int))
"""The AXI-Lite buses a boundary channel's end presents to the shell (none without an
end), which the shell root's admission counts."""


@dataclass(frozen=True, kw_only=True)
class EndOffer:
    """What a shell offers a boundary channel's free side: an end's ``kind`` (a key of
    ``ENDS``), its memory port's widest (``width_cap``, bits), the frames each call
    moves (``frames_per_call``), and the cycles each call adds beside the memory
    latency, with a width converter (``call_converted``) and without
    (``call_direct``)."""

    kind: str
    width_cap: int
    frames_per_call: int
    call_converted: int
    call_direct: int

    def __post_init__(self) -> None:
        if self.width_cap < 8 or self.width_cap % 8:
            raise ValueError(f"a memory port of {self.width_cap} bits is no whole bytes")
        if self.frames_per_call < 1:
            raise ValueError(f"{self.frames_per_call} frames a call is none")
        if self.call_converted < 0 or self.call_direct < 0:
            raise ValueError("a call adds no fewer than zero cycles")


def iodma_hls(width_cap: int, *, frames_per_call: int = 1) -> EndOffer:
    """The ``IODMA_hls`` end, its memory port at most ``width_cap`` bits, with its call
    constants as measured: 4 cycles a call with a width converter, 13 without."""
    return EndOffer(
        kind=IODMA_HLS,
        width_cap=width_cap,
        frames_per_call=frames_per_call,
        call_converted=4,
        call_direct=13,
    )


@dataclass(frozen=True, kw_only=True)
class EndContract:
    """An end's facts, as the shell integrates it and an exploration costs it:

    - ``kind``; ``direction``: ``in`` (memory to the root's input) or ``out``;
    - the stream side, the free side's contract: ``port`` (its AXIS port), ``tdata``
      (bits), ``beats`` a frame, ``lanes`` a beat and ``element`` (the channel's);
    - the memory side: ``memory_width`` (bits), ``words`` a frame, and whether a width
      ``converter`` sits between the two;
    - the rate: ``call_cycles`` (the constant each call adds), ``frames_per_call`` and
      ``cycles`` a frame, the memory latency excluded (``MEMORY_LATENCY``);
    - ``control_buses``: the AXI-Lite buses it presents to the shell's processor."""

    kind: str
    direction: str
    port: str
    tdata: int
    beats: int
    lanes: int
    element: ScalarEncoding
    memory_width: int
    words: int
    converter: bool
    call_cycles: int
    frames_per_call: int
    control_buses: int

    def cycles_at(self, frames_per_call: int) -> int:
        """Its cycles a frame were each call to move ``frames_per_call`` frames."""
        return max(self.beats, self.words) + ceil(self.call_cycles / frames_per_call)

    @property
    def cycles(self) -> int:
        """Its cycles a frame at its offer's frames a call."""
        return self.cycles_at(self.frames_per_call)

    @property
    def times(self) -> tuple[int, ...]:
        """Each stream beat's cycle from the frame's first: one a cycle, or once the
        memory word that completes it arrived (a word a cycle)."""
        beats, words = self.beats, self.words
        return tuple(max(beat, ceil((beat + 1) * words / beats) - 1) for beat in range(beats))


class IodmaEnd(Space):
    """The ``IODMA_hls`` end on a free side (``side``), where ``offers`` offer it."""

    offers: tuple[EndOffer, ...] = Param()
    side: StreamContract = Param()

    @constraint
    def offered(self) -> bool | Rejected:
        """The shell offers this kind."""
        if not any(offer.kind == IODMA_HLS for offer in self.offers):
            return reject("end-not-offered", f"the shell offers no {IODMA_HLS} end here")
        return True

    # The candidate's refusal, which the end Decision's viability reads.
    admission = ConstraintGroup(offered)

    @derived
    def contract(self) -> EndContract:
        """Its facts on ``side``, sized by its offer (see the module docstring)."""
        (offer,) = (offer for offer in self.offers if offer.kind == IODMA_HLS)
        side = self.side
        tdata, beats = side.transport.data_width, side.form.beats
        memory = gcd(tdata * beats, offer.width_cap)
        converter = memory != tdata
        return EndContract(
            kind=IODMA_HLS,
            # Seen from inside the root, its input is the AXIS target.
            direction="in" if side.transport.endpoint is Endpoint.TARGET else "out",
            port=side.transport.name,
            tdata=tdata,
            beats=beats,
            lanes=side.lanes,
            element=side.element,
            memory_width=memory,
            words=tdata * beats // memory,
            converter=converter,
            call_cycles=offer.call_converted if converter else offer.call_direct,
            frames_per_call=offer.frames_per_call,
            control_buses=1,
        )


ENDS: dict[str, type[Space] | Space] = {IODMA_HLS: IodmaEnd}
"""The ends a shell can offer a free side, by kind."""


__all__ = [
    "ENDS",
    "END_CONTROL",
    "IODMA_HLS",
    "MEMORY_LATENCY",
    "EndContract",
    "EndOffer",
    "IodmaEnd",
    "iodma_hls",
]
