# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cycles measured in simulation: each observed stream's handshakes by cycle, in frames.

``kernels.xsim.measure`` streams identical frames back to back, never stalled,
and records the cycle each handshake completes in, on the root's streams and on
each link between the module's instances. Every frame presents the same beats,
so a stream's handshakes split evenly into frames. From them, with every count
inclusive of both ends (one beat in and out in the same cycle spans one cycle):

- a frame's **latency**: from its first input beat accepted to its last output
  beat, over the root's streams, or between any two observed streams (``span``);
- the **interval** between frames: their last beats apart on a stream, the root's
  over its outputs; the last interval is the steady state's (II);
- the **total**: from the first input beat to the last output beat of all frames;
- a stream's **busy** span in a frame: its first to its last beat.

Pure arithmetic on the recorded cycles: the fast gate tests it without Vivado.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Frame = tuple[int, ...]
"""The cycles of one frame's beats on one stream, in order."""


@dataclass(frozen=True)
class Measured:
    """``frames`` frames' handshakes, by stream name: the root's ``inputs`` and
    ``outputs`` by port, each link by its sink (``instance.pin``)."""

    frames: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    beats: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        for name in (*self.inputs, *self.outputs):
            if name not in self.beats:
                raise ValueError(f"no beats recorded on the root's stream {name}")
        for name in self.beats:
            self.of(name)  # refuses uneven frames now

    def of(self, name: str) -> tuple[Frame, ...]:
        """``name``'s beats, frame by frame."""
        cycles = self.beats[name]
        if not cycles or len(cycles) % self.frames:
            raise ValueError(f"{name}: {len(cycles)} beats do not split into {self.frames} frames")
        count = len(cycles) // self.frames
        return tuple(tuple(cycles[f * count : (f + 1) * count]) for f in range(self.frames))

    def per_frame(self, name: str) -> int:
        """The beats a frame presents on ``name``."""
        return len(self.beats[name]) // self.frames

    def busy(self, name: str) -> tuple[int, ...]:
        """Each frame's first to last beat on ``name``."""
        return tuple(frame[-1] - frame[0] + 1 for frame in self.of(name))

    def _first(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(min(self.of(name)[f][0] for name in names) for f in range(self.frames))

    def _last(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(max(self.of(name)[f][-1] for name in names) for f in range(self.frames))

    def span(self, source: str, sink: str) -> tuple[int, ...]:
        """Each frame's first beat on ``source`` to its last beat on ``sink``."""
        return tuple(
            last - first + 1
            for first, last in zip(self._first([source]), self._last([sink]), strict=True)
        )

    @property
    def latencies(self) -> tuple[int, ...]:
        """Each frame's first input beat to its last output beat."""
        return tuple(
            last - first + 1
            for first, last in zip(self._first(self.inputs), self._last(self.outputs), strict=True)
        )

    def intervals(self, name: str | None = None) -> tuple[int, ...]:
        """Successive frames' last beats apart on ``name``; on the root's outputs by default."""
        ends = self._last(self.outputs if name is None else [name])
        return tuple(later - earlier for earlier, later in zip(ends, ends[1:]))

    def interval(self, name: str | None = None) -> int:
        """The steady state's interval (the last), on ``name`` or the root's outputs."""
        if self.frames < 2:
            raise ValueError("an interval needs at least two frames")
        return self.intervals(name)[-1]

    @property
    def total(self) -> int:
        """The first input beat to the last output beat of the last frame."""
        return self._last(self.outputs)[-1] - self._first(self.inputs)[0] + 1

    def table(self) -> str:
        """The measurement as text: the root's latency, interval and total, then per
        stream its beats per frame, busy span, interval and first beat."""
        start = self._first(self.inputs)[0]
        lines = [
            f"frames {self.frames}; latency per frame {list(self.latencies)}; "
            f"intervals {list(self.intervals()) if self.frames > 1 else []}; "
            f"total {self.total}",
            f"{'stream':<56} {'beats':>6} {'busy':>6} {'II':>6} {'first':>6}",
        ]
        for name in self.beats:
            interval = self.interval(name) if self.frames > 1 else "-"
            lines.append(
                f"{name:<56} {self.per_frame(name):>6} {self.busy(name)[-1]:>6} "
                f"{interval:>6} {self.of(name)[0][0] - start:>6}"
            )
        return "\n".join(lines)
