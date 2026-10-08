# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How a simulation paces its streams: one spec for every RTL driver.

A ``Pace`` is one stream's driver: after every ``burst`` handshakes it idles
``pause`` cycles, its valid low on an input, its ready low on an output; with
``pause`` 0 it never idles. A ``Pacing`` gives a run's streams their paces by
position: the i-th input in the order the run names them takes ``inputs[i]``,
cycling, and the outputs likewise, so streams side by side arrive out of step.

Both drivers realize it the same way, counting handshakes, not cycles: the
stream testbench (``finn.harness.rtl``) and the XSI numeric transport
(``tests/kernels/sweeps/rtl_transport.py``). ``FREE`` never stalls; ``STALLED``
stalls both sides, each input in bursts of two or three, each output after
every beat for five cycles.

A module of its own, apart from the testbench writer: the XSI path keys its
runs by the harness files it imports (``scripts/emitted_text.py``), and needs
only this.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pace:
    """After every ``burst`` handshakes, ``pause`` idle cycles."""

    burst: int = 1
    pause: int = 0

    def __post_init__(self) -> None:
        if type(self.burst) is not int or self.burst < 1:
            raise ValueError(f"a burst is at least one handshake, not {self.burst!r}")
        if type(self.pause) is not int or self.pause < 0:
            raise ValueError(f"a pause is at least zero cycles, not {self.pause!r}")

    def cycles(self, beats: int) -> int:
        """The cycles this driver takes for ``beats`` handshakes when the other side never
        stalls: each beat a cycle, and a pause after each full burst."""
        return beats + self.pause * (beats // self.burst)


@dataclass(frozen=True)
class Pacing:
    """Each input's and output's ``Pace``, by position, cycling."""

    inputs: tuple[Pace, ...] = (Pace(),)
    outputs: tuple[Pace, ...] = (Pace(),)

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError("a pacing gives at least one input and one output pace")

    def input(self, index: int) -> Pace:
        return self.inputs[index % len(self.inputs)]

    def output(self, index: int) -> Pace:
        return self.outputs[index % len(self.outputs)]

    @property
    def stalls(self) -> bool:
        """Whether any stream idles."""
        return any(pace.pause for pace in (*self.inputs, *self.outputs))

    def as_json(self) -> dict[str, list[list[int]]]:
        """The pacing as plain data, for a request to another process."""
        return {
            "inputs": [[pace.burst, pace.pause] for pace in self.inputs],
            "outputs": [[pace.burst, pace.pause] for pace in self.outputs],
        }

    @classmethod
    def from_json(cls, data: dict[str, list[list[int]]]) -> Pacing:
        return cls(
            tuple(Pace(burst, pause) for burst, pause in data["inputs"]),
            tuple(Pace(burst, pause) for burst, pause in data["outputs"]),
        )


FREE = Pacing()
STALLED = Pacing(inputs=(Pace(2, 1), Pace(3, 1), Pace(2, 3)), outputs=(Pace(1, 5),))


__all__ = ["FREE", "STALLED", "Pace", "Pacing"]
