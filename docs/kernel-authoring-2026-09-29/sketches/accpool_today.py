"""Worked example: integrating a new RTL module as a Kernel.

The RTL (hypothetical) is ``accpool_axi``: global sum pooling, FINN's GlobalAccPool.

    Y[b, c] = sum over s of X[b, s, c]        b: images, s: pixels, c: channels

Its datasheet says:
  - s_axis_input:  PE channels a beat, channels innermost, pixels in order, image after image;
                   no TLAST (it counts PIXELS itself).
  - m_axis_output: PE channels a beat, emitted after the last pixel of each image.
  - parameters:    CHANNELS, PIXELS, PE, IN_WIDTH, OUT_WIDTH, SIGNED.
  - it keeps one accumulator per channel (CHANNELS of them, PE updated per beat).

Run from the worktree root:
    PYTHONPATH=src:tests python docs/kernel-authoring-2026-09-29/sketches/accpool_today.py
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, log2

from qonnx.core.datatype import DataType

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    design_space,
    divisors_of,
    reject,
)
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.schedule import SCHEDULE, Index, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.dataflow.plan import plan
from finn.kernels.adapters import realize
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.datatypes.domains import Integer
from finn.kernels.port import ScheduledPort
from finn.kernels.streams import Stream as KernelStream

# Step 1: the operation's indices.
b, s, c = Index("b"), Index("s"), Index("c")


class AccPoolKernel(Kernel):
    """accpool_axi: Y[b, c] = sum_s X[b, s, c], PE channels a beat."""

    id = "example.accpool_axi"
    version = "1"
    module = "accpool_axi"

    # Step 2: the streams it sits on; their tensors are the use case's facts.
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    @derived
    def images(self) -> int:
        return self.y_stream.tensor.shape[0]

    @derived
    def channels(self) -> int:
        return self.y_stream.tensor.shape[1]

    @derived
    def pixels(self) -> int:
        return self.x_stream.tensor.shape[1]

    # Step 3: the folding factor is a Decision on an index, its domain the divisors.
    pe: int = Decision(domain=divisors_of(channels))

    # Step 4: the RTL's loop nest, written once: images, then pixels, then channel folds.
    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        return Schedule(
            {b: self.images, s: self.pixels, c: self.channels},
            folds={c: self.pe},
            beats=(b, s, c),
        )

    # Step 5: one port per interface: what it reads, its lane order, what it reduces.
    x = ScheduledPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        stream=x_stream,
        admits=Integer(),
        schedule=schedule,
        index=(b, s, c),
        lanes=(c,),
    )
    y = ScheduledPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        stream=y_stream,
        admits=Integer(signed=True),
        schedule=schedule,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
    )

    # Step 6: admission — what the RTL cannot build.
    @constraint
    def shapes_agree(self) -> bool | Rejected:
        """X is (B, S, C) for the B and C of Y (the model does not check this for you)."""
        want = (self.images, self.pixels, self.channels)
        if self.x_stream.tensor.shape != want:
            return reject("accpool-shape", f"x must be {want}, is {self.x_stream.tensor.shape}")
        return True

    @constraint
    def accumulator_fits(self) -> bool | Rejected:
        low, high = ordinary_integer_bounds(self.x.element.dtype)
        need = max(abs(low), high) * self.pixels
        bits = self.y.element.bits
        if need >= 1 << (bits - 1):
            return reject("accpool-width", f"{self.pixels} sums need more than {bits} bits")
        return True

    admission = ConstraintGroup(shapes_agree, accumulator_fits)

    # Step 7: the RTL's generics, from the choices; its sources.
    def parameters(self) -> Mapping[str, int | str]:
        return {
            "CHANNELS": self.channels,
            "PIXELS": self.pixels,
            "PE": self.pe,
            "IN_WIDTH": self.x.element.bits,
            "OUT_WIDTH": self.y.element.bits,
            "SIGNED": int(self.x.element.signed),
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("finnlib", "rtl/pool/accpool_axi.sv", provides=("module:accpool_axi",)),)


def show(label, sequence):
    form = sequence.form
    beats = list(form.positions())
    print(f"  {label:<10} {form.beats:>3} beats x {form.lanes} lanes  loops {[(l.extent, l.stride) for l in form.beat_loops]}"
          f"  first beats {[tuple(map(tuple, beat)) for beat in beats[:3]]}")


if __name__ == "__main__":
    B, S, C = 1, 4, 8
    INT4, INT8 = DataType["INT4"], DataType["INT8"]

    # Step 8: place it between two boundary streams and read what it presents.
    class Placed(Space):
        x = KernelStream(tensor=Tensor((B, S, C), ScalarEncoding(INT4)), port="in0_V")
        y = KernelStream(tensor=Tensor((B, C), ScalarEncoding(INT8)), port="out0_V")
        pool = AccPoolKernel(x_stream=x, y_stream=y)

    point = commit(design_space(Placed()), {"pool.pe": 4})
    print("schedule:", [(i.name, point.pool.schedule.extent(i), point.pool.schedule.fold(i)) for i in point.pool.schedule.beats])
    show("x port", point.pool.x.sequence)
    show("y port", point.pool.y.sequence)
    print("  x markers:", point.pool.x.sequence.markers, " y markers:", point.pool.y.sequence.markers)
    print("parameters:", point.pool.parameters())
    req = point.pool.query(AccPoolKernel.build_requirements)
    print("build_requirements:", type(req).__name__)

    # Step 9: what an upstream producer at another folding costs, decided by the stream.
    need = point.pool.x.sequence
    for lanes in (4, 2, 8):
        producer = BeatSequence(vector_major((B, S, C), lanes))
        p = plan(producer, need)
        print(f"producer vector_major PE={lanes}: plan = {p.describe() or 'direct'}",
              [(st.kind, st.module) for st in realize(p)])

    # The alternative RTL loop order: channel folds outer, pixels inner.
    alt = Schedule({b: B, s: S, c: C}, folds={c: 4}, beats=(b, c, s))
    x_alt = BeatSequence(alt.present((B, S, C), (b, s, c), lanes=(c,)))
    p = plan(BeatSequence(vector_major((B, S, C), 4)), x_alt)
    print("alt order (b, c, s): plan =", p.describe(), [(st.kind, st.module) for st in realize(p)],
          "| closing(s):", alt.closing((s,)))

    # Admission at work: a too-narrow result type.
    class Narrow(Space):
        x = KernelStream(tensor=Tensor((B, S, C), ScalarEncoding(INT4)), port="in0_V")
        y = KernelStream(tensor=Tensor((B, C), ScalarEncoding(DataType["INT4"])), port="out0_V")
        pool = AccPoolKernel(x_stream=x, y_stream=y)

    narrow = commit(design_space(Narrow()), {"pool.pe": 4})
    print("narrow result:", narrow.pool.query(AccPoolKernel.build_requirements))
