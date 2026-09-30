# Authoring a kernel: from an RTL datasheet to a model

A kernel binds one RTL module to the model. The author states the module's
hardware facts once; the model derives everything else from them, and the
conformance harness checks the one fact Python cannot see: that the RTL really
walks the order the kernel declares.

This guide integrates a new module, `accpool_axi` (global sum pooling, FINN's
GlobalAccPool), as a worked example. The module is hypothetical: no RTL exists,
so the model runs here and its conformance test is shown but not run.

## The datasheet, and what each fact becomes

`accpool_axi` computes `Y[b, c] = sum over s of X[b, s, c]` (`b` images, `s`
pixels, `c` channels). Its datasheet says:

| Datasheet fact | In the kernel |
|---|---|
| module `accpool_axi`, from FinnLib `rtl/pool/accpool_axi.sv` | `module`, `sources()` |
| generics `CHANNELS`, `PIXELS`, `PE`, `IN_WIDTH`, `OUT_WIDTH`, `SIGNED` | `parameters()`, from extents, folding factors and elements |
| `PE` channels a beat, PE dividing the channels | a folding factor: `pe = Decision(domain=divisors_of(channels))` |
| images, then pixels, then channel folds (channels innermost) | the loop nest: `bound_schedule(beats=(b, s, c), ...)` |
| `s_axis_input`: `X[b, s, c]`, PE channels a beat, no TLAST | a target `AxiStreamPort`: `index=(b, s, c)`, `lanes=(c,)` |
| `m_axis_output`: `Y[b, c]`, PE channels a beat, after the last pixel of an image | an initiator `AxiStreamPort`: `index=(b, c)`, `lanes=(c,)`, `reduces=(s,)` |
| the sum's width: the input's range times the pixels, signed | the producer's `dtype`, stated by the kernel |
| accumulators are at most 32 bits | `admission`: only what the RTL itself cannot build |

What the author does not write: extent getters (the ports' tensors bind them),
folding factor domains beyond the divisor rule, traversals (each port's is the
schedule's projection), idle widths, shape checks (ports that disagree on an
extent are refused), or `semantics=` on ordinary values.

## The checklist

1. **Indices.** Name one `Index` per dimension of the operation. An index a
   port reads alone takes its extent from that tensor axis; an axis read by an
   expression (a sliding window, `oh * 2 + kh`) binds nothing and is checked.
2. **Folding factors.** Each is a `Decision` on an index (`pe` on `c`), the
   lanes it spreads over each beat; its domain is the divisors of that index's
   extent, read from a named `extent_of` member. A parent may pin or narrow it
   by key (`pool.pe`).
3. **The loop nest.** `bound_schedule(beats, factors)`, outer to inner, exactly
   as the RTL walks it. This is the one order the model cannot check; the
   conformance harness does, in simulation.
4. **Each interface.** One `AxiStreamPort`: the indices it reads (`index`), its
   lane order (`lanes`, outer first: lane zero is the innermost), what it is
   presented after (`reduces`) or before (`holds`), and the reduction its TLAST
   closes (`closes`). A traversal no schedule derives is given as `sequence=`.
5. **Elements.** A producer states its `dtype` from the kernel's facts, choices
   and input elements, never from its own output stream; a consumer `admits`
   an integer policy. The stream refuses an element other than its own.
6. **Admission.** Only the RTL's own limits.

## The kernel

```python
from collections.abc import Mapping

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
from finn.dataflow.datatypes import QONNXDataType, ordinary_integer_bounds
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype_named
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import Kernel, extent_of
from finn.kernels.configure import commit
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.port import AxiStreamPort
from finn.kernels.streams import Stream as KernelStream

b, s, c = Index("b"), Index("s"), Index("c")


class AccPoolKernel(Kernel):
    """accpool_axi: Y[b, c] = sum over s of X[b, s, c], PE channels a beat."""

    id, version, module = "example.accpool_axi", "1", "accpool_axi"

    # The streams it sits on; their tensors are the use case's facts.
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    # Extents bound from the tensors the ports read.
    channels = extent_of(c)
    pixels = extent_of(s)

    pe: int = Decision(domain=divisors_of(channels))

    @derived
    def schedule(self) -> Schedule | Rejected:
        """The RTL's loop nest: images, then pixels, then channel folds."""
        return self.bound_schedule(beats=(b, s, c), factors={c: self.pe})

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def sum_dtype(self) -> QONNXDataType:
        """What the RTL emits: wide enough for PIXELS sums of the input's range, signed."""
        low, high = ordinary_integer_bounds(self.x.element.dtype)
        need = max(abs(low), high) * self.pixels
        return dtype_named(f"INT{need.bit_length() + 1}")

    x = AxiStreamPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        stream=x_stream,
        schedule=schedule,
        index=(b, s, c),
        lanes=(c,),
        admits=Integer(max_bits=16),
    )
    y = AxiStreamPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        stream=y_stream,
        schedule=schedule,
        index=(b, c),
        lanes=(c,),
        reduces=(s,),
        dtype=sum_dtype,
    )

    @constraint
    def accumulator_supported(self) -> bool | Rejected:
        if self.y.element.bits > 32:
            return reject("accpool-width", "the RTL's accumulators are at most 32 bits")
        return True

    admission = ConstraintGroup(accumulator_supported)

    def parameters(self) -> Mapping[str, int | str]:
        x, y = self.x.element, self.y.element
        return {
            "CHANNELS": self.channels,
            "PIXELS": self.pixels,
            "PE": self.pe,
            "IN_WIDTH": x.bits,
            "OUT_WIDTH": y.bits,
            "SIGNED": int(x.signed),
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (
            CopiedSource("finnlib", "rtl/pool/accpool_axi.sv", provides=("module:accpool_axi",)),
        )
```

## Placing it, and what the model derives

Place it between two streams and commit its folding factor. Each port presents the
schedule's projection: the input `PIXELS * CHANNELS / PE` beats of PE lanes an
image, the output `CHANNELS / PE` beats after the pixels.

```python
INT4 = DataType["INT4"]


def placed(x_shape, y_shape=(2, 8), y_dtype="INT7"):
    class Placed(Space):
        x = KernelStream(tensor=Tensor(x_shape, ScalarEncoding(INT4)), port="in0_V")
        y = KernelStream(tensor=Tensor(y_shape, ScalarEncoding(DataType[y_dtype])), port="out0_V")
        pool = AccPoolKernel(x_stream=x, y_stream=y)

    return design_space(Placed())


point = commit(placed((2, 4, 8)), {"pool.pe": 4})
assert point.pool.extents == {b: 2, s: 4, c: 8}
x, y = point.pool.x.presented.form, point.pool.y.presented.form
assert (x.beats, x.lanes) == (2 * 4 * 8 // 4, 4)
assert (y.beats, y.lanes) == (2 * 8 // 4, 4)
assert dict(point.pool.build_requirements.parameters) == {
    "CHANNELS": 8,
    "PIXELS": 4,
    "PE": 4,
    "IN_WIDTH": 4,
    "OUT_WIDTH": 7,
    "SIGNED": 1,
}
```

The folding factor's domain is the divisors of the bound channel count, and a parent may
pin it by key (`pool.pe = 4` in its body). Ports that disagree on an extent
are refused, naming the ports and axes, rather than leaving part of a tensor
unread:

```python
assert placed((2, 4, 8)).pool.field(AccPoolKernel.pe).candidates().value == (1, 2, 4, 8)
refused = placed((2, 4, 6)).pool.query(AccPoolKernel.extents)
assert [(f.code, f.message) for f in refused.findings] == [
    ("kernel-extents", "c is 6 (x axis 2) and 8 (y axis 1)")
]
```

The producer states its element from facts and inputs only, so it is known
with the output unplaced (a compiler infers output types node by node), and a
stream of another element refuses it:

```python
class Unplaced(Space):
    x = KernelStream(tensor=Tensor((2, 4, 8), ScalarEncoding(INT4)), port="in0_V")
    pool = AccPoolKernel(x_stream=x)


assert design_space(Unplaced()).pool.y.element == ScalarEncoding(DataType["INT7"])
wrong = commit(placed((2, 4, 8), y_dtype="INT8"), {"pool.pe": 4})
assert "stream-tensor" in {f.code for f in wrong.y.query(KernelStream.connection).findings}
```

## The loop order is a choice, with a cost

The RTL's order decides what the stream in front must do. With channels
innermost, a producer presenting the input row-major at the same PE connects
directly, and at another PE takes a width conversion (`vpc`). Had the RTL
walked channel folds outer and pixels inner (`beats=(b, c, s)`), the same
producer would need a reorder buffer (`input_gen`) on the stream in front:
the order the kernel declares is the adapter every producer pays for. Keep
accumulators in the kernel (channels innermost, one accumulator per channel)
or pay a reorder buffer on the stream; the model prices both, the author
chooses.

```python
from finn.dataflow.plan import plan
from finn.dataflow.traversal import BeatSequence, vector_major

wanted = point.pool.x.presented
for lanes, steps in ((4, ()), (2, ("width_conversion",))):
    produced = BeatSequence(vector_major((2, 4, 8), lanes))
    assert tuple(step.value for step in plan(produced, wanted).steps) == steps

channels_outer = Schedule({b: 2, s: 4, c: 8}, factors={c: 4}, beats=(b, c, s))
reordered = BeatSequence(channels_outer.present((2, 4, 8), (b, s, c), lanes=(c,)))
produced = BeatSequence(vector_major((2, 4, 8), 4))
assert tuple(step.value for step in plan(produced, reordered).steps) == ("reorder",)
```

## The conformance test

`tests/kernels/conformance.py` checks a kernel against its RTL over sampled
folding factors (the smallest, an interior, the largest, and one fed through a width
converter): its pins and parameter names against the sources, the model's
consistency (coverage, beat counts, boundary presentation, every output
element stated with the outputs unplaced), and, in XSim, stalled and free, its
outputs against a reference. A declared order the RTL does not walk passes
every Python check and fails in simulation, which is the harness's reason to
exist.

```python
from kernels.conformance import SAMPLED, conformance


def test_accpool_conforms(tmp_path):
    conformance(
        AccPoolKernel,
        inputs={"x_stream": Tensor((2, 16, 8), ScalarEncoding(INT4))},
        outputs={"y_stream": (2, 8)},  # its element is the kernel's sum_dtype
        reference=lambda x_stream: {"y_stream": x_stream.sum(axis=1)},
        factors=SAMPLED,
        xsim=tmp_path,
    )
```

The case lives in `tests/kernels/test_conformance.py` with the other kernels'
(`CASES`). Every kernel in this package has one.
