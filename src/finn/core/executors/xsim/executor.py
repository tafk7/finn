# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim executor: a partition of KernelOps run as its hardware, in XSim.

``XSim`` claims a partition node (StreamingDataflowPartition) whose body is a partition
of KernelOps (``kernel_partition_body``), and runs it as packaging builds it:

1. **the point**: the body's shell root, its nodes' and its tensors' channel choices
   replayed and the rest
   completed by ``completion`` (``Baseline()`` by default), and its boundary
   (``configured_root``): each graph input and output at its port;
2. **the words**: each input tensor, as integers of its boundary element (a value
   that is no integer, or one the element does not hold, is refused: ``Unstreamable``),
   presented as the port presents it, one word a beat (``finn.dataflow.traversal.pack``);
3. **the simulation**: the root's module in XSim (``rtl.stream_out``), paced by
   ``pacing``, every input streamed and each output's words taken as they arrive,
   none expected;
4. **the tensors**: each output's words read back as the port presents them
   (``finn.dataflow.traversal.unpack``), sign-extended by its element, into the
   context in the container the context holds the tensor in.

It only executes: it compares nothing. Its outputs are exactly the words that arrived,
as a tensor: ``pack`` of an output gives back its words, and words that are no tensor
(an element presented twice with two values) are refused with the words
(``Unreadable``), so a caller that compares (with a ``Python()`` run) compares the
hardware's words.

Its options are the executor's, never the graph's: ``pacing`` (``STALLED`` by default),
``completion``, ``toolchain`` (the machine's by default), ``directory`` (where each run
writes its testbench and simulates, a directory of its own for each partition node it
runs: ``<directory>/<node>``, ``<node>.1``, ...; a fresh build directory by default)
and ``cache``, the HLS cache ``materialize`` takes an HLS leaf's product from. Nothing
compiled is pointed to by a node attribute. Each run simulates in the simulator's own
processes (xvlog, xelab, xsim), so runs in one process do not share simulator state.

**Links**, when the run returns its full context (``finn.core.onnx_exec.running``):
each tensor between two kernels of the body is tapped in the simulation, at its
producer's output, before any adapter (``rtl.Tap``), and read back with the producer's
traversal and element: the ONNX tensor itself, as its producer presents it. It enters
the context as the partition node's Python run names a body's tensor,
``<node>_<tensor>``, in the container the body's context holds it in, so a hardware run
and a Python run compare entry by entry. A run that does not return its full context
taps nothing.

It is ``hardware``: a run that requires hardware accepts it. It runs a partition whole,
from its inputs: a run that starts inside the body is refused before it runs
(``finn.core.onnx_exec.InsidePartition``). A run that ends inside it (at ``end_node``,
a node of the body) simulates the whole partition and observes: of the body's tensors,
it writes those its nodes up to ``end_node`` produce (the outputs among them, and the
links, when the full context is returned), as Python's run up to that node would.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from finn.core.executors.xsim.pacing import STALLED, Pacing
from finn.core.executors.xsim.rtl import Receive, SimulationFailed, Tap, Words, stream_out
from finn.core.onnx_exec import running, window
from finn.core.space import Available
from finn.custom_op.kernels.shell import configured_root, member
from finn.custom_op.partition.kernel_partitions import kernel_partition_body
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.traversal import Traversal, pack, unpack
from finn.transformation.kernels.package import free_side
from finn.util.basic import make_build_dir

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.core.executors.base import Context
    from finn.kernels.explore import Completion
    from finn.util.toolchain import Toolchain


class Unstreamable(ValueError):
    """A partition input holds a value its boundary element cannot present: no integer,
    or out of the element's range."""


class Unreadable(SimulationFailed):
    """An output's words are no tensor of its port's traversal (an element presented
    twice with two values), or not one its container in the context holds exactly;
    ``received`` holds each output's words as they arrived."""

    def __init__(self, message: str, received: Mapping[str, tuple[int, ...]]) -> None:
        super().__init__(message)
        self.received = dict(received)


@dataclass(frozen=True)
class Stream:
    """A boundary port of a partition: its tensor in the parent's context and in the
    body (``inner``), whether it is an output, the traversal the port presents, and its
    element's payload bits and range."""

    tensor: str
    inner: str
    port: str
    output: bool
    form: Traversal
    bits: int
    low: int
    high: int

    @property
    def width(self) -> int:
        """A word's payload bits: every lane's element."""
        return self.bits * self.form.lanes


@dataclass(frozen=True)
class XSim:
    """Runs a partition of KernelOps as its hardware, in XSim (module docstring)."""

    pacing: Pacing = STALLED
    completion: Completion | None = None
    toolchain: Toolchain | None = None
    directory: Path | None = None
    cache: Path | None = None

    hardware: ClassVar[bool] = True

    def claims(self, node: NodeProto, model: ModelWrapper) -> bool:
        return kernel_partition_body(node) is not None

    def run(self, node: NodeProto, context: Context, model: ModelWrapper) -> None:
        body = kernel_partition_body(node)
        assert body is not None, f"{node.name}: XSim runs a partition of KernelOps"
        run = running()
        reached = {tensor for inner in window(body, None, run.end_node) for tensor in inner.output}
        point, streams = boundary(body, node, self.completion)
        tapped = links(body, point, reached) if run.full_context else ()
        cycles = point.query(type(point).cycles)
        inputs: dict[str, Words] = {}
        outputs: dict[str, Receive] = {}
        for stream in streams:
            if stream.output:
                outputs[stream.port] = Receive(stream.form.beats, stream.width)
            else:
                values = _integers(context[stream.tensor], stream, node.name)
                inputs[stream.port] = (pack(stream.form, values.ravel(), stream.bits), stream.width)
        received = stream_out(
            point.module,
            self._run_directory(node.name),
            inputs=inputs,
            outputs=outputs,
            pacing=self.pacing,
            cycles=cycles.value if isinstance(cycles, Available) else 0,
            toolchain=self.toolchain,
            cache=self.cache,
            taps={link.name: link.tap for link in tapped},
        )
        for stream in streams:
            if stream.output and stream.inner in reached:
                where = f"{node.name}: {stream.tensor} ({stream.port})"
                context[stream.tensor] = _read(
                    stream.form,
                    stream.bits,
                    stream.low < 0,
                    context[stream.tensor],
                    received[stream.port],
                    where,
                    received,
                )
        held = body.make_empty_exec_context() if tapped else {}
        for link in tapped:
            where = f"{node.name}: link {link.tensor} ({link.tap.end.instance})"
            context[f"{node.name}_{link.tensor}"] = _read(
                link.form,
                link.bits,
                link.signed,
                held[link.tensor],
                received[link.name],
                where,
                received,
            )

    def _run_directory(self, name: str) -> Path:
        """A directory of its own for one run of the partition node ``name``."""
        if self.directory is None:
            return Path(make_build_dir(f"xsim_{name}_"))  # type: ignore[no-untyped-call]
        found = self.directory / name
        index = 0
        while found.exists():
            index += 1
            found = self.directory / f"{name}.{index}"
        found.mkdir(parents=True)
        return found


@dataclass(frozen=True)
class Tapped:
    """A link of a partition's body: its tensor between two kernels, the tap at its
    producer's output (``name``, the testbench's), and the traversal its producer
    presents it in, its element's bits and whether they are signed."""

    tensor: str
    name: str
    tap: Tap
    form: Traversal
    bits: int
    signed: bool


def links(body: ModelWrapper, point: Any, tensors: Collection[str]) -> tuple[Tapped, ...]:
    """The links of ``body`` among ``tensors``, in the body's node order: each tensor a
    node of the body produces and another consumes, tapped at its producer's end of its
    channel in ``point``, the body's configured root (``boundary``): its first hop's
    source, before any adapter."""
    consumed = {tensor for inner in body.graph.node for tensor in inner.input}
    found: list[Tapped] = []
    for inner in body.graph.node:
        for tensor in inner.output:
            if tensor not in tensors or tensor not in consumed:
                continue
            channel = getattr(point, member(tensor))
            produced = channel.endpoints.source
            end = channel.hops[0].source.under(member(tensor))
            form, bits = produced.form, produced.element.bits
            low, _ = ordinary_integer_bounds(produced.element.dtype)
            tap = Tap(end, form.beats, form.lanes * bits)
            found.append(Tapped(tensor, f"link_{len(found)}", tap, form, bits, low < 0))
    return tuple(found)


def _read(
    form: Traversal,
    bits: int,
    signed: bool,
    held: Any,
    words: tuple[int, ...],
    where: str,
    received: Mapping[str, tuple[int, ...]],
) -> npt.NDArray[Any]:
    """The tensor ``words`` present in ``form``, in the shape and container of ``held``,
    the context's: refused (``Unreadable``) when they are no tensor of ``form`` or the
    container does not hold their values exactly."""
    try:
        values = unpack(form, words, bits, signed=signed)
    except ValueError as error:
        raise Unreadable(f"{where}: {error}", received) from None
    found = values.reshape(np.shape(held)).astype(np.asarray(held).dtype)
    if not np.array_equal(found, values.reshape(found.shape)):
        raise Unreadable(
            f"{where}: its container ({found.dtype}) does not hold the values presented exactly",
            received,
        )
    return found


def boundary(
    body: ModelWrapper, node: NodeProto, completion: Completion | None = None
) -> tuple[Any, tuple[Stream, ...]]:
    """The body's configured root point (``configured_root``) and its boundary ports,
    each with the tensor it carries in the parent graph: the partition node's inputs and
    outputs, by position, are its body's graph inputs and outputs."""
    point, ports = configured_root(body, node.name, completion)
    # As the partition op maps them: its i-th input is its body's i-th graph input.
    outer = {body.graph.input[index].name: name for index, name in enumerate(node.input)}
    outputs = {item.name for item in body.graph.output}
    outer |= dict(zip((item.name for item in body.graph.output), node.output, strict=True))
    streams = []
    for tensor, port in ports:
        end = free_side(point, tensor)
        low, high = ordinary_integer_bounds(end.element.dtype)
        streams.append(
            Stream(
                outer[tensor],
                tensor,
                port,
                tensor in outputs,
                end.form,
                end.element.bits,
                low,
                high,
            )
        )
    return point, tuple(streams)


def _integers(values: Any, stream: Stream, name: str) -> npt.NDArray[np.int64]:
    """``values`` as the integers the port presents, refused when one is no integer or
    out of the element's range."""
    found = np.asarray(values)
    if found.dtype.kind == "f" and not np.array_equal(np.rint(found), found):
        raise Unstreamable(f"{name}: {stream.tensor} holds values that are not integers")
    integers = found.astype(np.int64)
    if integers.size and not stream.low <= integers.min() <= integers.max() <= stream.high:
        raise Unstreamable(
            f"{name}: {stream.tensor}'s values span [{integers.min()}, {integers.max()}], "
            f"which its port {stream.port} (integers in [{stream.low}, {stream.high}]) "
            "does not present"
        )
    return integers


__all__ = ["Unreadable", "Unstreamable", "XSim"]
