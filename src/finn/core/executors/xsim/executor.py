# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim executor: a partition of KernelOps run as its hardware, in XSim.

``XSim`` claims a partition node (StreamingDataflowPartition) whose body is a partition
of KernelOps (``kernel_partition_body``), and runs it as packaging builds it:

1. **the point**: the body's shell root, its nodes' choices replayed and the rest
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

It is ``hardware``: a run that requires hardware accepts it. It runs a partition whole:
the tensors inside the body (the links between its kernels) are not observed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from finn.core.executors.base import is_partition
from finn.core.executors.xsim.pacing import STALLED, Pacing
from finn.core.executors.xsim.rtl import Receive, SimulationFailed, Words, stream_out
from finn.core.space import Available
from finn.custom_op.kernels.shell import member
from finn.custom_op.partition.kernel_partitions import kernel_partition_body
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.traversal import Traversal, pack, unpack
from finn.transformation.kernels.package import configured_root
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
    """A boundary port of a partition: its tensor in the parent's context, whether it is
    an output, the traversal the port presents, and its element's payload bits and
    range."""

    tensor: str
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
        return is_partition(node) and kernel_partition_body(node) is not None

    def run(self, node: NodeProto, context: Context, model: ModelWrapper) -> None:
        body = kernel_partition_body(node)
        assert body is not None, f"{node.name}: XSim runs a partition of KernelOps"
        point, streams = boundary(body, node, self.completion)
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
        )
        for stream in streams:
            if not stream.output:
                continue
            words = received[stream.port]
            try:
                values = unpack(stream.form, words, stream.bits, signed=stream.low < 0)
            except ValueError as error:
                where = f"{node.name}: {stream.tensor} ({stream.port})"
                raise Unreadable(f"{where}: {error}", received) from None
            held = context[stream.tensor]
            found = values.reshape(np.shape(held)).astype(np.asarray(held).dtype)
            if not np.array_equal(found, values.reshape(found.shape)):
                raise Unreadable(
                    f"{node.name}: {stream.tensor}'s container ({found.dtype}) does not hold "
                    "the values its port presented exactly",
                    received,
                )
            context[stream.tensor] = found

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
        ends = getattr(point, member(tensor)).endpoints
        end = ends.source if ends.source_owner is None else ends.sink
        low, high = ordinary_integer_bounds(end.element.dtype)
        streams.append(
            Stream(outer[tensor], port, tensor in outputs, end.form, end.element.bits, low, high)
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


__all__ = ["Stream", "Unreadable", "Unstreamable", "XSim", "boundary"]
