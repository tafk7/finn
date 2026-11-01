# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shared driver (``Driver``): a build's cores run on a ``Device``, by the runtime file
(``description.Description``).

A driver refuses a runtime file whose edge a mover kind moves that no protocol drives
(``mover-unknown``, ``movers.PROTOCOLS``). A core runs by its node's name (``run_core``):
each input's frames packed into its mover's buffer and synced to the device (``stage``);
each output mover, idle, started on its buffer, then each input mover, and each output
mover waited on (``launch``, which a throughput test times alone); each output's buffer
synced back and unpacked (``collect``). An input holds a whole number of frames: its edge's
shape, its first axis that many times (``frames``); every input of a call holds as many.
Buffers are allocated once a size, by mover. ``execute`` runs every core on the inputs it
is given; ``run_graph`` runs the parent graph, its host nodes by ONNX execution (onnx and
qonnx, imported only there) and each ``DataflowCore`` node through ``run_core``, its inputs
and outputs the core's edge ports in port order.

A node's run-time weights (``write_weights``) are 32-bit words written from its endpoint's
offset, each read back where asked; more words than the endpoint holds are refused
(``weights-outside``). A directory of them holds a ``<node>.dat`` a node, its words in
hexadecimal (``write_weight_files``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from .description import Core, Description, Edge
from .device import Buffer, Device
from .errors import Refused
from .movers import PROTOCOLS
from .packing import Element, beat_bytes, pack, unpack


class Driver:
    """A build's cores on ``device``, as its runtime file ``description`` states them (the
    module docstring)."""

    def __init__(self, device: Device, description: Description) -> None:
        for core in description.cores:
            for edge in (*core.inputs, *core.outputs):
                if edge.mover not in PROTOCOLS:
                    raise Refused(
                        "mover-unknown",
                        f"{edge.unit}: its mover kind {edge.mover!r} has no protocol in this "
                        f"driver (the kinds: {sorted(PROTOCOLS)})",
                    )
        self.device = device
        self.description = description
        self._buffers: dict[tuple[str, int], Buffer] = {}

    def load(self, image: Path) -> None:
        """Load the device image ``image``."""
        self.device.load(image)

    # -- a core ---------------------------------------------------------------------------

    def _frames(self, core: Core, inputs: Mapping[str, np.ndarray[Any, Any]]) -> int:
        """The frames every input of ``core`` holds, the same for each; an input absent, of
        another shape or of a frame count another holds is refused."""
        counts: dict[str, int] = {}
        for edge in core.inputs:
            if edge.tensor not in inputs:
                raise Refused("input-absent", f"{core.name}: no value given for {edge.tensor}")
            shape = tuple(np.shape(inputs[edge.tensor]))
            first = edge.shape[0] if edge.shape else 1
            if (
                len(shape) != len(edge.shape)
                or shape[1:] != edge.shape[1:]
                or not shape
                or shape[0] % first
                or shape[0] == 0
            ):
                raise Refused(
                    "input-shape",
                    f"{core.name}: {edge.tensor} is {shape}, not frames of {edge.shape}",
                )
            counts[edge.tensor] = shape[0] // first
        if len(set(counts.values())) > 1:
            raise Refused("input-frames", f"{core.name}: its inputs hold {counts} frames")
        return next(iter(counts.values()), 1)

    def _buffer(self, edge: Edge, frames: int) -> Buffer:
        """The buffer of ``edge``'s mover for ``frames`` frames, allocated once a size."""
        element = Element.named(edge.element)
        per_beat = beat_bytes(element, edge.lanes)
        if per_beat * 8 != edge.bits:
            raise Refused(
                "stream-unpacked",
                f"{edge.unit}: {edge.lanes} lanes of {edge.element} take {per_beat} bytes a "
                f"beat, and its stream is {edge.bits} bits",
            )
        nbytes = frames * edge.beats * per_beat
        key = (edge.unit, nbytes)
        if key not in self._buffers:
            port, target = self.description.bindings[edge.unit]
            self._buffers[key] = self.device.buffer(edge.unit, port, nbytes, target)
        return self._buffers[key]

    def stage(self, name: str, inputs: Mapping[str, np.ndarray[Any, Any]]) -> int:
        """Pack the inputs of the core ``name`` into its input movers' buffers and sync them
        to the device (the module docstring): the frames they hold."""
        core = self.description.core(name)
        for edge in (*core.inputs, *core.outputs):
            if prod(edge.shape) != edge.beats * edge.lanes:
                raise Refused(
                    "frame-unfolded",
                    f"{edge.unit}: a frame of {edge.shape} is not {edge.beats} beats of "
                    f"{edge.lanes} lanes",
                )
        frames = self._frames(core, inputs)
        for edge in core.inputs:
            element = Element.named(edge.element)
            folded = np.asarray(inputs[edge.tensor]).reshape(frames, edge.beats, edge.lanes)
            buffer = self._buffer(edge, frames)
            np.copyto(buffer.array, pack(folded, element).reshape(-1))
            self.device.sync(buffer, to_device=True)
        return frames

    def launch(self, name: str, frames: int) -> None:
        """Move ``frames`` frames through the core ``name``, its inputs staged (``stage``):
        each output mover, idle, started, then each input mover; each output mover waited
        on."""
        core = self.description.core(name)
        movers = {
            edge.unit: PROTOCOLS[edge.mover](
                self.device, edge.unit, self.description.endpoint(edge.unit)
            )
            for edge in (*core.inputs, *core.outputs)
        }
        for edge in core.outputs:
            if not movers[edge.unit].idle():
                raise Refused("mover-busy", f"{edge.unit}: its output mover is not idle")
        for edge in (*core.outputs, *core.inputs):
            movers[edge.unit].start(self._buffer(edge, frames), frames)
        for edge in core.outputs:
            movers[edge.unit].wait()

    def collect(self, name: str, frames: int) -> dict[str, np.ndarray[Any, Any]]:
        """The outputs of the core ``name`` after ``frames`` frames were moved
        (``launch``): each output mover's buffer synced back and unpacked, by tensor, as
        float32."""
        core = self.description.core(name)
        outputs: dict[str, np.ndarray[Any, Any]] = {}
        for edge in core.outputs:
            buffer = self._buffer(edge, frames)
            self.device.sync(buffer, to_device=False)
            element = Element.named(edge.element)
            packed = buffer.array.reshape(frames, edge.beats, -1)
            first = edge.shape[0] if edge.shape else 1
            shape = (frames * first, *edge.shape[1:])
            outputs[edge.tensor] = unpack(packed, element, edge.lanes).reshape(shape)
        return outputs

    def run_core(
        self, name: str, inputs: Mapping[str, np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Run the core ``name`` on ``inputs``, by its edge's tensors (``stage``,
        ``launch``, ``collect``): its outputs, by tensor, as float32."""
        frames = self.stage(name, inputs)
        self.launch(name, frames)
        return self.collect(name, frames)

    def execute(
        self, inputs: Mapping[str, np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Run every core, in order, each on the inputs given by its edge's tensors: every
        core's outputs, by tensor."""
        outputs: dict[str, np.ndarray[Any, Any]] = {}
        for core in self.description.cores:
            outputs.update(self.run_core(core.name, inputs))
        return outputs

    # -- run-time weights -----------------------------------------------------------------

    def write_weights(self, node: str, words: Sequence[int], verify: bool = True) -> None:
        """Write ``words``, 32 bits each, from the offset of the endpoint configuring graph
        node ``node``; each read back and compared where ``verify``."""
        endpoint = self.description.node(node)
        if len(words) * 4 > endpoint.range:
            raise Refused(
                "weights-outside",
                f"{node}: {len(words)} words, and its endpoint {endpoint.unit}/{endpoint.bus} "
                f"holds {endpoint.range // 4}",
            )
        for index, word in enumerate(words):
            self.device.write32(endpoint.unit, endpoint.offset + 4 * index, word)
        if verify:
            read = [
                self.device.read32(endpoint.unit, endpoint.offset + 4 * index)
                for index in range(len(words))
            ]
            differ = [index for index, (a, b) in enumerate(zip(words, read)) if a != b]
            if differ:
                raise Refused(
                    "weights-unwritten",
                    f"{node}: the words at {differ[:4]} read back otherwise than written",
                )

    def write_weight_files(self, directory: Path) -> list[str]:
        """Write the run-time weights in ``directory``: each ``<node>.dat``, its words in
        hexadecimal, whitespace between them, written to the node ``<node>``
        (``write_weights``), the files in name order. The nodes written; none where the
        directory is absent."""
        written = []
        for path in sorted(Path(directory).glob("*.dat")):
            self.write_weights(path.stem, [int(word, 16) for word in path.read_text().split()])
            written.append(path.stem)
        return written

    # -- the parent graph -----------------------------------------------------------------

    def run_graph(
        self, parent: Path, inputs: Mapping[str, np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Run the parent graph at ``parent`` on ``inputs``, by its inputs' names: each
        ``DataflowCore`` node through ``run_core``, by its name, every other node by ONNX
        execution (onnx and qonnx; their absence refused, ``host-execution-unavailable``).
        Its outputs, by name."""
        try:
            from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
            from qonnx.core.onnx_exec import execute_node  # noqa: PLC0415
        except ImportError as error:
            raise Refused(
                "host-execution-unavailable",
                f"running the parent graph's host nodes takes onnx and qonnx ({error.name} "
                "is not installed): install them, or run the cores alone (execute)",
            ) from error
        model = ModelWrapper(str(parent))
        context: dict[str, Any] = model.make_empty_exec_context()
        context.update(inputs)
        opset = model.model.opset_import[0].version
        for node in model.graph.node:
            if node.op_type != "DataflowCore":
                execute_node(node, context, model.graph, opset, model=model)  # type: ignore[no-untyped-call]
                continue
            core = self.description.core(node.name)
            given = {edge.tensor: context[name] for edge, name in zip(core.inputs, node.input)}
            found = self.run_core(node.name, given)
            for edge, name in zip(core.outputs, node.output):
                context[name] = found[edge.tensor]
        return {each.name: context[each.name] for each in model.graph.output}


__all__ = ["Driver"]
