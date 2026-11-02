# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shared driver (``Driver``): a build's cores run on a ``Device``, by the runtime file
(``description.Description``).

A driver refuses a runtime file whose edge a mover kind moves that no protocol drives
(``mover-unknown``, ``movers.PROTOCOLS``). A core runs by its node's name (``run_core``):
each input's frames packed into its mover's buffer and synced to the device, and each
output's buffer allocated (``stage``); each output mover, idle, started on its buffer, then
each input mover, and each output mover waited on (``launch``, which a throughput test
times alone: it allocates nothing); each output's buffer synced back and unpacked
(``collect``). An input holds a whole number of frames: its edge's shape, its first axis
that many times (``frames``); every input of a call holds as many. A mover holds one
buffer, of the size its last call took: a call of another size releases it and allocates
another (``Device.release``). ``execute`` runs every core on the inputs it
is given; ``run_graph`` runs the parent graph, its host nodes by ONNX execution (onnx and
qonnx, imported only there) and each ``DataflowCore`` node through ``run_core``, its inputs
and outputs the core's edge ports in port order.

A core is staged whole or not at all: staging marks it unstaged first, whatever it then
refuses (a frame, an input's presence, shape or values), packs every input before any
buffer is written, and marks it staged with its frames once every input is synced and every
output allocated; a launch or a collection of a core not staged at those frames is refused
(``core-unstaged``), so no launch moves one call's inputs beside another's. A stage the
driver's state refuses (closed, faulted, a launch not completed) changes nothing.

A mover's buffer is the hardware's from the moment the mover is started until its core's
launch completes, every output mover done (by then each input mover has moved its whole
buffer: the core has consumed it). While it is, the buffer is held in ``RETAINED`` too, with
its device and its driver, so dropping the driver frees none under a mover; loading the
image again on that device (``load``, by any driver of it) releases those of drivers since
dropped. A launch that did not complete:

- a status read failed, every mover started: its movers stay armed, their buffers neither
  written, released nor read again, and the core is refused (``mover-armed``) until
  ``settle`` finds each of them idle;
- a mover failed to start (a partial launch: its start may have reached the hardware, and
  others may have started): a started output mover may wait for ever, so the driver
  refuses every further use (``driver-faulted``), ``settle``, ``close`` and run-time
  weights included, until the image is loaded again (``load``), which resets every mover.

``close`` releases every buffer the driver holds, each armed mover found idle first; one
that is not is refused (``mover-busy``) and every buffer kept, never freed under the
hardware. A driver is a context manager that closes on its exit; a closed one is refused
(``driver-closed``).

A node's run-time weights (``write_weights``) are 32-bit words written from its endpoint's
offset, each read back where asked; more words than the endpoint holds are refused
(``weights-outside``), and so are weights on a closed or faulted driver, or while a launch
has not completed (``mover-armed``): they change between launches. A directory of them
holds a ``<node>.dat`` a node, its words in hexadecimal (``write_weight_files``).
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from contextlib import suppress
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from .description import Core, Description, Edge
from .device import Buffer, Device
from .errors import Refused
from .movers import PROTOCOLS
from .packing import Element, beat_bytes, pack, unpack

RETAINED: dict[int, tuple[Device, weakref.ref[Driver], Buffer]] = {}
"""Every buffer an armed mover holds, by its identity, across every driver of the process,
with its device and the driver that armed it: held here until its launch completes, it is
found idle, or the image is loaded again on its device, so none is freed under the hardware
when its driver is dropped."""


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
        self._buffers: dict[str, Buffer] = {}
        self._edges = {
            edge.unit: edge for core in description.cores for edge in (*core.inputs, *core.outputs)
        }
        self._armed: set[str] = set()
        self._staged: dict[str, int] = {}
        self._faulted: str | None = None
        self._closed = False

    def __enter__(self) -> Driver:
        return self

    def __exit__(self, kind: object, error: object, trace: object) -> None:
        """Close the driver; on an error, one that closing meets is not raised over it: a
        buffer whose mover was not found idle stays held."""
        if error is None:
            self.close()
            return
        with suppress(Exception):
            self.close()

    def load(self, image: Path) -> None:
        """Load the device image ``image``, which resets every mover: none is armed, the
        driver is no longer faulted, and no core is staged. The buffers armed movers of this
        device held for drivers since dropped are released (``RETAINED``); another
        driver's stay its own."""
        self.device.load(image)
        self._disarm(set(self._armed))
        self._faulted = None
        self._staged.clear()
        for key, (device, driver, buffer) in list(RETAINED.items()):
            if device is self.device and driver() is None:
                del RETAINED[key]
                self.device.release(buffer)

    # -- a core ---------------------------------------------------------------------------

    def _mover(self, edge: Edge) -> Any:
        """The protocol driving ``edge``'s mover."""
        return PROTOCOLS[edge.mover](self.device, edge.unit, self.description.endpoint(edge.unit))

    def _disarm(self, units: set[str]) -> None:
        """``units``' movers stopped: their buffers no longer retained."""
        for unit in units & self._armed:
            RETAINED.pop(id(self._buffers[unit]), None)
        self._armed -= units

    def _usable(self, name: str) -> None:
        """Refuse a closed driver, and a faulted one (the module docstring), for ``name``."""
        if self._closed:
            raise Refused("driver-closed", f"{name}: the driver is closed")
        if self._faulted is not None:
            raise Refused(
                "driver-faulted",
                f"{name}: {self._faulted}: a started mover may wait for ever; load the image "
                "again (load) to reset the movers",
            )

    def _quiet(self, core: Core) -> None:
        """Refuse ``core`` on a closed or faulted driver, or while a launch on its movers
        has not completed (the module docstring)."""
        self._usable(core.name)
        armed = sorted(
            edge.unit for edge in (*core.inputs, *core.outputs) if edge.unit in self._armed
        )
        if armed:
            raise Refused(
                "mover-armed",
                f"{core.name}: a launch on {armed} did not complete: their buffers are kept "
                "until the movers are found idle (settle)",
            )

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
        """The buffer of ``edge``'s mover for ``frames`` frames: the one it holds if of
        that size, else a new one, the one it held released."""
        element = Element.named(edge.element)
        per_beat = beat_bytes(element, edge.lanes)
        if per_beat * 8 != edge.bits:
            raise Refused(
                "stream-unpacked",
                f"{edge.unit}: {edge.lanes} lanes of {edge.element} take {per_beat} bytes a "
                f"beat, and its stream is {edge.bits} bits",
            )
        nbytes = frames * edge.beats * per_beat
        held = self._buffers.get(edge.unit)
        if held is not None and held.array.size == nbytes:
            return held
        if held is not None:
            del self._buffers[edge.unit]
            self.device.release(held)
        port, target = self.description.bindings[edge.unit]
        self._buffers[edge.unit] = self.device.buffer(edge.unit, port, nbytes, target)
        return self._buffers[edge.unit]

    def stage(self, name: str, inputs: Mapping[str, np.ndarray[Any, Any]]) -> int:
        """Pack the inputs of the core ``name`` into its input movers' buffers and sync them
        to the device, and allocate its output movers' (the module docstring): the frames
        they hold."""
        core = self.description.core(name)
        self._quiet(core)
        self._staged.pop(name, None)
        for edge in (*core.inputs, *core.outputs):
            if prod(edge.shape) != edge.beats * edge.lanes:
                raise Refused(
                    "frame-unfolded",
                    f"{edge.unit}: a frame of {edge.shape} is not {edge.beats} beats of "
                    f"{edge.lanes} lanes",
                )
        frames = self._frames(core, inputs)
        packed = {
            edge.unit: pack(
                np.asarray(inputs[edge.tensor]).reshape(frames, edge.beats, edge.lanes),
                Element.named(edge.element),
            ).reshape(-1)
            for edge in core.inputs
        }
        for edge in core.inputs:
            buffer = self._buffer(edge, frames)
            np.copyto(buffer.array, packed[edge.unit])
            self.device.sync(buffer, to_device=True)
        for edge in core.outputs:
            self._buffer(edge, frames)
        self._staged[name] = frames
        return frames

    def _held(self, core: Core, frames: int) -> dict[str, Buffer]:
        """The buffers of ``core``'s movers, staged for ``frames`` frames; a core not staged
        at those frames is refused (``core-unstaged``)."""
        if self._staged.get(core.name) != frames:
            raise Refused(
                "core-unstaged", f"{core.name}: not staged for {frames} frames: stage them first"
            )
        return {edge.unit: self._buffers[edge.unit] for edge in (*core.outputs, *core.inputs)}

    def launch(self, name: str, frames: int) -> None:
        """Move ``frames`` frames through the core ``name``, its inputs staged (``stage``):
        each output mover, idle, started, then each input mover; each output mover waited
        on. Each mover is armed as it is started, its buffer retained, and the core's
        movers are disarmed once every output mover is done; a mover that fails to start
        faults the driver (the module docstring)."""
        core = self.description.core(name)
        self._quiet(core)
        edges = (*core.outputs, *core.inputs)
        buffers = self._held(core, frames)
        movers = {edge.unit: self._mover(edge) for edge in edges}
        for edge in core.outputs:
            if not movers[edge.unit].idle():
                raise Refused("mover-busy", f"{edge.unit}: its output mover is not idle")
        started: list[str] = []
        for edge in edges:
            self._armed.add(edge.unit)
            buffer = buffers[edge.unit]
            RETAINED[id(buffer)] = (self.device, weakref.ref(self), buffer)
            try:
                movers[edge.unit].start(buffer, frames)
            except BaseException:
                after = f" after {started} started" if started else ""
                self._faulted = f"a launch of {core.name}: {edge.unit} failed to start{after}"
                raise
            started.append(edge.unit)
        for edge in core.outputs:
            movers[edge.unit].wait()
        self._disarm(set(buffers))

    def settle(self, name: str) -> None:
        """After a launch of the core ``name`` that did not complete, find each of its
        armed movers idle, its buffer the driver's again; one not idle is refused
        (``mover-busy``), its buffer kept. A faulted driver is refused (``driver-faulted``):
        only loading the image again resets its movers."""
        core = self.description.core(name)
        self._usable(core.name)
        for edge in (*core.outputs, *core.inputs):
            if edge.unit in self._armed:
                if not self._mover(edge).idle():
                    raise Refused(
                        "mover-busy", f"{edge.unit}: its mover is not idle: its buffer is kept"
                    )
                self._disarm({edge.unit})

    def collect(self, name: str, frames: int) -> dict[str, np.ndarray[Any, Any]]:
        """The outputs of the core ``name`` after ``frames`` frames were moved
        (``launch``): each output mover's buffer synced back and unpacked, by tensor, as
        float32."""
        core = self.description.core(name)
        self._quiet(core)
        buffers = self._held(core, frames)
        outputs: dict[str, np.ndarray[Any, Any]] = {}
        for edge in core.outputs:
            buffer = buffers[edge.unit]
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

    def close(self) -> None:
        """Release every buffer the driver holds, each armed mover found idle first; one
        that is not is refused (``mover-busy``), as is a faulted driver
        (``driver-faulted``), and every buffer kept, an armed mover's retained
        (``RETAINED``). Closing a closed driver does nothing."""
        if self._closed:
            return
        self._usable("close")
        for unit in sorted(self._armed):
            if not self._mover(self._edges[unit]).idle():
                raise Refused(
                    "mover-busy", f"{unit}: its mover is not idle: the driver's buffers are kept"
                )
        self._disarm(set(self._armed))
        self._staged.clear()
        for unit in sorted(self._buffers):
            self.device.release(self._buffers[unit])
            del self._buffers[unit]
        self._closed = True

    # -- run-time weights -----------------------------------------------------------------

    def write_weights(self, node: str, words: Sequence[int], verify: bool = True) -> None:
        """Write ``words``, 32 bits each, from the offset of the endpoint configuring graph
        node ``node``; each read back and compared where ``verify``. Refused on a closed or
        faulted driver, and while a launch has not completed (the module docstring)."""
        self._usable(node)
        if self._armed:
            raise Refused(
                "mover-armed",
                f"{node}: a launch on {sorted(self._armed)} did not complete: weights are "
                "written between launches (settle)",
            )
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
        directory is absent. Refused as ``write_weights`` is, before any is written."""
        self._usable("write_weight_files")
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


__all__ = ["RETAINED", "Driver"]
