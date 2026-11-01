# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The command line a build's driver runs on its board (``main``): the shared driver over a
runtime's device, by the runtime file beside it.

A runtime writes a launcher (``driver.py``) beside the runtime file; it calls ``main`` with
its device's constructor (``OpenDevice``: the runtime file and the device's index) and the
device image it runs unless told another. Before either mode, the run-time weights in
``--runtime_weight_dir`` are written (``Driver.write_weight_files``).

- ``--exec_mode execute`` (the default) runs each core on the inputs in ``--inputfile``, one
  ``.npy`` file an input edge, the cores' edges in order, and saves its outputs into
  ``--outputfile`` in the same order. Another count of files is refused
  (``inputs-count``).
- ``--exec_mode throughput_test`` runs each core on ``--batchsize`` frames of values each
  element holds and writes what it measured into ``nw_metrics.txt`` (``throughput``).

A refusal is printed, its code first, and the command exits with status 1.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .description import Description, Edge, read
from .device import Device
from .driver import Driver
from .errors import Refused
from .packing import Element, beat_bytes

OpenDevice = Callable[[Description, int], Device]
"""A runtime's device, opened for the runtime file on the device of the index given."""

RUNTIME_FILE = "integration.json"
"""The runtime file's name, beside the launcher."""

METRICS = Path("nw_metrics.txt")
"""Where the throughput test writes what it measured."""


def arguments(image: Path) -> argparse.ArgumentParser:
    """The command line's arguments, the device image ``image`` the default of
    ``--bitfile``."""
    parser = argparse.ArgumentParser(
        description="Run a FINN build's cores on numpy inputs, or test their throughput"
    )
    parser.add_argument(
        "--exec_mode",
        help='functional execution ("execute") or a throughput test ("throughput_test")',
        choices=("execute", "throughput_test"),
        default="execute",
    )
    parser.add_argument(
        "--batchsize", help="the frames a throughput test moves", type=int, default=1
    )
    parser.add_argument("--device", help="the index of the device to use", type=int, default=0)
    parser.add_argument(
        "--bitfile", help="the bitfile to run (default: %(default)s)", default=str(image)
    )
    parser.add_argument(
        "--inputfile",
        help="an input .npy file an input of the build's cores, in order",
        nargs="*",
        default=["input.npy"],
    )
    parser.add_argument(
        "--outputfile",
        help="an output .npy file an output of the build's cores, in order",
        nargs="*",
        default=["output.npy"],
    )
    parser.add_argument(
        "--runtime_weight_dir",
        help="a directory of run-time weights, a <node>.dat file a node",
        default="runtime_weights/",
    )
    return parser


def edges(description: Description, side: str) -> list[Edge]:
    """Every core's input (``side`` ``inputs``) or output (``outputs``) edges, the cores in
    order."""
    return [edge for core in description.cores for edge in getattr(core, side)]


def held(edge: Edge, frames: int) -> np.ndarray[Any, Any]:
    """``frames`` frames of ``edge`` holding a value its element holds: zero, or one for an
    element without it."""
    value = 1.0 if edge.element == "BIPOLAR" else 0.0
    first = edge.shape[0] if edge.shape else 1
    return np.full((frames * first, *edge.shape[1:]), value, dtype=np.float32)


def throughput(driver: Driver, frames: int) -> dict[str, float]:
    """Each core run on ``frames`` frames (``held``), timed apart: the moves alone
    (``Driver.launch``) as the runtime, the frames a second, the bytes in and out a second,
    the clock the runtime file states, and the staging (packing and syncing the inputs)
    and the collection (syncing and unpacking the outputs)."""
    description = driver.description
    staged = launched = collected = 0.0
    for core in description.cores:
        start = time.perf_counter()
        driver.stage(core.name, {edge.tensor: held(edge, frames) for edge in core.inputs})
        middle = time.perf_counter()
        driver.launch(core.name, frames)
        end = time.perf_counter()
        driver.collect(core.name, frames)
        staged, launched = staged + middle - start, launched + end - middle
        collected += time.perf_counter() - end

    def moved(side: str) -> int:
        return sum(
            frames * edge.beats * beat_bytes(Element.named(edge.element), edge.lanes)
            for edge in edges(description, side)
        )

    return {
        "runtime[ms]": launched * 1000,
        "throughput[images/s]": frames / launched,
        "DRAM_in_bandwidth[MB/s]": moved("inputs") * 0.000001 / launched,
        "DRAM_out_bandwidth[MB/s]": moved("outputs") * 0.000001 / launched,
        "fclk[mhz]": description.clocks_mhz.get("main", 0.0),
        "batch_size": frames,
        "pack_input[ms]": staged * 1000,
        "unpack_output[ms]": collected * 1000,
    }


def execute(driver: Driver, inputs: Sequence[str], outputs: Sequence[str]) -> None:
    """Run each core on the ``.npy`` files ``inputs`` and save its outputs into
    ``outputs``, each an edge, the cores' edges in order."""
    description = driver.description
    given, taken = edges(description, "inputs"), edges(description, "outputs")
    if (len(inputs), len(outputs)) != (len(given), len(taken)):
        raise Refused(
            "inputs-count",
            f"{len(inputs)} input and {len(outputs)} output files, and the build's cores "
            f"take {[edge.tensor for edge in given]} and return "
            f"{[edge.tensor for edge in taken]}",
        )
    values = {edge.tensor: np.load(path) for edge, path in zip(given, inputs)}
    found = driver.execute(values)
    for edge, path in zip(taken, outputs):
        np.save(path, found[edge.tensor])


def main(argv: Sequence[str] | None, open_device: OpenDevice, here: Path, image: Path) -> int:
    """The command line (the module docstring) on ``argv``: the runtime file in ``here``, the
    device ``open_device`` opens, ``image`` the device image it runs unless told another.
    Its exit status."""
    args = arguments(image).parse_args(argv)
    try:
        description = read(here / RUNTIME_FILE)
        driver = Driver(open_device(description, args.device), description)
        driver.load(Path(args.bitfile))
        driver.write_weight_files(Path(args.runtime_weight_dir))
        if args.exec_mode == "execute":
            execute(driver, args.inputfile, args.outputfile)
        else:
            found = throughput(driver, args.batchsize)
            METRICS.write_text(str(found))
            sys.stdout.write(f"Results written to {METRICS}\n")
    except Refused as refused:
        sys.stderr.write(f"refused: {refused}\n")
        return 1
    return 0


__all__ = [
    "METRICS",
    "RUNTIME_FILE",
    "OpenDevice",
    "arguments",
    "edges",
    "execute",
    "held",
    "main",
    "throughput",
]
