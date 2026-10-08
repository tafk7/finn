# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The packaged partition's XSim testbench: one frame streamed through the partition's
module, its outputs checked against the partition's body executed in Python.

``write_testbench`` writes it with ``finn.harness.rtl``'s stream-testbench writer
(``stream_bench``: inputs and outputs stalled, every pin the module declares driven:
``ap_clk2x`` aligned to ``ap_clk``, each AXI-Lite bus's declared register writes made
before any stream, held ports at their values; each output word compared on its
payload bits; a watchdog from the beats and the model's cycles) into a directory that
holds what it needs and nothing of the machine:

- ``check.sv``, the testbench, and its stimulus and expected words in ``*.mem`` files;
- ``module/``, the module's sources, and each memory's INIT_FILE beside the testbench;
- ``run.sh``, which compiles, elaborates and runs it with Vivado's simulator found on
  PATH (``xvlog``, ``xelab``, ``xsim``), ``glbl.v`` from ``$XILINX_VIVADO``, and exits
  0 printing ``PASS`` when every output word matched.

The writer simulates nothing. ``write_testbench`` then runs the testbench once, through
the toolchain it is given (``simulate``), so a testbench a build writes is one that
passed. ``run.sh`` repeats ``simulate``'s commands with the directory's own paths.

The words are the partition's boundary values as each boundary channel's end presents
them (``boundary_words``): the end's form (``finn.dataflow.traversal.Traversal``) gives
each beat's elements, packed lane zero lowest. The frame is the partition's inputs:
``partition_frame`` executes the parent graph, its partition node running its body, on
a source input (``verify_input_npy``'s first), or ``generated_frame`` draws each input
from its datatype.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

from finn.core.onnx_exec import execute_onnx as execute_parent_graph
from finn.custom_op.kernels.partition import member
from finn.harness.rtl import Words, pack, simulate, stream_bench
from finn.kernels.artifacts.module import module_name
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.kernels.explore import Completion
from finn.transformation.fpgadataflow.kernel_partitions import partition_body
from finn.transformation.kernels.package import boundary_facts, configured_root
from finn.util.toolchain import Toolchain

#: The directory of the testbench, beside the packaged IP.
TESTBENCH_DIR = "testbench"


def words(form: Any, values: NDArray[Any], bits: int) -> list[int]:
    """The beats ``form`` (a Traversal) presents of ``values``, each packed lane zero
    lowest, ``bits`` a lane."""
    flat = values.reshape(form.shape)
    return [pack([int(flat[position]) for position in beat], bits) for beat in form.positions()]


def boundary_words(
    model: ModelWrapper,
    point: Any,
    boundary: Sequence[tuple[str, str]],
    context: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Words], dict[str, Words]]:
    """Each boundary port's words of the frame in ``context``, inputs then outputs, as the
    port's channel end presents them (the end no kernel of the partition owns)."""
    inputs, outputs = boundary_facts(model, point, boundary, label)
    found: tuple[dict[str, Words], dict[str, Words]] = ({}, {})
    for side, facts in zip(found, (inputs, outputs), strict=True):
        for each in facts:
            ends = getattr(point, member(each["tensor"])).endpoints
            end = ends.source if ends.source_owner is None else ends.sink
            bits = int(end.element.bits)
            side[each["port"]] = (
                words(end.form, context[each["tensor"]], bits),
                bits * each["lanes"],
            )
    return found


def _inputs(body: ModelWrapper) -> list[str]:
    initializers = {tensor.name for tensor in body.graph.initializer}
    return [item.name for item in body.graph.input if item.name not in initializers]


def generated_frame(body: ModelWrapper, seed: int = 0) -> dict[str, NDArray[Any]]:
    """One frame of the partition's inputs, each drawn uniformly from its datatype's
    integers (seeded)."""
    rng = np.random.default_rng(seed)
    frame: dict[str, NDArray[Any]] = {}
    for name in _inputs(body):
        datatype = body.get_tensor_datatype(name)
        if not datatype.is_integer():
            raise ValueError(f"{name}: a {datatype.name} input; the testbench streams integers")
        shape = body.get_tensor_shape(name)
        low, high = int(datatype.min()), int(datatype.max())
        frame[name] = rng.integers(low, high + 1, size=shape).astype(np.float32)
    return frame


def partition_frame(parent: ModelWrapper, source_input: NDArray[Any]) -> dict[str, NDArray[Any]]:
    """The partition's inputs on ``source_input``, one frame of the source model's input
    (reshaped to the parent graph's input): the parent graph executed, its partition
    node running its body (``kernel_partitions.partition_body``)."""
    node, body, _ = partition_body(parent)
    shape = parent.get_tensor_shape(parent.graph.input[0].name)
    if shape is None:
        raise ValueError(f"{node.name}: the parent graph's input states no shape")
    frame = source_input.reshape(shape)
    context = execute_parent_graph(  # type: ignore[no-untyped-call]
        parent, {parent.graph.input[0].name: frame}, return_full_exec_context=True
    )
    return {name: context[name] for name in _inputs(body)}


def _run_script(top: str, sources: Sequence[str]) -> str:
    """``run.sh``: the harness's simulator commands (``finn.harness.rtl.simulate``) with
    the testbench directory's own paths."""
    compiled = [source for source in sources if not is_header(source)]
    xvlog = [
        "xvlog",
        "--sv",
        "--relax",
        *(f"--include={directory}" for directory in include_directories(sources)),
        *compiled,
        '"$XILINX_VIVADO/data/verilog/src/glbl.v"',
        "check.sv",
    ]
    xelab = [
        "xelab",
        "work.check",
        "work.glbl",
        "--mt",
        "2",
        "-L",
        "unisims_ver",
        "-L",
        "unimacro_ver",
        "--snapshot",
        "check",
        "--timescale",
        "1ns/1ps",
    ]

    def line(command: Sequence[str]) -> str:
        return " \\\n    ".join(
            word if word.startswith('"$') else shlex.quote(word) for word in command
        )

    return f"""#!/bin/sh
# The XSim testbench of {top}: one frame streamed through the module (check.sv), each
# output word checked against the partition executed in Python. Needs Vivado's
# simulator on PATH and XILINX_VIVADO set (Vivado's settings64.sh); prints PASS and
# exits 0 when every word matched.
set -e
cd "$(dirname "$0")"
: "${{XILINX_VIVADO:?set XILINX_VIVADO (source Vivado's settings64.sh)}}"
{line(xvlog)}
{line(xelab)}
xsim check --runall --log xsim.log
grep -q STREAM_PASS xsim.log
echo PASS
"""


def write_testbench(
    body: ModelWrapper,
    directory: Path,
    frame: Mapping[str, NDArray[Any]],
    *,
    completion: Completion | None = None,
    label: str = "the partition",
    toolchain: Toolchain | None = None,
) -> None:
    """Write the XSim testbench of the partition ``body`` into ``directory``, streaming
    ``frame`` (its inputs by tensor) and expecting the body's outputs on it, executed in
    Python; see the module docstring. The module is the one PackagePartition packages
    (``configured_root`` under ``completion``), and the watchdog allows the cycles its
    root states. The testbench is run once, through ``toolchain`` (the machine's by
    default): a mismatch raises ``finn.harness.rtl.SimulationFailed``."""
    point, boundary = configured_root(body, label, completion)
    context = execute_onnx(body, dict(frame), return_full_exec_context=True)
    inputs, outputs = boundary_words(body, point, boundary, context, label)
    directory.mkdir(parents=True, exist_ok=True)
    bench = stream_bench(
        point.module, directory, inputs=inputs, outputs=outputs, cycles=int(point.cycles)
    )
    relative = [str(Path(source).relative_to(directory)) for source in bench.sources]
    script = directory / "run.sh"
    script.write_text(_run_script(module_name(point.module), relative))
    script.chmod(0o755)
    simulate(bench.sources, bench.text, directory, toolchain=toolchain)


__all__ = [
    "TESTBENCH_DIR",
    "boundary_words",
    "generated_frame",
    "partition_frame",
    "words",
    "write_testbench",
]
