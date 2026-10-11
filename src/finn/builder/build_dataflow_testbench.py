# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The packaged partition's XSim testbench: one frame streamed through the partition's
module, its outputs checked against the partition's body executed in Python.

``write_testbench`` writes it with ``finn.core.rtlsim_bench``'s stream-testbench writer
(``stream_bench``: inputs and outputs stalled, every pin the module declares driven:
``ap_clk2x`` aligned to ``ap_clk``, each AXI-Lite bus's declared register writes made
before any stream, held ports at their values; each output word compared on its
payload bits; a watchdog from the beats and the model's cycles) into a directory that
holds what it needs and nothing of the machine:

- ``check.sv``, the testbench, and its stimulus and expected words in ``*.mem`` files;
- ``module/``, the module's sources, and each memory's INIT_FILE beside the testbench;
- ``run.sh``, which compiles, elaborates and runs it with Vivado's simulator found on
  PATH (``xvlog``, ``xelab``, ``xsim``), ``glbl.v`` from ``$XILINX_VIVADO``, and exits
  0 printing ``PASS`` when every output word matched; a failed compilation or
  elaboration exits with its own status (``finn.toolchain.xsim.EXIT_STATUS``).

``write_testbench`` simulates nothing: the testbench is an output, and no build
re-verifies the partition's computation (PRINCIPLES §8). ``run_testbench`` runs it,
its ``run.sh`` as its user would, in a toolchain's environment: the build's
verification step ``stitched_ip_testbench``, which the user asks for, and which fails
on a mismatch. ``run.sh``'s commands are XSim's driver's (``finn.toolchain.xsim.script``),
which ``finn.core.rtlsim_bench.simulate`` runs, with the directory's own paths.

The words are the partition's boundary values as each boundary channel's end presents
them (``boundary_words``): the end's form (``finn.dataflow.traversal.Traversal``) gives
each beat's elements, packed lane zero lowest. The frame is the partition's inputs:
``partition_frame`` executes the parent graph, its partition node running its body, on
a source input (``verify_input_npy``'s first), or ``generated_frame`` draws each input
from its datatype.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.containers import container, held, numpy_type
from finn.core.onnx_exec import execute_onnx
from finn.core.rtlsim_bench import SimulationFailed, Words, stream_bench
from finn.custom_op.kernels.shell import configured_root
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.dataflow.traversal import pack
from finn.harness.ops import graph_inputs
from finn.kernels.artifacts.module import module_name
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.toolchain import xsim
from finn.toolchain.launcher import Toolchain, ToolFailed, run_process
from finn.transformation.kernels.package import boundary_facts, free_side

#: The directory of the testbench, beside the packaged IP.
TESTBENCH_DIR = "testbench"

#: The script that compiles, elaborates and runs the testbench, in its directory.
RUN_SCRIPT = "run.sh"

#: How long ``run_testbench`` lets the script run: three tool runs of at most five
#: minutes each, as ``finn.core.rtlsim_bench.simulate`` allows each.
RUN_TIMEOUT = 900


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
            end = free_side(point, each["tensor"])
            bits = int(end.element.bits)
            side[each["port"]] = (
                list(pack(end.form, np.asarray(context[each["tensor"]]).reshape(-1), bits)),
                bits * each["lanes"],
            )
    return found


def generated_frame(body: ModelWrapper, seed: int = 0) -> dict[str, NDArray[Any]]:
    """One frame of the partition's inputs, each drawn uniformly from its datatype's
    integers (seeded), held by the input's container (finn.core.containers: the
    export's float32, or float64 where graph preparation widened the region)."""
    rng = np.random.default_rng(seed)
    frame: dict[str, NDArray[Any]] = {}
    for name in graph_inputs(body):
        datatype = body.get_tensor_datatype(name)
        if not datatype.is_integer():
            raise ValueError(f"{name}: a {datatype.name} input; the testbench streams integers")
        shape = body.get_tensor_shape(name)
        low, high = int(datatype.min()), int(datatype.max())
        element_type = container(body, name)
        if element_type is None:
            raise ValueError(f"{name}: the partition's input states no container")
        drawn = rng.integers(low, high + 1, size=shape, dtype=np.int64)
        frame[name] = held(drawn, element_type).astype(numpy_type(element_type))
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
    context = execute_onnx(
        parent, {parent.graph.input[0].name: frame}, return_full_exec_context=True
    )
    return {name: context[name] for name in graph_inputs(body)}


def _run_script(top: str, sources: Sequence[str]) -> str:
    """``run.sh``: the simulator's commands (``finn.toolchain.xsim.script``), with the
    testbench directory's own paths and two ``xelab`` threads, then the verdict."""
    commands = xsim.script(
        [source for source in sources if not is_header(source)],
        "check",
        includes=include_directories(sources),
        testbench="check.sv",
        snapshot="check",
        threads="2",
        log="xsim.log",
    )
    return f"""#!/bin/sh
# The XSim testbench of {top}: one frame streamed through the module (check.sv), each
# output word checked against the partition executed in Python. Needs Vivado's
# simulator on PATH and XILINX_VIVADO set (Vivado's settings64.sh); prints PASS and
# exits 0 when every word matched.
set -e
cd "$(dirname "$0")"
{commands}grep -q STREAM_PASS xsim.log
echo PASS
"""


def write_testbench(
    body: ModelWrapper,
    directory: Path,
    frame: Mapping[str, NDArray[Any]],
    *,
    toolchain: Toolchain,
    label: str = "the partition",
) -> None:
    """Write the XSim testbench of the partition ``body`` into ``directory``, streaming
    ``frame`` (its inputs by tensor) and expecting the body's outputs on it, executed in
    Python; see the module docstring. The module is the one PackagePartition packages
    (``configured_root``), and the watchdog allows the cycles its
    root states. Nothing is simulated (``run_testbench`` runs it); ``toolchain`` synthesizes
    an HLS leaf's request first, unless the HLS cache holds it
    (``finn.core.rtlsim_bench.materialize``)."""
    point, boundary = configured_root(body, label)
    context = execute_onnx(body, dict(frame), return_full_exec_context=True)
    inputs, outputs = boundary_words(body, point, boundary, context, label)
    directory.mkdir(parents=True, exist_ok=True)
    bench = stream_bench(
        point.module,
        directory,
        inputs=inputs,
        outputs=outputs,
        cycles=int(point.cycles),
        toolchain=toolchain,
    )
    (directory / "check.sv").write_text("`timescale 1ns/1ps\n" + bench.text)
    relative = [str(Path(source).relative_to(directory)) for source in bench.sources]
    script = directory / RUN_SCRIPT
    script.write_text(_run_script(module_name(point.module), relative))
    script.chmod(0o755)


def run_testbench(directory: Path, *, toolchain: Toolchain) -> str:
    """Run the testbench ``write_testbench`` wrote into ``directory``: its ``run.sh``, in
    ``toolchain``'s environment, a command directory it
    selects first on ``PATH``, written into its transcript as it runs: in the
    toolchain's log directory when it has one (``Toolchain.transcript``), else beside
    ``run.sh`` (``run.sh.stdout.log``, ``.stderr.log``). Returns what it printed, which
    ends in ``PASS``. ``xvlog`` or ``xelab`` failing (the script's
    ``finn.toolchain.xsim.EXIT_STATUS``) raises ``ToolFailed``, naming the transcript;
    a mismatch, any other failure or a timeout
    ``finn.core.rtlsim_bench.SimulationFailed`` with what it printed. A launcher route,
    which runs a tool elsewhere, runs no local script: refused (``ValueError``)."""
    selection = toolchain.selection
    if selection.launcher:
        raise ValueError(
            f"{directory / RUN_SCRIPT}: the toolchain's launcher route runs tools elsewhere, "
            "not a local script; run the testbench on a local route"
        )
    environment = dict(toolchain.environment)
    if selection.command_dir:
        environment["PATH"] = f"{selection.command_dir}:{environment.get('PATH', '')}"
    try:
        result = run_process(
            ["sh", str(directory / RUN_SCRIPT)],
            env=environment,
            cwd=directory,
            timeout=RUN_TIMEOUT,
            check=False,
            transcript=toolchain.transcript(RUN_SCRIPT) or directory / RUN_SCRIPT,
        )
    except subprocess.TimeoutExpired as expired:
        printed = (expired.output or b"") + (expired.stderr or b"")
        raise SimulationFailed(
            f"{RUN_SCRIPT} ran over {RUN_TIMEOUT} s:\n" + printed.decode(errors="replace")
        ) from expired
    for tool, status in xsim.EXIT_STATUS.items():
        if result.returncode == status:
            # The script's transcript holds the tool's output (each also writes its own
            # log where it runs, ``xvlog.log``, ``xelab.log``).
            raise ToolFailed.of_run(tool, result, what=str(directory / RUN_SCRIPT))
    output = result.stdout.decode(errors="replace")
    printed = output + result.stderr.decode(errors="replace")
    if result.returncode != 0 or output.splitlines()[-1:] != ["PASS"]:
        raise SimulationFailed(printed)
    return output


__all__ = [
    "RUN_SCRIPT",
    "TESTBENCH_DIR",
    "boundary_words",
    "generated_frame",
    "partition_frame",
    "run_testbench",
    "write_testbench",
]
