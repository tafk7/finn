# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The packaged partition's XSim testbench: one frame streamed through the partition's
module, its outputs checked against the partition's body executed in Python.

``write_testbench`` writes it with the stream testbench of ``finn.core.rtlsim_bench``
(``stream_bench``). That testbench stalls inputs and outputs, drives every pin the
module declares (``ap_clk2x`` aligned to ``ap_clk``, each AXI-Lite bus's register writes
before any stream, held ports at their values), compares each output word on its
payload bits, and sets a watchdog from the beats and the model's cycles. The directory
holds everything the testbench needs and nothing from the build machine:

- ``check.sv``, the testbench, with its stimulus and expected words in ``*.mem`` files;
- ``module/``, the module's sources, with each memory's INIT_FILE beside the testbench;
- ``run.sh``, which compiles, elaborates and runs it with Vivado's simulator from PATH
  (``xvlog``, ``xelab``, ``xsim``) and ``glbl.v`` from ``$XILINX_VIVADO``, and exits 0
  printing ``PASS`` when every output word matched. A failed compilation or elaboration
  exits with its own status (``finn.toolchain.xsim.EXIT_STATUS``), and so does an
  environment without the simulator (``UNAVAILABLE_STATUS``), before anything runs.

``write_testbench`` simulates nothing: the testbench is an output, and a build does not
re-verify the partition's computation. ``run_testbench`` runs ``run.sh`` as a user
would, in a toolchain's environment: a utility a test or a user calls, not a step, which
fails on a mismatch. ``run.sh`` holds the same commands
``finn.core.rtlsim_bench.simulate`` runs (``finn.toolchain.xsim.script``), with the
directory's own paths.

The words are the partition's boundary values as each boundary channel's free side
presents them (``boundary_words``): its stream's form
(``finn.dataflow.traversal.Traversal``) gives each beat's elements, packed with lane zero
lowest. The build's frame draws each of the partition's inputs from its range
(``generated_frame``).
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.onnx_exec import execute_onnx
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.rtlsim_bench import SimulationFailed, Words, stream_bench
from finn.custom_op.kernels.shell import configured_network, refuse_unverified
from finn.dataflow.traversal import pack
from finn.kernels.artifacts.module import module_name
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.toolchain import xsim
from finn.toolchain.launcher import Toolchain, ToolFailed, ToolUnavailable, run_process
from finn.transformation.kernels.package import boundary_facts, free_side
from finn.util.draws import graph_inputs

#: The directory of the testbench, beside the packaged IP.
TESTBENCH_DIR = "testbench"

#: The script that compiles, elaborates and runs the testbench, in its directory.
RUN_SCRIPT = "run.sh"

#: How long ``run_testbench`` lets the script run: three tool runs of at most five
#: minutes each, the limit ``finn.core.rtlsim_bench.simulate`` gives each.
RUN_TIMEOUT = 900


def boundary_words(
    model: ModelWrapper,
    point: Any,
    boundary: Sequence[tuple[str, str]],
    context: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Words], dict[str, Words]]:
    """Each boundary port's words for the frame in ``context``, inputs then outputs, as
    the port's channel presents them at its free side (the side no kernel of the partition
    owns)."""
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
    """One frame of the partition's inputs, each drawn uniformly (seeded) from the
    integers of its range (its datatype's where it states none), as int64: FINN executes
    the body on integers as they are (``finn.core.onnx_exec``). An input that is not an
    integer is refused (``input-not-integer``)."""
    rng = np.random.default_rng(seed)
    frame: dict[str, NDArray[Any]] = {}
    for name in graph_inputs(body):
        datatype = body.get_tensor_datatype(name)
        if not datatype.is_integer():
            message = f"{name}: a {datatype.name} input; the testbench streams integers"
            details = (("tensor", name),)
            refused = Finding(
                FindingKind.REJECTION, "input-not-integer", "generated_frame", message, details
            )
            raise Refusal((refused,))
        shape = body.get_tensor_shape(name)
        stated = body.get_tensor_range(name)
        low, high = int(stated.least), int(stated.greatest)
        frame[name] = rng.integers(low, high + 1, size=shape, dtype=np.int64)
    return frame


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
    """Write the XSim testbench of the partition ``body`` into ``directory`` (see the
    module docstring). It streams ``frame`` (the inputs by tensor name) and expects the
    body's outputs for it, executed in Python. Like PackagePartition, it writes only the
    design step_verify_partition verified: a body that records none, or whose design changed
    since, is refused (``refuse_unverified``) before anything is written. The module is the
    one PackagePartition packages (``configured_network``), and the watchdog allows the
    cycles the network states. Nothing is simulated here (``run_testbench`` does that);
    ``toolchain`` synthesizes any HLS leaf first, unless the HLS cache holds it
    (``finn.core.rtlsim_bench.materialize``)."""
    refuse_unverified(body, label)
    point, boundary = configured_network(body, label)
    # Keep this held timed path's refusal before reference execution or file setup.
    cycles = int(point.cycles)
    context = execute_onnx(body, dict(frame), return_full_exec_context=True)
    inputs, outputs = boundary_words(body, point, boundary, context, label)
    directory.mkdir(parents=True, exist_ok=True)
    bench = stream_bench(
        point.module,
        directory,
        inputs=inputs,
        outputs=outputs,
        cycles=cycles,
        toolchain=toolchain,
    )
    (directory / "check.sv").write_text("`timescale 1ns/1ps\n" + bench.text)
    relative = [str(Path(source).relative_to(directory)) for source in bench.sources]
    script = directory / RUN_SCRIPT
    script.write_text(_run_script(module_name(point.module), relative))
    script.chmod(0o755)


def run_testbench(directory: Path, *, toolchain: Toolchain) -> str:
    """Run the testbench ``write_testbench`` wrote into ``directory`` (a build's is
    ``stitched_ip/testbench`` in its output directory): its ``run.sh``, in
    ``toolchain``'s environment with the selected command directory first on ``PATH``.
    Its output goes to a transcript as it runs: in the toolchain's log directory if it
    has one (``Toolchain.transcript``), else beside ``run.sh`` (``run.sh.stdout.log``,
    ``.stderr.log``). Returns what it printed, which ends in ``PASS``.

    Raises ``ToolUnavailable`` if the environment names no Vivado or lacks a simulator
    command (``finn.toolchain.xsim.UNAVAILABLE_STATUS``); ``ToolFailed``, naming the
    transcript, if ``xvlog`` or ``xelab`` fails (``finn.toolchain.xsim.EXIT_STATUS``);
    and ``finn.core.rtlsim_bench.SimulationFailed``, with the output, on a mismatch, any
    other failure or a timeout. A launcher route runs tools elsewhere and cannot run a
    local script, so it raises ``ValueError``."""
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
    if result.returncode == xsim.UNAVAILABLE_STATUS:
        said = result.stderr.decode(errors="replace").strip().splitlines()
        raise ToolUnavailable(
            "xsim", f"{directory / RUN_SCRIPT}: {said[-1] if said else 'not runnable here'}"
        )
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
    "run_testbench",
    "write_testbench",
]
