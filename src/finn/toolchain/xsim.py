# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What FINN knows of XSim, Vivado's simulator: its three commands and their flags,
stated once. ``xvlog`` compiles the sources with Vivado's global module (``glbl.v``),
``xelab`` elaborates a top beside it into a snapshot, ``xsim`` runs the snapshot.

``elaborate`` and ``simulate`` run them through a toolchain; ``script`` writes them as
the shell lines a testbench's ``run.sh`` replays. A driver: it builds its commands,
runs them in the toolchain's environment and reads whether they failed, and writes
nothing but its own invocation. What is compiled (the sources, less headers, and their
include directories), the top and the snapshot are the caller's.

``available`` says whether a toolchain can simulate at all.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from finn.toolchain import vivado
from finn.toolchain.launcher import StrPath, Toolchain, ToolFailed, ToolUnavailable

#: The simulator's tools, in the order a simulation runs them.
TOOLS = ("xvlog", "xelab", "xsim")

#: The libraries of Xilinx primitives a top elaborates against.
LIBRARIES = ("unisims_ver", "unimacro_ver")

#: The time scale of a module that states none.
TIMESCALE = "1ns/1ps"

#: Vivado's global module as ``script`` names it, from the installation the script's
#: environment selects.
GLBL_WORD = '"$XILINX_VIVADO/data/verilog/src/glbl.v"'

#: What a ``script`` exits with when ``xvlog`` or ``xelab`` fails, by tool; a
#: simulation that fails exits 1.
EXIT_STATUS = {"xvlog": 10, "xelab": 11}


def xelab_threads(environment: Mapping[str, str]) -> str:
    """The value of ``xelab --mt`` in ``environment``: ``FINN_XELAB_MT``, else
    ``NUM_DEFAULT_WORKERS``, else 8; one thread is ``off``.

    Xelab's own default, ``auto``, can expand to hundreds of workers on a shared
    server, and large stitched FINNLoop designs have shown intermittent elaborator
    SIGABRTs in that mode, so the default stays bounded.
    """
    threads = environment.get("FINN_XELAB_MT", environment.get("NUM_DEFAULT_WORKERS", "8"))
    return "off" if threads == "1" else threads


def available(toolchain: Toolchain) -> bool:
    """Whether ``toolchain`` selects a Vivado that provides xvlog, xelab and xsim.

    FINN images put tool shims on PATH, so a command being found does not mean a
    Vivado installation is selected: the toolchain must also name one, with the
    ``glbl.v`` a simulation elaborates (``finn.toolchain.vivado.glbl``).
    """
    try:
        vivado.glbl(toolchain)
        for tool in TOOLS:
            toolchain.command(tool)
    except ToolUnavailable:
        return False
    return True


def compile_args(
    sources: Sequence[StrPath],
    glbl: StrPath,
    *,
    includes: Sequence[StrPath] = (),
    testbench: StrPath | None = None,
) -> list[str]:
    """``xvlog``'s arguments: SystemVerilog, relaxed as FINN's own flow elaborates
    (``finn.xsi``: ``xelab -relax``), each of ``includes`` searched for headers, then
    ``sources``, ``glbl`` and the ``testbench``, if there is one."""
    return [
        "--sv",
        "--relax",
        *(f"--include={directory}" for directory in includes),
        *(str(source) for source in sources),
        str(glbl),
        *(() if testbench is None else (str(testbench),)),
    ]


def elaborate_args(top: str, snapshot: str, threads: str) -> list[str]:
    """``xelab``'s arguments: ``top`` beside the global module, against ``LIBRARIES``,
    on ``threads`` (``xelab_threads``), into ``snapshot``."""
    libraries = [word for library in LIBRARIES for word in ("-L", library)]
    return [
        f"work.{top}",
        "work.glbl",
        "--mt",
        threads,
        *libraries,
        "--snapshot",
        snapshot,
        "--timescale",
        TIMESCALE,
    ]


def simulate_args(snapshot: str, log: str | None = None) -> list[str]:
    """``xsim``'s arguments: ``snapshot`` run to its end, its log ``log`` if named
    (``xsim.log`` by default)."""
    return [snapshot, "--runall", *(() if log is None else ("--log", log))]


def elaborate(
    toolchain: Toolchain,
    directory: Path,
    sources: Sequence[StrPath],
    top: str,
    *,
    includes: Sequence[StrPath] = (),
    snapshot: str,
    testbench: StrPath | None = None,
    timeout: float | None = None,
) -> None:
    """Compile ``sources`` (and the ``testbench``) in ``directory`` and elaborate ``top``
    into ``snapshot`` there, through ``toolchain``, each tool within ``timeout`` seconds.
    Each writes its log in ``directory`` (``xvlog.log``, ``xelab.log``). ``ToolFailed``,
    naming that log, when one fails; ``ToolUnavailable`` when the toolchain selects no
    Vivado (for ``glbl.v``)."""
    glbl = vivado.glbl(toolchain)
    stages = (
        ("xvlog", "compile", compile_args(sources, glbl, includes=includes, testbench=testbench)),
        (
            "xelab",
            "elaborate",
            elaborate_args(top, snapshot, xelab_threads(toolchain.environment)),
        ),
    )
    for tool, stage, args in stages:
        result = toolchain.run(tool, args, cwd=directory, timeout=timeout, check=False)
        if result.returncode != 0:
            raise ToolFailed.of_run(
                tool, result, transcript=directory / f"{tool}.log", what=f"{top} does not {stage}"
            )


def simulate(
    toolchain: Toolchain, directory: Path, snapshot: str, *, timeout: float | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run ``snapshot``, elaborated in ``directory``, to its end through ``toolchain``
    within ``timeout`` seconds, and return the run whatever its exit status: whether a
    testbench passed (its ``$fatal``, what it displays) is the caller's verdict."""
    return toolchain.run(
        "xsim", simulate_args(snapshot), cwd=directory, timeout=timeout, check=False
    )


def script(
    sources: Sequence[StrPath],
    top: str,
    *,
    includes: Sequence[StrPath] = (),
    testbench: StrPath | None = None,
    snapshot: str,
    threads: str,
    log: str | None = None,
) -> str:
    """The shell lines that compile, elaborate and simulate as ``elaborate`` and
    ``simulate`` do, with the paths as given (relative to where the script runs) and
    ``glbl.v`` from ``$XILINX_VIVADO``, which they first require. ``xvlog`` or ``xelab``
    failing exits with its ``EXIT_STATUS``; ``xsim`` failing exits 1."""

    def words(command: Sequence[str]) -> list[str]:
        return [word if word.startswith('"$') else shlex.quote(word) for word in command]

    xvlog = ["xvlog", *compile_args(sources, GLBL_WORD, includes=includes, testbench=testbench)]
    xelab = ["xelab", *elaborate_args(top, snapshot, threads)]
    # The two long commands a word a line; the simulation on one.
    return (
        ': "${XILINX_VIVADO:?set XILINX_VIVADO (source Vivado\'s settings64.sh)}"\n'
        + " \\\n    ".join(words(xvlog))
        + f" || exit {EXIT_STATUS['xvlog']}\n"
        + " \\\n    ".join(words(xelab))
        + f" || exit {EXIT_STATUS['xelab']}\n"
        + " ".join(words(["xsim", *simulate_args(snapshot, log)]))
        + " || exit 1\n"
    )
