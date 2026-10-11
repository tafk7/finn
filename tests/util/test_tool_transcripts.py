# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A tool run's transcript (``finn.toolchain.launcher``): written by the tool as it runs,
where the caller, the build's step or the replay says; a failure names it and quotes
its error lines, read by Vivado's message parser (``finn.toolchain.vivado.messages``),
which also tells a missing licence. The tools are fakes; no Vivado, no HLS."""

from __future__ import annotations

import pytest

import os
import pickle
import subprocess
import sys
import threading
import time

from finn.toolchain import vivado
from finn.toolchain.launcher import (
    Selection,
    Toolchain,
    ToolFailed,
    run_process,
    transcript_files,
)

pytestmark = pytest.mark.util


def tool(directory, name, body):
    """An executable ``name`` in ``directory`` running the Python ``body``."""
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


def toolchain(tools):
    """A toolchain over a bare PATH, its tools under ``tools``."""
    return Toolchain(Selection(command_dir=str(tools)), {"PATH": os.defpath})


#: A tool that prints a line, then waits for the file its argument names before it
#: prints another and exits.
WAITS = """
import os, sys, time
print("first", flush=True)
print("on stderr", file=sys.stderr, flush=True)
while not os.path.exists(sys.argv[1]):
    time.sleep(0.01)
print("second", flush=True)
"""


def test_a_transcript_is_written_as_the_tool_runs(tmp_path):
    """The transcript grows while the tool runs, before it exits; what it printed is
    the result's too."""
    tool(tmp_path / "tools", "xsim", WAITS)
    go = tmp_path / "go"
    stdout, stderr = transcript_files(tmp_path / "logs" / "xsim")
    finished = []
    run = threading.Thread(
        target=lambda: finished.append(
            toolchain(tmp_path / "tools").run(
                "xsim", [str(go)], transcript=tmp_path / "logs" / "xsim", timeout=60
            )
        )
    )
    run.start()
    try:
        deadline = time.monotonic() + 30
        while "first" not in (stdout.read_text() if stdout.exists() else ""):
            assert time.monotonic() < deadline, "the transcript never grew"
            time.sleep(0.01)
        # Written while the tool still waits: not captured and written after it ended.
        assert run.is_alive() and not finished
        assert stdout.read_text() == "first\n"
    finally:
        go.touch()
        run.join(60)
    (result,) = finished
    assert stdout.read_text() == "first\nsecond\n"
    assert stderr.read_text() == "on stderr\n"
    assert (result.stdout, result.stderr) == (b"first\nsecond\n", b"on stderr\n")
    assert result.transcript == stdout


def test_where_a_runs_transcript_goes(tmp_path):
    """The caller's stem; else, in a build's step, a fresh one in its log directory,
    named after the replay or the tool; else the replay's; else none, captured."""
    tool(tmp_path / "tools", "vivado", "print('hello')\n")
    plain = toolchain(tmp_path / "tools")
    named = plain.run("vivado", transcript=tmp_path / "named")
    assert named.transcript == tmp_path / "named.stdout.log"
    beside = plain.run("vivado", cwd=tmp_path, replay=tmp_path / "make.sh")
    assert beside.transcript == tmp_path / "make.sh.stdout.log"
    captured = plain.run("vivado")
    assert (captured.transcript, captured.stdout) == (None, b"hello\n")

    logs = tmp_path / "out" / "logs" / "step_package_partition"
    step = plain.transcripts_in(logs)
    assert (step.selection, step.environment, plain.transcripts) == (
        plain.selection,
        plain.environment,
        None,
    )
    runs = [
        step.run("vivado", cwd=tmp_path, replay=tmp_path / "make.sh"),
        step.run("vivado", cwd=tmp_path, replay=tmp_path / "make.sh"),
        step.run("vivado"),
        step.run("vivado", transcript=tmp_path / "kept"),
    ]
    assert [run.transcript for run in runs] == [
        logs / "make.sh.stdout.log",
        logs / "make.sh.2.stdout.log",
        logs / "vivado.stdout.log",
        tmp_path / "kept.stdout.log",
    ]
    assert all(run.transcript.read_text() == "hello\n" for run in runs)
    assert (logs / "make.sh.2.stderr.log").read_text() == ""
    # The replay is still written where the caller says.
    assert (tmp_path / "make.sh").read_text().endswith("vivado\n")
    # The directory crosses into a worker process with the toolchain.
    assert pickle.loads(pickle.dumps(step)).transcripts == logs


def test_a_probe_keeps_no_transcript(tmp_path):
    tool(tmp_path / "tools", "vivado", "print('vivado v2031.1 (64-bit)')\n")
    logs = tmp_path / "logs"
    assert toolchain(tmp_path / "tools").transcripts_in(logs).probe("vivado") == (2031, 1)
    assert not logs.exists()


def test_a_failure_in_a_step_names_its_transcript_with_its_error_lines(tmp_path):
    tool(
        tmp_path / "tools",
        "vivado",
        "print('INFO: [Common 17-206] Exiting')\n"
        "print('CRITICAL WARNING: [Vivado 12-1411] cannot place')\n"
        "print('ERROR: [Synth 8-439] module foo not found')\n"
        "raise SystemExit(1)\n",
    )
    logs = tmp_path / "logs" / "step_build_bitfile"
    with pytest.raises(ToolFailed) as failed:
        toolchain(tmp_path / "tools").transcripts_in(logs).run(
            "vivado", cwd=tmp_path, replay=tmp_path / "synth_project.sh"
        )
    assert failed.value.transcript == logs / "synth_project.sh.stdout.log"
    assert failed.value.code == "tool-error"
    # Errors first, then the critical warnings that may not end a run.
    assert failed.value.errors == (
        "ERROR: [Synth 8-439] module foo not found",
        "CRITICAL WARNING: [Vivado 12-1411] cannot place",
    )
    assert f"transcript {logs / 'synth_project.sh.stdout.log'}" in str(failed.value)


#: Vivado's transcript of a synthesis on a device the machine holds no licence for.
UNLICENSED = """\
****** Vivado v2025.2 (64-bit)
INFO: [Common 17-206] Exiting Vivado
INFO: [Device 21-403] Loading part xcvc1902-vsva2197-2MP-e-S
WARNING: [Vivado 12-13651] The IP file has been moved
ERROR: [Common 17-345] A valid license was not found for feature 'Synthesis' and/or \
device 'xcvc1902'. Please run the Vivado License Manager for assistance.
ERROR: [Common 17-69] Command failed: Synthesis failed - please see the console
INFO: [Common 17-206] Exiting Vivado
"""


def test_the_parser_reads_vivados_messages_and_tells_a_missing_licence():
    read = vivado.messages(UNLICENSED)
    assert read.counts == {
        ("INFO", "Common 17-206"): 2,
        ("INFO", "Device 21-403"): 1,
        ("WARNING", "Vivado 12-13651"): 1,
        ("ERROR", "Common 17-345"): 1,
        ("ERROR", "Common 17-69"): 1,
    }
    assert read.first[("INFO", "Common 17-206")] == "INFO: [Common 17-206] Exiting Vivado"
    assert [line.split("]")[0] for line in read.errors] == [
        "ERROR: [Common 17-345",
        "ERROR: [Common 17-69",
    ]
    assert (read.licence_unavailable, read.code) == (True, vivado.LICENCE_UNAVAILABLE)
    # xelab's and xsim's messages are the same form, some without an ID.
    sim = vivado.messages(["ERROR: [VRFC 10-2989] 'x' is not declared", "FATAL_ERROR: Iteration"])
    assert sim.counts == {("ERROR", "VRFC 10-2989"): 1, ("FATAL_ERROR", ""): 1}
    assert (sim.licence_unavailable, sim.code) == (False, "tool-error")
    # FlexLM's words, without Vivado's ID.
    assert vivado.messages("FLEXnet Licensing error:-5,357").licence_unavailable


def test_a_licence_failure_is_classified_from_the_transcript(tmp_path):
    body = f"import sys\nsys.stdout.write({UNLICENSED!r})\nraise SystemExit(1)\n"
    tool(tmp_path / "tools", "vivado", body)
    with pytest.raises(ToolFailed) as failed:
        toolchain(tmp_path / "tools").run("vivado", transcript=tmp_path / "synth")
    assert failed.value.code == "licence-unavailable"
    assert failed.value.transcript == tmp_path / "synth.stdout.log"
    assert failed.value.errors[0].startswith("ERROR: [Common 17-345] A valid license")
    assert "(licence-unavailable); transcript " in str(failed.value)


def test_a_run_stopped_carries_what_its_transcript_holds(tmp_path):
    code = "import time; print('started', flush=True); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired) as stopped:
        run_process([sys.executable, "-c", code], timeout=2, transcript=tmp_path / "slow")
    assert stopped.value.output == b"started\n"
    assert (tmp_path / "slow.stdout.log").read_text() == "started\n"
