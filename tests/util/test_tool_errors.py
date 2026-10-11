# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A tool's two errors (``finn.toolchain.launcher``): ``ToolUnavailable`` when the
toolchain names no installation of it or has no such tool on its route, at every
reader of "which Vivado" (``finn.toolchain.vivado``, ``finn.xsi``); ``ToolFailed`` when
it ran and failed, naming its transcript. The tools are fakes; no Vivado, no HLS."""

from __future__ import annotations

import pytest

import os
import pickle
import sys

from finn.toolchain import vivado
from finn.toolchain.hls import synthesize
from finn.toolchain.launcher import (
    Selection,
    Toolchain,
    ToolFailed,
    ToolUnavailable,
    identity,
)
from finn.xsi import paths
from finn.xsi.compile import compile_sim_obj, get_simkernel_so

pytestmark = pytest.mark.util


def tool(directory, name, body):
    """An executable ``name`` in ``directory`` running the Python ``body``."""
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


def toolchain(tools=None, **environment):
    """A toolchain over a bare PATH and ``environment``, under ``tools`` if given."""
    selection = Selection(command_dir=None if tools is None else str(tools))
    return Toolchain(selection, {"PATH": os.defpath, **environment})


# -- the kinds -------------------------------------------------------------------------


def test_the_two_kinds_and_what_they_carry():
    assert issubclass(ToolUnavailable, LookupError)
    assert issubclass(ToolFailed, RuntimeError)
    assert not issubclass(ToolFailed, AssertionError)  # no verdict on the hardware
    missing = ToolUnavailable("vivado", "no installation")
    assert (missing.tool, missing.reason) == ("vivado", "no installation")
    assert str(missing).startswith("vivado: no installation; the host file selects")
    assert "build configuration's toolchain" in str(missing)
    failed = ToolFailed("xelab", 2, "/x/xelab.log", ("ERROR: one",), what="top")
    assert (
        str(failed) == "xelab failed with exit status 2: top; transcript /x/xelab.log\n  ERROR: one"
    )
    # Each crosses a process boundary as it was raised.
    for error in (missing, failed):
        copied = pickle.loads(pickle.dumps(error))
        assert type(copied) is type(error) and str(copied) == str(error)
        assert vars(copied) == vars(error)


def test_a_failure_quotes_its_first_error_lines_and_tells_a_missing_licence(tmp_path):
    log = tmp_path / "vivado.log"
    log.write_text(
        "INFO: starting\n"
        "ERROR: [Common 17-345] A valid license was not found for feature 'Synthesis'\n"
        "ERROR: [Common 17-69] Command failed\n"
    )
    failed = ToolFailed.of("vivado", "INFO: starting\n", status=1, transcript=log)
    assert failed.code == "licence-unavailable"
    assert failed.errors == (
        "ERROR: [Common 17-345] A valid license was not found for feature 'Synthesis'",
        "ERROR: [Common 17-69] Command failed",
    )
    assert "(licence-unavailable); transcript " + str(log) in str(failed)
    # No error line: what it printed last; no licence words: a tool error.
    quiet = ToolFailed.of("g++", "one\ntwo\n", status=1)
    assert (quiet.errors, quiet.code, quiet.transcript) == (("one", "two"), "tool-error", None)


# -- not on the route --------------------------------------------------------------------


def test_a_tool_not_on_the_route_is_unavailable(tmp_path):
    with pytest.raises(ToolUnavailable, match="on the PATH of its environment") as missing:
        toolchain().command("xelab")
    assert missing.value.tool == "xelab"
    with pytest.raises(ToolUnavailable, match=f"in the command directory {tmp_path}"):
        toolchain(tmp_path).run("vivado")
    with pytest.raises(ToolUnavailable, match="XILINX_HLS or XILINX_VITIS") as missing:
        toolchain().hls_installation()
    assert missing.value.tool == "HLS"


def test_a_tool_that_exits_non_zero_is_failed_naming_its_transcript(tmp_path):
    tool(tmp_path / "tools", "vivado", "print('ERROR: it broke'); raise SystemExit(3)\n")
    with pytest.raises(ToolFailed) as failed:
        toolchain(tmp_path / "tools").run(
            "vivado", cwd=tmp_path, replay=tmp_path / "run.sh", transcript=tmp_path / "own.log"
        )
    assert (failed.value.status, failed.value.errors) == (3, ("ERROR: it broke",))
    assert failed.value.transcript == tmp_path / "own.log"  # the tool's own log, if named
    assert (tmp_path / "run.sh.stdout.log").read_text() == "ERROR: it broke\n"
    result = toolchain(tmp_path / "tools").run("vivado", check=False)
    assert result.returncode == 3


# -- one Vivado lookup -------------------------------------------------------------------


def test_the_installation_and_its_glbl_or_unavailable(tmp_path, vivado_installation):
    with pytest.raises(ToolUnavailable, match=r"names no Vivado installation \(XILINX_VIVADO\)"):
        vivado.installation(toolchain())
    with pytest.raises(ToolUnavailable, match="is no directory"):
        vivado.installation(toolchain(XILINX_VIVADO=str(tmp_path / "absent")))
    with pytest.raises(ToolUnavailable, match="glbl.v") as missing:
        vivado.glbl(toolchain(XILINX_VIVADO=str(tmp_path)))
    assert missing.value.tool == "vivado"
    installed = vivado_installation(tmp_path / "Vivado")
    selected = toolchain(XILINX_VIVADO=str(installed))
    assert vivado.installation(selected) == installed
    assert vivado.glbl(selected) == installed / "data/verilog/src/glbl.v"


def test_the_release_is_the_probes_or_unavailable(tmp_path):
    tool(tmp_path / "tools", "vivado", "print('vivado v2025.2 (64-bit)')\n")
    assert vivado.release(toolchain(tmp_path / "tools")) == (2025, 2)
    with pytest.raises(ToolUnavailable, match="not on this toolchain's route") as missing:
        vivado.release(toolchain(tmp_path / "none"))
    assert missing.value.tool == "vivado"
    tool(tmp_path / "broken", "vivado", "raise SystemExit(139)\n")
    with pytest.raises(ToolUnavailable, match="could not probe it through the local route"):
        vivado.release(toolchain(tmp_path / "broken"))
    tool(tmp_path / "mute", "vivado", "print('no release here')\n")
    with pytest.raises(ToolUnavailable, match="no recognizable AMD release"):
        vivado.release(toolchain(tmp_path / "mute"))


@pytest.mark.parametrize(
    "banner, library",
    [("v2024.1", "librdi_simulator_kernel.so"), ("v2024.2", "libxv_simulator_kernel.so")],
)
def test_the_simulation_kernel_follows_the_release(tmp_path, banner, library):
    tool(tmp_path / "tools", "vivado", f"print('vivado {banner}')\n")
    assert get_simkernel_so(toolchain(tmp_path / "tools")) == library


def test_xsi_reads_vivado_through_the_toolchain(tmp_path, monkeypatch):
    """No ``XILINX_VIVADO`` of this process reaches XSI: a toolchain naming none is
    unavailable, whatever the process's environment says."""
    monkeypatch.setenv("XILINX_VIVADO", str(tmp_path))
    monkeypatch.delenv("FINN_XSI_BUILD_DIR", raising=False)
    bare = toolchain(tmp_path / "tools")
    source = tmp_path / "top.v"
    source.write_text("module top(); endmodule\n")
    for reads in (
        lambda: compile_sim_obj("top", [source], tmp_path, toolchain=bare),
        lambda: paths.xsi_artifact_dir(bare),
        lambda: identity(bare, "xsim"),
    ):
        with pytest.raises(ToolUnavailable, match="XILINX_VIVADO") as missing:
            reads()
        assert missing.value.tool == "vivado"


def test_an_installation_is_identified_by_what_it_is(tmp_path, vivado_installation):
    one = vivado_installation(tmp_path / "a/Vivado", "2025.2")
    other = vivado_installation(tmp_path / "b/Vivado", "2025.2")
    hls = tmp_path / "hls"
    (hls / "data").mkdir(parents=True)
    (hls / "data/version.sh").write_text("XILINX_VERSION_DEFAULT=2025.2\n")
    (hls / "data/version.dat").write_text("HLS\n")

    def of(installation, tool):
        return identity(toolchain(XILINX_VIVADO=str(installation), XILINX_HLS=str(hls)), tool)

    assert of(one, "xelab") == of(other, "xelab")  # never where it is installed
    assert of(one, "xelab").startswith("xelab 2025.2 ")
    # An HLS frontend's is the HLS installation's, as the HLS cache keyed it before.
    assert of(one, "vitis-run").startswith("vitis-run 2025.2 ")
    assert of(one, "vitis-run").split()[2] != of(one, "xelab").split()[2]


# -- HLS ---------------------------------------------------------------------------------


def test_a_failed_synthesis_is_the_frontends_failure_naming_its_transcript(tmp_path):
    tools = tmp_path / "tools"
    tool(
        tools,
        "vitis_hls",
        "import sys\n"
        "if sys.argv[1:] == ['-version']: print('vitis_hls v2024.2'); raise SystemExit\n"
        "print('ERROR: [HLS 200-70] no top'); raise SystemExit(1)\n",
    )
    (tmp_path / "hls").mkdir()
    selected = Toolchain(
        Selection(command_dir=str(tools), hls_frontend="vitis_hls"),
        {"PATH": os.defpath, "XILINX_HLS": str(tmp_path / "hls")},
    )
    request = tmp_path / "request"
    request.mkdir()
    (request / "script.tcl").write_text("exit\n")
    with pytest.raises(ToolFailed) as failed:
        synthesize(request, "xc7z020clg400-1", name="top", toolchain=selected, cache=tmp_path / "c")
    assert (failed.value.tool, failed.value.status) == ("vitis_hls", 1)
    assert failed.value.errors == ("ERROR: [HLS 200-70] no top",)
    # The build is kept, and its transcript is named where it is kept.
    (kept,) = (tmp_path / "c").glob("top-*.failed")
    assert failed.value.transcript == kept / "synthesize.sh.stdout.log"
    assert failed.value.transcript.read_text() == "ERROR: [HLS 200-70] no top\n"
    with pytest.raises(ValueError, match="not a staged request"):
        synthesize(tmp_path, "xc7z020clg400-1", name="top", toolchain=selected)
