# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The packaged-IP store (``finn.toolchain.ip``): Vivado's packaging of a staged project,
keyed by what Vivado is given, published once and copied into each build; and the HLS
cache's key for a request that reads a library in place (``finn.toolchain.hls``). The
tools are fakes; no Vivado, no HLS."""

from __future__ import annotations

import pytest

import logging
import os
import shutil
import stat
import sys
from pathlib import Path

from finn.core.reporting.manifest import RECORDS, STORED
from finn.resources._store import MARKER
from finn.toolchain.hls import synthesis_key
from finn.toolchain.ip import packaged, packaging_key, place
from finn.toolchain.launcher import Selection, Toolchain, ToolFailed

pytestmark = pytest.mark.util

#: A Vivado that packages as package.tcl would: it notes each run in ``runs`` beside
#: itself, makes a project, and an IP whose description names an include directory of
#: the staged sources by absolute path, as Vivado's packager records one.
FAKE_VIVADO = """
import os, sys
if sys.argv[1:] == ["-version"]:
    print("vivado v2025.2"); sys.exit()
with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "runs"), "a") as log:
    log.write(os.getcwd() + "\\n")
os.makedirs("project/project.runs")
os.makedirs("ip/src")
open("ip/src/top.sv", "w").write("module top; endmodule\\n")
here = os.path.realpath(os.getcwd())
open("ip/component.xml", "w").write(
    "<spirit:component>\\n"
    f"  <spirit:dependency>{here}/src/rtl</spirit:dependency>\\n"
    "  <spirit:dependency>/opt/elsewhere</spirit:dependency>\\n"
    "</spirit:component>\\n"
)
print("INFO: [IP_Flow 19-234] packaging")
"""


def vivado(directory, body=FAKE_VIVADO, release="2025.2"):
    """A toolchain whose Vivado is ``body`` in ``directory``, its installation of
    ``release``: the toolchain."""
    tools = directory / "tools"
    tools.mkdir(parents=True)
    (tools / "vivado").write_text("#!" + sys.executable + "\n" + body)
    (tools / "vivado").chmod(0o755)
    installation = directory / "Vivado"
    (installation / "data").mkdir(parents=True)
    (installation / "data" / "version.sh").write_text(f"XILINX_VERSION_DEFAULT={release}\n")
    (installation / "data" / "version.dat").write_text(f"Vivado {release}\n")
    return Toolchain(
        Selection(command_dir=str(tools)),
        {"PATH": os.defpath, "XILINX_VIVADO": str(installation)},
    )


def staged(directory, top="module top; endmodule\n"):
    """A staged project in ``directory``: its script and its sources."""
    (directory / "src" / "rtl").mkdir(parents=True)
    (directory / "src" / "top.sv").write_text(top)
    (directory / "src" / "rtl" / "pkg.svh").write_text("`define W 8\n")
    (directory / "package.tcl").write_text("set project [file dirname [info script]]\n")
    return directory


def runs(toolchain):
    """The directories the fake Vivado ran in, in order."""
    log = Path(toolchain.selection.command_dir) / "runs"
    return log.read_text().split() if log.exists() else []


def test_a_project_is_packaged_once_and_its_ip_can_move(tmp_path):
    """The first request packages in a temporary directory and publishes the entry; a
    second, of the same project staged elsewhere, is the store's: no Vivado. The entry
    keeps the IP, the project and the replay, not Vivado's own project; the include
    directory Vivado recorded inside the project is relative to the IP, and every file is
    read-only."""
    toolchain = vivado(tmp_path)
    store = tmp_path / "store"
    first = packaged(
        staged(tmp_path / "a"), name="sdp", run_synth=False, toolchain=toolchain, store=store
    )
    assert first.directory == store / f"sdp-{first.key.removeprefix('sha256:')[:16]}"
    assert (first.directory / MARKER).read_text() == first.key + "\n"
    assert len(runs(toolchain)) == 1 and runs(toolchain)[0] != str(first.directory)
    assert sorted(path.name for path in first.directory.iterdir()) == [
        MARKER,
        "ip",
        "package.sh",
        "package.sh.stderr.log",
        "package.sh.stdout.log",
        "package.tcl",
        "src",
    ]
    assert (first.ip / "component.xml").read_text() == (
        "<spirit:component>\n"
        "  <spirit:dependency>../src/rtl</spirit:dependency>\n"
        "  <spirit:dependency>/opt/elsewhere</spirit:dependency>\n"
        "</spirit:component>\n"
    )
    assert 'cd -- "$(dirname -- "$0")"\n' in (first.directory / "package.sh").read_text()
    for path in first.directory.rglob("*"):
        if path.is_file() and path.name != MARKER:
            assert not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    again = packaged(
        staged(tmp_path / "b"), name="sdp", run_synth=False, toolchain=toolchain, store=store
    )
    assert (again.directory, again.key) == (first.directory, first.key)
    assert first.built and not again.built and len(runs(toolchain)) == 1


def test_each_use_of_the_store_is_recorded_built_or_reused(tmp_path, caplog):
    """The running step's manifest lists each entry the store gives, built by this call
    or reused (``finn.core.reporting.manifest.stored``)."""
    toolchain = vivado(tmp_path)
    store = tmp_path / "store"
    with caplog.at_level(logging.INFO, logger=RECORDS.name):
        for where in ("a", "b"):
            entry = packaged(
                staged(tmp_path / where),
                name="sdp",
                run_synth=False,
                toolchain=toolchain,
                store=store,
            )
    recorded = [getattr(record, STORED) for record in caplog.records if hasattr(record, STORED)]
    assert recorded == [
        {"store": "ip", "entry": str(entry.directory), "key": entry.key, "status": status}
        for status in ("built", "reused")
    ]


def test_the_key_is_what_vivado_is_given(tmp_path):
    """The project's content, Vivado's identity and run_synth; never where the project
    is."""
    toolchain = vivado(tmp_path)
    key = packaging_key(staged(tmp_path / "a"), run_synth=False, toolchain=toolchain)
    assert key.startswith("sha256:") and len(key) == len("sha256:") + 64
    assert packaging_key(staged(tmp_path / "b"), run_synth=False, toolchain=toolchain) == key
    assert packaging_key(tmp_path / "a", run_synth=True, toolchain=toolchain) != key
    other = staged(tmp_path / "c", top="module top; wire w; endmodule\n")
    assert packaging_key(other, run_synth=False, toolchain=toolchain) != key
    newer = vivado(tmp_path / "newer", release="2026.1")
    assert packaging_key(tmp_path / "a", run_synth=False, toolchain=newer) != key


def test_an_entry_is_placed_by_links_replacing_what_it_names(tmp_path):
    """``place`` links the entry's files into the build's directory, replacing what the
    directory holds of the same names, and leaves the rest; the marker stays the
    store's."""
    toolchain = vivado(tmp_path)
    stored = packaged(
        staged(tmp_path / "a"),
        name="sdp",
        run_synth=False,
        toolchain=toolchain,
        store=tmp_path / "store",
    )
    build = tmp_path / "build"
    (build / "src").mkdir(parents=True)
    (build / "src" / "stale.sv").write_text("an earlier packaging's")
    (build / "package.tcl").write_text("an earlier packaging's")
    (build / "testbench").mkdir()
    place(stored.directory, build)
    assert sorted(path.name for path in build.iterdir()) == [
        "ip",
        "package.sh",
        "package.sh.stderr.log",
        "package.sh.stdout.log",
        "package.tcl",
        "src",
        "testbench",
    ]
    assert not (build / "src" / "stale.sv").exists()
    assert (build / "ip" / "component.xml").samefile(stored.ip / "component.xml")
    assert (build / "package.tcl").samefile(stored.directory / "package.tcl")
    # Placed again, as a second packaging into the same directory does.
    place(stored.directory, build)
    assert (build / "src" / "top.sv").samefile(stored.directory / "src" / "top.sv")


def test_a_failed_packaging_is_kept_beside_the_entry_naming_its_transcript(tmp_path):
    failing = 'import sys\nprint("ERROR: [Common 17-1] no")\nsys.exit(1)\n'
    toolchain = vivado(tmp_path, failing)
    store = tmp_path / "store"
    with pytest.raises(ToolFailed) as failed:
        packaged(
            staged(tmp_path / "a"), name="sdp", run_synth=False, toolchain=toolchain, store=store
        )
    (kept,) = store.glob("sdp-*.failed")
    assert (failed.value.tool, failed.value.status) == ("vivado", 1)
    assert failed.value.transcript == kept / "package.sh.stdout.log"
    assert failed.value.errors == ("ERROR: [Common 17-1] no",)
    assert [path.name for path in store.iterdir() if not path.name.startswith(".")] == [kept.name]


def test_an_ip_that_names_where_it_was_packaged_is_refused(tmp_path):
    """An absolute path of the project anywhere but an include directory: the IP could not
    move, so it is not stored."""
    named = FAKE_VIVADO + 'open("ip/component.xml", "a").write(f"<x>{here}/src/top.sv</x>")\n'
    store = tmp_path / "store"
    with pytest.raises(ToolFailed, match="names the directory it was packaged in"):
        packaged(
            staged(tmp_path / "a"),
            name="sdp",
            run_synth=False,
            toolchain=vivado(tmp_path, named),
            store=store,
        )
    assert not [
        path
        for path in store.iterdir()
        if path.name.startswith("sdp-") and not path.name.endswith(".failed")
    ]


def test_a_directory_without_its_script_is_no_staged_project(tmp_path):
    with pytest.raises(ValueError, match="holds no package.tcl"):
        packaged(
            tmp_path, name="sdp", run_synth=False, toolchain=vivado(tmp_path), store=tmp_path / "s"
        )


def test_a_library_enters_the_hls_key_by_its_content_not_its_location(tmp_path):
    """A request that reads a library in place (an IODMA, finn-hlslib): two copies of the
    library give one key, a changed one another; so does the part."""
    request = tmp_path / "request"
    request.mkdir()
    (request / "script.tcl").write_text("source $hlslib/x.tcl\n")
    library = tmp_path / "hlslib"
    library.mkdir()
    (library / "dma.h").write_text("// dma\n")
    copy = tmp_path / "elsewhere" / "hlslib"
    shutil.copytree(library, copy)
    part, identity = "xc7z020clg400-1", "vitis_hls 2024.2 0"
    key = synthesis_key(request, part, identity, {"hlslib": library})
    assert synthesis_key(request, part, identity, {"hlslib": copy}) == key
    assert synthesis_key(request, "xczu3eg-sbva484-1-e", identity, {"hlslib": copy}) != key
    (copy / "dma.h").write_text("// dma, changed\n")
    assert synthesis_key(request, part, identity, {"hlslib": copy}) != key
    # A request that reads no library keys as before.
    assert synthesis_key(request, part, identity) == synthesis_key(request, part, identity, {})
