# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging without Vivado: a partition verified as the build verifies it before
emission (``verified``, ``verified_parent``), which packaging, the shell's integration
and the testbench ask for; a toolchain double that stops PackagePartition where Vivado
would start, so a test reads what packaging checks before then; one that stands in for
Vivado's packaging, so a test reads what PackagePartition writes beside the IP; and one
that fakes the pynq shell's Vivado project. ``read_back`` checks an interface
description against a module's ABI pins, and ``bitfile_default`` runs a driver's arguments'
parser with pynq stubbed."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.partition import partition_body
from finn.custom_op.kernels.shell import verify_partition
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Endpoint,
    Pin,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.shells.integration import Timing
from finn.toolchain.launcher import Selection, Toolchain
from finn.toolchain.vivado import Check
from finn.transformation.kernels import PackagePartition
from support.pynq_stub import stub

#: A toolchain double's environment: a Vivado installation directory with no Vivado in
#: it, whose identity the packaged-IP store's key reads as unstated.
NO_VIVADO = {"XILINX_VIVADO": str(Path(__file__).parent)}


def verified(model: ModelWrapper, name: str) -> ModelWrapper:
    """``model``, a partition's body, its module named ``name`` (its node's), verified as
    the build verifies it before emission
    (``verify_partition``, not the build's step: its FIFOs checked, the design it verified
    recorded on it), so that packaging, the shell's integration and the testbench take it;
    the model itself. A test that changes the body after it verifies it again."""
    verify_partition(model, name)
    return model


def verified_parent(parent: ModelWrapper) -> ModelWrapper:
    """The build's parent graph ``parent``, its partition's body verified (``verified``,
    named as its node) and saved in its file, as step_verify_partition leaves it; the
    parent itself."""
    node, body, body_file = partition_body(parent)
    verified(body, node.name)
    body.save(body_file)
    return parent


class ReachedVivado(Exception):
    """Raised by ``NoVivado`` when packaging gets as far as running a tool."""


class NoVivado:
    """A toolchain double (packaging calls ``run`` only): it records the first tool
    packaging runs, and stops there."""

    environment = NO_VIVADO

    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, tool: str, *args: object, **options: object) -> None:
        self.ran.append(tool)
        raise ReachedVivado(tool)


def reaches_vivado(model: ModelWrapper, name: str, project: Path) -> None:
    """Package ``model``, verified first (``verified``), against ``NoVivado``: its emitted
    top elaborates, and packaging gets as far as running Vivado."""
    verified(model, name)
    toolchain = NoVivado()
    with pytest.raises(ReachedVivado):
        model.transform(
            PackagePartition(name, directory=project, toolchain=cast(Toolchain, toolchain))
        )
    assert toolchain.ran == ["vivado"]


#: A synthesized IP's flat utilization report (``<run>_utilization_synth.rpt``), Vivado's
#: format with invented counts; ``lut`` and ``ff`` to fill.
UTILIZATION_SYNTH = """\
1. CLB Logic
------------

+----------------------------+-------+-------+------------+-----------+-------+
|          Site Type         |  Used | Fixed | Prohibited | Available | Util% |
+----------------------------+-------+-------+------------+-----------+-------+
| CLB LUTs*                  | {lut} |     0 |          0 |     70560 |  1.80 |
|   LUT as Logic             | {lut} |     0 |          0 |     70560 |  1.71 |
| CLB Registers              | {ff}  |     0 |          0 |    141120 |  1.57 |
+----------------------------+-------+-------+------------+-----------+-------+

2. BLOCKRAM
-----------

+-------------------+------+-------+------------+-----------+-------+
|     Site Type     | Used | Fixed | Prohibited | Available | Util% |
+-------------------+------+-------+------------+-----------+-------+
| Block RAM Tile    |  2.5 |     0 |          0 |       216 |  1.16 |
|   RAMB36/FIFO*    |    2 |     0 |          0 |       216 |  0.93 |
|   RAMB18          |    1 |     0 |          0 |       432 |  0.23 |
+-------------------+------+-------+------------+-----------+-------+

3. ARITHMETIC
-------------

+-----------+------+-------+------------+-----------+-------+
| Site Type | Used | Fixed | Prohibited | Available | Util% |
+-----------+------+-------+------------+-----------+-------+
| DSPs      |    3 |     0 |          0 |       360 |  0.83 |
+-----------+------+-------+------------+-----------+-------+

8. Primitives
-------------

+----------+------+---------------------+
| Ref Name | Used | Functional Category |
+----------+------+---------------------+
| RAMB18E2 |    1 |            BLOCKRAM |
+----------+------+---------------------+
"""


def _placed_row(name: str, *counts: int) -> str:
    cells = [name, "m", *(str(count) for count in counts)]
    return (
        "<tablerow>" + "".join(f'<tablecell contents="{cell}"/>' for cell in cells) + "</tablerow>"
    )


_PLACED_COLUMNS = (
    "Instance",
    "Module",
    "Total LUTs",
    "FFs",
    "RAMB36",
    "RAMB18",
    "URAM",
    "DSP Blocks",
)

#: The routed design's hierarchical utilization (``synth_report.xml``), Vivado's XML
#: format with invented counts: the processor and its reset are not listed.
PLACED_HIERARCHY = (
    '<RptDoc><section title="Utilization by Hierarchy"><table><tablerow>'
    + "".join(f'<tableheader contents="{name}"/>' for name in _PLACED_COLUMNS)
    + "</tablerow>"
    + _placed_row("top_wrapper", 1000, 2000, 2, 1, 0, 4)
    + _placed_row("  top_i", 1000, 2000, 2, 1, 0, 4)
    + _placed_row("    smartconnect_0", 300, 500, 0, 0, 0, 0)
    + _placed_row("      inst", 300, 500, 0, 0, 0, 0)
    + _placed_row("    DataflowCore_0", 400, 800, 1, 1, 0, 4)
    + _placed_row("    control_0_0", 100, 200, 0, 0, 0, 0)
    + _placed_row("    in0", 100, 250, 1, 0, 0, 0)
    + _placed_row("    out0", 90, 240, 0, 0, 0, 0)
    + "</table></section></RptDoc>"
)


#: Real routed timing reports (``tests/shells/recorded``), each its summary tables, the
#: report cut at its timing details, the header's host line dropped and trailing spaces
#: stripped:
#: - ``TIMING_REPORT``: TFC's writable build on Ultra96 (shell landing 1's full tier, Vivado
#:   2025.2), 5 ns asked, its PS's PL clock routed at 5.333 ns (187.512 MHz), its worst
#:   setup slack 1.667 ns;
#: - ``UNMET_REPORT``: a registered 16x16 multiply-accumulate routed on Ultra96's part at
#:   1 ns, which it misses: its worst setup slack -0.906 ns, reported with the options of
#:   the ``pynq`` template's routed report (Vivado 2025.2); its clock named ``clk_pl_0``, as
#:   the PS names the PL clock, though no block design;
#: - ``UNMET_SLASHKIT_REPORT``: the same routed design, reported as ``slashkit`` reports
#:   its routed design (``slash_project_build.tcl`` at SLASH 26f9c55c); no ``slash``
#:   hardware build has run.
#: Each report's ``-file`` path is cut to its name.
RECORDED = Path(__file__).parent.parent / "shells" / "recorded"
TIMING_REPORT = (RECORDED / "ultra96-tfc-timing.rpt").read_text()
UNMET_REPORT = (RECORDED / "unmet-pynq-timing.rpt").read_text()
UNMET_SLASHKIT_REPORT = (RECORDED / "unmet-slashkit-timing.txt").read_text()

#: ``TIMING_REPORT`` as the ``pynq`` linker admits it (``finn.shells.pynq.linker.TIMING``
#: its report's key): its PL clock's period, each of its design's checks.
TFC_TIMING = Timing(
    5.333,
    {
        "setup": Check(1.667, 0.0, 0, 63298),
        "hold": Check(0.014, 0.0, 0, 63298),
        "pulse_width": Check(1.166, 0.0, 0, 19432),
    },
    "post_route_timing.rpt",
)

#: ``TFC_TIMING`` as a bitfile's build records it (``finn.shells.build.timing_record``).
TFC_TIMING_RECORD = {
    "period_ns": 5.333,
    "setup": {"worst_ns": 1.667, "total_ns": 0.0, "failing_endpoints": 0, "endpoints": 63298},
    "hold": {"worst_ns": 0.014, "total_ns": 0.0, "failing_endpoints": 0, "endpoints": 63298},
    "pulse_width": {
        "worst_ns": 1.166,
        "total_ns": 0.0,
        "failing_endpoints": 0,
        "endpoints": 19432,
    },
    "report": "report/post_route_timing.rpt",
}


class FakeVivado:
    """A toolchain double for the pynq shell's linker (``finn.shells.pynq.linker``): its
    Vivado run keeps the project's Tcl and makes what the template's project makes
    (unless ``makes`` is False), each file naming itself, the routed timing summary
    ``timing`` (by default TFC's on Ultra96, ``TIMING_REPORT``; None, none).
    ``selection`` is the toolchain's (``Selection.vivado_jobs``)."""

    environment = NO_VIVADO

    def __init__(self, makes: bool = True, timing: str | None = TIMING_REPORT) -> None:
        self.runs: list[tuple[str, list[str], str]] = []
        self.makes = makes
        self.timing = timing
        self.selection = Selection()

    def transcripts_in(self, directory: object) -> FakeVivado:
        """Itself: its runs keep no transcript."""
        return self

    def run(self, tool: str, args: list[str], *, cwd: str, **options: object) -> None:
        self.runs.append((tool, args, (Path(cwd) / args[-1]).read_text()))
        if not self.makes:
            return
        root = Path(cwd)
        runs = root / "finn_zynq_link.runs"
        made = [
            runs / "impl_1" / "top_wrapper.bit",
            runs / "impl_1" / "top_wrapper_timing_summary_routed.rpt",
            root / "finn_zynq_link.gen" / "sources_1" / "bd" / "top" / "hw_handoff" / "top.hwh",
            root / "synth_report.xml",
        ]
        synthesized = [
            runs / f"top_{name}_0_synth_1" / f"top_{name}_0_utilization_synth.rpt"
            for name in ("in0", "DataflowCore_0", "smartconnect_0")
        ]
        for path in made + synthesized:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)
        for index, path in enumerate(synthesized):
            path.write_text(UTILIZATION_SYNTH.format(lut=100 * (index + 1), ff=10))
        made[3].write_text(PLACED_HIERARCHY)
        if self.timing is None:
            made[1].unlink()
        else:
            made[1].write_text(self.timing)


def bitfile_default(script: Path, cwd: Path) -> str:
    """The bitfile a driver's script (driver.py, validate.py) runs unless told another, as
    its --help names it: the script run from ``cwd``, pynq stubbed
    (``support.pynq_stub``)."""
    stubbed = stub(cwd / "pynq_stub")
    env = {
        **os.environ,
        "PYTHONPATH": str(stubbed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "COLUMNS": "10000",
    }
    shown = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = re.search(r"--bitfile BITFILE\s+the bitfile to run \(default: (\S+)\)", shown)
    assert found is not None, shown
    return found.group(1)


class PackagedByStub:
    """A toolchain double for Vivado's packaging: it writes ``ip/component.xml``, as the
    emitted-text tool's stub does, so PackagePartition completes without Vivado."""

    environment = NO_VIVADO

    def transcripts_in(self, directory: object) -> PackagedByStub:
        """Itself: its runs keep no transcript."""
        return self

    def run(
        self, tool: str, args: object, *, cwd: Path, **options: object
    ) -> subprocess.CompletedProcess[bytes]:
        (Path(cwd) / "ip").mkdir(parents=True, exist_ok=True)
        (Path(cwd) / "ip" / "component.xml").write_text("<stub/>")
        return subprocess.CompletedProcess([tool], 0, b"", b"")


def read_back(described: dict[str, Any], pins: Sequence[Pin], period_ns: float) -> None:
    """Check an interface description against the module's ABI pins: every pin is stated
    once, in its section, with its direction, widths, clock and rate, and nothing else."""
    base = round(1e9 / period_ns)
    clocks = {clock["name"]: clock for clock in described["clocks"]}
    resets = {reset["name"]: reset for reset in described["resets"]}
    streams = {stream["name"]: stream for stream in described["streams"]}
    buses = {bus["name"]: bus for bus in described["axilite"]}
    seen = []
    for pin in pins:
        seen.append(pin.name)
        if isinstance(pin, Signal) and isinstance(pin.role, Clock):
            rate = pin.role.rate
            ratio = rate.ratio if isinstance(rate, Derived) else 1
            assert clocks[pin.name]["freq_hz"] == ratio * base, pin.name
        elif isinstance(pin, Signal) and isinstance(pin.role, Reset):
            polarity = "ACTIVE_LOW" if pin.role.active_low else "ACTIVE_HIGH"
            assert resets[pin.name]["polarity"] == polarity
        elif isinstance(pin, Bus):
            widths = {member.logical: member.width for member in pin.signals}
            if pin.protocol is StandardProtocol.AXIS:
                stream = streams[pin.name]
                direction = "in" if pin.endpoint is Endpoint.TARGET else "out"
                assert (stream["direction"], stream["tdata"]) == (direction, widths["tdata"])
                assert stream["clock"] == pin.associated_clock
                assert stream["lanes"] * DataType[stream["element"]].bitwidth() <= stream["tdata"]
            else:
                assert buses[pin.name]["address_width"] == widths["awaddr"]
        else:
            raise AssertionError(f"{pin.name}: a pin no section describes")
    described_names = [*clocks, *resets, *streams, *buses]
    described_names += [port["name"] for port in described["aximm"]]
    assert sorted(described_names) == sorted(seen)


__all__ = [
    "FakeVivado",
    "NoVivado",
    "PLACED_HIERARCHY",
    "PackagedByStub",
    "ReachedVivado",
    "UTILIZATION_SYNTH",
    "reaches_vivado",
    "read_back",
    "verified",
    "verified_parent",
]
