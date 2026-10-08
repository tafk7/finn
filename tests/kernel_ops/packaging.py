# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging without Vivado: a toolchain double that stops PackagePartition where Vivado
would start, so a test reads what packaging checks before then; one that stands in for
Vivado's packaging, so a test reads what PackagePartition writes beside the IP; and one
that fakes the pynq shell's Vivado project. ``read_back`` checks an interface
description against a module's ABI pins; ``io_shape_dict`` reads a generated driver's
I/O without pynq, and ``bitfile_default`` runs its arguments' parser with pynq stubbed."""

from __future__ import annotations

import ast
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
from finn.transformation.kernels import PackagePartition
from finn.util.toolchain import Selection, Toolchain


class ReachedVivado(Exception):
    """Raised by ``NoVivado`` when packaging gets as far as running a tool."""


class NoVivado:
    """A toolchain double (packaging calls ``run`` only): it records the first tool
    packaging runs, and stops there."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, tool: str, *args: object, **options: object) -> None:
        self.ran.append(tool)
        raise ReachedVivado(tool)


def reaches_vivado(model: ModelWrapper, name: str, project: Path) -> None:
    """Package ``model`` against ``NoVivado``: its emitted top elaborates, and packaging
    gets as far as running Vivado."""
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
    + _placed_row("    partition", 400, 800, 1, 1, 0, 4)
    + _placed_row("    axi_interconnect_0", 100, 200, 0, 0, 0, 0)
    + _placed_row("    idma0", 100, 250, 1, 0, 0, 0)
    + _placed_row("    odma0", 90, 240, 0, 0, 0, 0)
    + "</table></section></RptDoc>"
)


class FakeVivado:
    """A toolchain double for the pynq shell's runner (``pynq_runner.build_pynq``): its
    Vivado run keeps the project's Tcl and makes what the template's project makes
    (unless ``makes`` is False), each file naming itself, the routed timing summary
    ``timing`` if given. ``selection`` is the toolchain's (``Selection.vivado_jobs``)."""

    def __init__(self, makes: bool = True, timing: str | None = None) -> None:
        self.runs: list[tuple[str, list[str], str]] = []
        self.makes = makes
        self.timing = timing
        self.selection = Selection()

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
            for name in ("idma0", "partition", "smartconnect_0")
        ]
        for path in made + synthesized:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)
        for index, path in enumerate(synthesized):
            path.write_text(UTILIZATION_SYNTH.format(lut=100 * (index + 1), ff=10))
        made[3].write_text(PLACED_HIERARCHY)
        if self.timing is not None:
            made[1].write_text(self.timing)


def io_shape_dict(driver: str) -> dict[str, Any]:
    """The generated driver's io_shape_dict, read without running it (it imports pynq)."""
    tree = ast.parse(driver)
    (value,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [getattr(target, "id", None) for target in node.targets] == ["io_shape_dict"]
    ]
    shapes: dict[str, Any] = eval(
        compile(ast.Expression(value), "driver", "eval"), {"DataType": DataType}
    )
    return shapes


#: What a generated driver imports of pynq, stubbed: enough to parse its arguments.
PYNQ_STUB = {
    "pynq/__init__.py": "class Overlay:\n    pass\n\n\nallocate = None\n",
    "pynq/ps.py": "Clocks = None\n",
    "pynq/pl_server/__init__.py": "",
    "pynq/pl_server/device.py": "class Device:\n    devices = []\n",
}


def bitfile_default(script: Path, cwd: Path) -> str:
    """The bitfile a generated driver script (driver.py, validate.py) runs unless told
    another, as its --help names it: the script run from ``cwd``, pynq stubbed."""
    stub = cwd / "pynq_stub"
    for name, text in PYNQ_STUB.items():
        (stub / name).parent.mkdir(parents=True, exist_ok=True)
        (stub / name).write_text(text)
    env = {
        **os.environ,
        "PYTHONPATH": str(stub),
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
    (default,) = re.findall(r"--bitfile BITFILE\s+the bitfile to run \(default: (\S+)\)", shown)
    return default


class PackagedByStub:
    """A toolchain double for Vivado's packaging: it writes ``ip/component.xml``, as the
    emitted-text tool's stub does, so PackagePartition completes without Vivado."""

    def run(self, tool: str, args: object, *, cwd: Path, **options: object) -> None:
        (Path(cwd) / "ip").mkdir(parents=True, exist_ok=True)
        (Path(cwd) / "ip" / "component.xml").write_text("<stub/>")


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
    "io_shape_dict",
    "reaches_vivado",
    "read_back",
]
