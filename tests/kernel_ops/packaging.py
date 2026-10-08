# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging without Vivado: a toolchain double that stops PackagePartition where Vivado
would start, so a test reads what packaging checks before then; and one that fakes the
pynq shell's Vivado project; and a generated driver's I/O, read without pynq."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

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
        made += [
            runs / f"top_{name}_0_synth_1" / f"top_{name}_0_utilization_synth.rpt"
            for name in ("idma0", "partition", "smartconnect")
        ]
        for path in made:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)
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


__all__ = ["FakeVivado", "NoVivado", "ReachedVivado", "io_shape_dict", "reaches_vivado"]
