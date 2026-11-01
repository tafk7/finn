# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two cores in one build: a parent graph of two TFC bodies side by side, made by hand
(``support.two_cores``), through the description (``describe``), the ``pynq`` linker's
block design and the runtime file, each against its recorded text (``recorded/``). The cut
makes one core today; the description, the linker and the runtime file take several."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.two_cores import PREFIX, two_tfc_cores

import finn.custom_op.kernels.shell as kernel_shell
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import core_bodies
from finn.custom_op.kernels.shell import configured_network
from finn.platform.shells import shell_row
from finn.shells.integration import MAIN, Integration, IpRef, Linked, describe, runtime_file
from finn.shells.pynq.linker import block_design, linked_row, unit_bases

RECORDED = Path(__file__).parent / "recorded"

#: The main clock as TFC's Ultra96 builds deliver it (report/delivered_clock.json).
DELIVERED = {MAIN: 187.512}


@pytest.fixture(scope="module")
def parent(tfc_streamlined: Path, tmp_path_factory: pytest.TempPathFactory) -> ModelWrapper:
    directory = tmp_path_factory.mktemp("two_cores")
    return two_tfc_cores(ModelWrapper(str(tfc_streamlined)), directory)


@pytest.fixture(scope="module")
def described(parent: ModelWrapper) -> Integration:
    return describe(parent)


def packaged(described: Integration) -> Integration:
    """``described``, each unit's IP named by the unit, in a repository named so."""
    return described.packaged(
        {unit.name: IpRef(f"v:{unit.name}", (f"/{unit.name}",)) for unit in described.units}
    )


def the_runtime_file(described: Integration) -> str:
    """The runtime file of ``described``, linked by the ``pynq`` linker's placement, its main
    clock delivered as TFC's builds deliver it, as written."""
    linked = Linked(
        image=Path("finn-accel.bit"),
        placement=unit_bases(described, linked_row(described)),
        clocks_mhz=DELIVERED,
        reports={},
    )
    return json.dumps(runtime_file(packaged(described), linked), indent=2) + "\n"


def test_each_core_is_a_unit_between_its_own_movers_numbered_across_the_build(
    described: Integration,
) -> None:
    """The cores in the parent graph's order, each its node's name; each core's units its
    input movers, itself and its output movers; the movers numbered across the build."""
    assert [core.name for core in described.cores] == ["DataflowCore_0", "DataflowCore_1"]
    assert [unit.name for unit in described.units] == [
        "in0",
        "DataflowCore_0",
        "out0",
        "in1",
        "DataflowCore_1",
        "out1",
    ]
    first, second = described.cores
    assert [(edge.unit, edge.tensor) for edge in (*first.inputs, *first.outputs)] == [
        ("in0", first.inputs[0].tensor),
        ("out0", first.outputs[0].tensor),
    ]
    assert all(edge.tensor.startswith(PREFIX) for edge in (*second.inputs, *second.outputs)), second
    assert [(edge.unit, edge.port) for edge in (*second.inputs, *second.outputs)] == [
        ("in1", "s_axis_0"),
        ("out1", "m_axis_0"),
    ]
    # The two bodies are one network each: the same streams at each edge.
    assert [edge.contract for edge in first.inputs] == [edge.contract for edge in second.inputs]


def test_the_block_design_is_the_recorded_text(described: Integration) -> None:
    """The ``pynq`` linker's block design: one control SmartConnect of four masters for the
    four movers, four memory masters, each core's cell between its movers."""
    design = block_design(packaged(described), linked_row(described))
    assert design == (RECORDED / "two-tfc-block-design.tcl").read_text()


def test_the_runtime_file_is_the_recorded_text(described: Integration) -> None:
    """The runtime file: both cores, six units, four placed, each mover's registers; no
    unit's IP, no path of the machine that built it."""
    written = the_runtime_file(described)
    assert written == (RECORDED / "two-tfc-integration.json").read_text()


def test_the_recorded_texts_hold_no_machine_s_path() -> None:
    """The recorded texts are FINN's test data: no path of a machine, no home, no temporary
    directory, no IP repository but the stand-ins the test names."""
    for name in ("two-tfc-block-design.tcl", "two-tfc-integration.json"):
        text = (RECORDED / name).read_text()
        for marker in ("/home", "/tmp", "/proj", "/scratch", "\\", str(Path.home())):
            assert marker not in text, (name, marker)


def test_the_build_s_shell_rules_read_every_core_s_units(
    parent: ModelWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A processor window of three units' windows holds either core and its two movers, so
    each core alone is configured; the build's four movers exceed it, so the build's shell
    refuses it, by its rule."""
    row = shell_row("pynq", "Ultra96")
    assert row.control is not None
    narrow = replace(row, control=replace(row.control, aperture=3 * 4096))
    monkeypatch.setattr(kernel_shell, "shell_row", lambda shell, board: narrow)
    for node, body, _ in core_bodies(parent):
        configured_network(body, node.name)
    with pytest.raises(KernelOpError, match="refused by the 'pynq' shell") as refused:
        describe(parent)
    assert [finding.code for finding in refused.value.findings] == ["control-window-exceeded"]
