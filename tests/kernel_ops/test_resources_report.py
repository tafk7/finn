# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``report/resources.json``: the resources per member of the shell, by the model, out of
context and placed (``finn.builder.kernel_resources``).

The reports are Vivado's formats with invented numbers (``kernel_ops.packaging``'s
``UTILIZATION_SYNTH`` and ``PLACED_HIERARCHY``)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from finn.builder.kernel_resources import (
    placed_hierarchy,
    shell_resources_report,
    utilization_synth,
)
from finn.custom_op.kernels.base import write_target
from finn.custom_op.kernels.shell import ShellResources, shell_resources
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.kernels.utilization import SHELL_CHARACTERISED, Resources
from finn.platform import resolve_target, shell_row
from finn.platform.shells import PYNQ_CHARACTERISED
from finn.transformation.kernels import explore_kernel_choices
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import configured_root
from kernel_ops.models import configure_partition, kernel_model
from kernel_ops.packaging import PLACED_HIERARCHY, UTILIZATION_SYNTH


def test_an_ips_synthesis_report_is_read_from_its_site_types() -> None:
    # Two RAMB36 and one RAMB18 are five halves; the primitives' later rows are not read.
    assert utilization_synth(UTILIZATION_SYNTH.format(lut=120, ff=200)) == Resources(
        lut=120, ff=200, bram18=5, dsp=3
    )


def test_the_placed_hierarchy_is_read_by_depth() -> None:
    rows = placed_hierarchy(PLACED_HIERARCHY)
    assert [(depth, name) for depth, name, _ in rows] == [
        (0, "top_wrapper"),
        (1, "top_i"),
        (2, "smartconnect_0"),
        (3, "inst"),
        (2, "partition"),
        (2, "axi_interconnect_0"),
        (2, "idma0"),
        (2, "odma0"),
    ]
    assert rows[0][2] == Resources(lut=1000, ff=2000, bram18=5, dsp=4)
    assert rows[4][2] == Resources(lut=400, ff=800, bram18=3, dsp=4)


def test_a_partition_on_the_ip_shell_states_its_model_alone(tmp_path: Path) -> None:
    """The Chain on the ip shell, nothing synthesized: the shell is its partition, with no
    end and no static region; neither Vivado column was made, and each says why."""
    parent = kernel_model().transform(CutKernelPartition(tmp_path))
    node, body, _ = partition_body(parent)
    point, _ = configured_root(body, node.name)
    modelled = shell_resources(point)
    assert isinstance(modelled, ShellResources)
    stated = shell_resources_report(parent)
    assert (stated["shell"], stated["caveat"]) == ("ip", None)
    assert stated["members"] == {
        "partition": {
            "instance": "partition",
            "model": asdict(modelled.partition),
            "out_of_context": None,
            "placed": None,
        },
        "ends": {},
        "static_region": {},
    }
    assert stated["total"] == {
        "model": asdict(modelled.partition),
        "out_of_context": None,
        "placed": None,
    }
    assert stated["unattributed"]["model"] == asdict(Resources())
    assert set(stated["absent"]) == {"out_of_context", "placed"}
    assert "not placed" in stated["absent"]["placed"]


@pytest.mark.parametrize("board", ["Ultra96", "ZCU104"])
def test_off_the_board_the_shell_was_timed_on_both_reports_name_the_caveat(
    board: str, tmp_path: Path
) -> None:
    """SZ2: the pynq shell was built and timed on Ultra96 only, its ends and static region
    characterised on xczu3eg and xczu7ev. Every pynq row says so; on another board
    kernel_exploration.json's ``exact`` and resources.json's ``caveat`` name it, the
    board and its part; on Ultra96 neither has a caveat."""
    row = shell_row("pynq", board)
    assert row.characterised == PYNQ_CHARACTERISED
    assert "xczu3eg and xczu7ev" in PYNQ_CHARACTERISED
    assert "built and timed on Ultra96 only" in PYNQ_CHARACTERISED
    model = kernel_model()
    write_target(model, resolve_target(board=board, period_ns=5.0, shell="pynq"))
    configure_partition(model)
    exact = explore_kernel_choices(model, []).report["resources"]["exact"]
    # Every memory is counted in the target's fabric's table; a FIFO's shallow hi space
    # is Vivado's, a model; any other memory in auto is Vivado's, not stated.
    assert "every memory's block RAM, UltraRAM and LUTRAM storage" in exact
    assert "in the ultrascale primitive table" in exact
    assert "a FIFO's shallow hi space, which Vivado places, is a model" in exact
    assert "placed by Vivado, not stated" in exact
    stated = shell_resources_report(model.transform(CutKernelPartition(tmp_path)))
    assert stated["caveat"] == row.caveat
    assert SHELL_CHARACTERISED in stated["columns"]["model"]
    if board == "Ultra96":
        assert row.caveat is None and "carried over" not in exact
        return
    assert row.caveat is not None and exact.endswith("; " + row.caveat)
    for named in ("built and timed on Ultra96 only", SHELL_CHARACTERISED, board):
        assert named in row.caveat
    assert "on ZCU104 (xczu7ev-ffvc1156-2-e) its cost is carried over" in row.caveat
