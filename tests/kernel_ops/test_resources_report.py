# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``report/resources.json``: the resources per member of the shell, by the model, out of
context and placed (``finn.builder.kernel_resources``).

The reports are Vivado's formats with invented numbers (``kernel_ops.packaging``'s
``UTILIZATION_SYNTH`` and ``PLACED_HIERARCHY``)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from finn.builder.kernel_resources import (
    placed_hierarchy,
    shell_resources_report,
    utilization_synth,
)
from finn.custom_op.kernels.shell import ShellResources, shell_resources
from finn.kernels.utilization import Resources
from finn.transformation.fpgadataflow.cut_kernel_partition import CutKernelPartition
from finn.transformation.fpgadataflow.kernel_partitions import partition_body
from finn.transformation.kernels.package import configured_root
from kernel_ops.models import kernel_model
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
    assert stated["shell"] == "ip"
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
