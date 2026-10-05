# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ZynqBuild over a model of KernelOps, up to its IP builds: the partitions it prepares,
and the toolchain it hands to each transformation that runs Vivado.

test_design's Chain, its choices saved, as the KernelOps' model; no Vivado (the
build itself: the TFC_W2A2 build script in the scratchpad's
records/zynq-kernel-build-2026-10-04).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow import make_zynq_proj
from finn.transformation.fpgadataflow.create_stitched_ip import collect_ip_dirs
from finn.transformation.fpgadataflow.kernel_partitions import partition_facts
from finn.transformation.fpgadataflow.make_driver import get_driver_shapes
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from kernel_ops.test_partition import configured, kernel_model


def test_a_model_of_kernel_ops_becomes_iodma_and_kernel_partitions(tmp_path: Path) -> None:
    model = kernel_model()
    configured(model)
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path))
    parent = build.prepare_kernel_partitions(model)
    bodies = [ModelWrapper(getCustomOp(node).get_nodeattr("model")) for node in parent.graph.node]
    assert [node.op_type for node in parent.graph.node] == ["StreamingDataflowPartition"] * 3
    assert [[node.op_type for node in body.graph.node] for body in bodies] == [
        ["IODMA_hls"],
        ["MatMul", "Thresholding", "MatMul"],
        ["IODMA_hls"],
    ]
    # The KernelOps' partition states its facts; the IODMAs' widths came from them.
    inputs, outputs = partition_facts(bodies[1])
    assert (inputs[0]["tdata"], outputs[0]["tdata"]) == (8, 16)
    assert getCustomOp(bodies[0].graph.node[0]).get_nodeattr("streamWidth") == 8
    assert getCustomOp(bodies[2].graph.node[0]).get_nodeattr("streamWidth") == 16
    assert get_driver_shapes(parent)["ishape_folded"] == [(1, 6, 2)]
    # The packaged IP is self-contained: the shell adds only its directory.
    assert collect_ip_dirs(bodies[1], "/stitch") == ["/stitch/ip"]


VIVADO_STEPS = ("PackagePartition", "CreateStitchedIP", "MakeZYNQProject")


def recorded_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, toolchain: object
) -> list[tuple[str, object]]:
    """ZynqBuild over the Chain, each step that runs a tool replaced by a recorder of
    the toolchain it is given: the order and toolchain of each Vivado step."""
    seen: list[tuple[str, object]] = []

    def recorder(name: str) -> type[Transformation]:
        class Recorded(Transformation):  # type: ignore[misc]
            def __init__(self, *args: object, toolchain: object = None, **kwargs: object):
                super().__init__()
                if name in VIVADO_STEPS:
                    seen.append((name, toolchain))

            def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
                return model, False

        return Recorded

    for name in (*VIVADO_STEPS, "PrepareIP", "HLSSynthIP"):
        monkeypatch.setattr(make_zynq_proj, name, recorder(name))
    model = kernel_model()
    configured(model)
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path), toolchain=toolchain)
    model.transform(build)
    return seen


def test_a_build_runs_vivado_through_the_toolchain_it_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    given = object()
    monkeypatch.setattr(make_zynq_proj, "legacy_toolchain", lambda: pytest.fail("prepared"))
    seen = recorded_build(monkeypatch, tmp_path, given)
    # The IODMAs' partitions are stitched, the KernelOps' is packaged, then the project.
    assert seen == [
        ("CreateStitchedIP", given),
        ("PackagePartition", given),
        ("CreateStitchedIP", given),
        ("MakeZYNQProject", given),
    ]


def test_a_build_prepares_its_default_toolchain_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared: list[object] = []

    def legacy_toolchain() -> object:
        prepared.append(object())
        return prepared[-1]

    monkeypatch.setattr(make_zynq_proj, "legacy_toolchain", legacy_toolchain)
    seen = recorded_build(monkeypatch, tmp_path, None)
    assert len(prepared) == 1
    assert [name for name, _ in seen] == [
        "CreateStitchedIP",
        "PackagePartition",
        "CreateStitchedIP",
        "MakeZYNQProject",
    ]
    assert all(toolchain is prepared[0] for _, toolchain in seen)
