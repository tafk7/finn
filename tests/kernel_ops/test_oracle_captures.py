# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The finn-dev oracle's captures (``tests/oracle``) are one oracle's, whole, and cover
what the kernel path's tests ask of them.

The tests that compare with the HWCustomOp flow read these captures instead of calling
it: ``test_fifo`` its FIFO model, ``measure_cycles`` its cycle estimates,
``test_builder_phase`` SetFolding and the Zynq build's IODMAs and block design.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from oracle import CAPTURES, capture_file, document
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels.shell import shell_root
from kernel_ops.measure_cycles import finn_cycles, schedule_of
from kernel_ops.models import configure_partition, kernel_model
from kernel_ops.tfc import partition

PROBES = ("fifo_cost", "exp_cycles", "tfc_streamlined", "tfc_set_folding", "tfc_zynq_build")


def test_every_capture_is_one_oracles_with_the_files_it_names() -> None:
    """Every probe's capture is there, from one oracle commit and venv, and every file a
    capture names is the one it captured; no file lies there that none names."""
    documents = [document(probe) for probe in PROBES]
    assert sorted(path.stem for path in CAPTURES.glob("*.json")) == sorted(PROBES)
    assert len({(each["oracle"]["commit"], str(each["versions"])) for each in documents}) == 1
    named = {name for probe, each in zip(PROBES, documents) for name in each["files"]}
    assert {path.name for path in CAPTURES.iterdir() if path.suffix != ".json"} == named
    for probe, each in zip(PROBES, documents):
        for name in each["files"]:
            assert capture_file(probe, name).is_file()


def measured(model: ModelWrapper, label: str) -> dict[str, int]:
    """FINN's cycles of each layer ``measure_cycles`` measures in ``model``."""
    root = shell_root(model, model.graph.node, name=label)
    return {node.name: finn_cycles(node, schedule_of(root, node)) for node in model.graph.node}


def test_the_chains_layers_have_finns_cycles_captured() -> None:
    model = kernel_model()
    configure_partition(model)
    assert measured(model, "chain") == {"first": 12, "activate": 6, "second": 12}


@pytest.mark.slow
def test_tfcs_layers_at_16_lanes_have_finns_cycles_captured(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    _, body = partition(ModelWrapper(str(tfc_streamlined)), tmp_path)
    assert measured(body, "tfc_w2a2") == {
        "MultiThreshold_0": 49,
        "MatMul_0": 196,
        "MultiThreshold_1": 4,
        "MatMul_1": 16,
        "MultiThreshold_2": 4,
        "MatMul_2": 16,
        "MultiThreshold_3": 4,
        "MatMul_3": 4,
    }
