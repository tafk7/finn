# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 for the kernel path's tests, from the trained network to a partition of KernelOps.

The BNN-PYNQ TFC at two bits (``brevitas_examples``, its weights from the torch
hub cache), exported by Brevitas with its preprocessing beside it
(``finn.util.pytorch.ToTensor``, ``exported``), through the builder's
graph-preparation phase as TFC's build states it (``io_settings``: the preprocessing
merged, the input UINT8, the label select appended) to the streamlined graph
(``prepared``), then ``ToKernelOps``, ``InferKernelTensors``
and the build's cut (``CutKernelPartition``): the input flatten (a Reshape) before the
partition and the label select (TopK) after it, both on the host.

Every kernel choice is created at its baseline, a KernelOp's own at conversion and a
channel's at the cut; then every one (folding, memories, transports) is set again on the
partition's body, as the build explores it, ranked by hand
(``explore_kernel_choices(body, [Ranked(Lanes(16))])``, ``support.helpers.Lanes``): 16 lanes
where they divide, the whole extent otherwise, every other choice its kernel's baseline.

Building TFC takes about 17 s (torch, the export, streamlining). A test that only
reads it loads ``conftest.py``'s files (``built``), built once a run, across the
pytest-xdist workers too.
"""

from __future__ import annotations

import fcntl
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.builder.build_dataflow import build_plan
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.core.search import Ranked
from finn.platform.request import TargetRequest
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
)
from finn.transformation.kernels.cut import CutKernelPartition
from support.helpers import Lanes
from support.packaging import verified

SHAPE = (1, 1, 28, 28)
LANES = Lanes(16)
"""The fixture's chain: every choice ranked by hand, at 16 lanes."""
ULTRA96 = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
"""TFC's target: Ultra96 in the Zynq shell (no UltraRAM, no doubled clock)."""
EXPORT = "tfc_w2a2.onnx"
"""The export's file name, in the directory ``exported`` writes."""
PREPROCESSING = "preproc.onnx"
"""The preprocessing model's file name, beside the export."""


def exported(directory: Path) -> ModelWrapper:
    """TFC_W2A2 as Brevitas exports it (QONNX, opset 13) into ``directory``, and beside it
    its preprocessing (``PREPROCESSING``: ``ToTensor``, its inputs' scaling to [0, 1]),
    exported alike."""
    # Imported here: torch and brevitas are only for exporting it.
    import torch  # noqa: PLC0415
    from brevitas.export import export_qonnx  # noqa: PLC0415

    from finn.util.pytorch import ToTensor  # noqa: PLC0415
    from support.trained_networks import get_test_model_trained  # noqa: PLC0415

    network = directory / EXPORT
    directory.mkdir(parents=True, exist_ok=True)
    export_qonnx(get_test_model_trained("TFC", 2, 2), torch.randn(SHAPE), network, opset_version=13)
    export_qonnx(ToTensor(), torch.randn(SHAPE), directory / PREPROCESSING, opset_version=13)
    return ModelWrapper(str(network))


def io_settings(directory: Path) -> dict[str, Any]:
    """TFC's inputs and outputs as its build states them (step_add_pre_post_processing's
    settings), its preprocessing model in ``directory``: the images' scaling merged
    ahead, the input UINT8 (a pixel), the top label selected."""
    return {"preprocessing": str(directory / PREPROCESSING), "input_datatype": "UINT8", "topk": 1}


def preparation_config(directory: Path, **fields: Any) -> DataflowBuildConfig:
    """A build of TFC from its export in ``directory`` (``io_settings``), its outputs
    under ``directory / "output"``, for Ultra96 in the Zynq shell at 5 ns, no debugger
    and no intermediate models."""
    return DataflowBuildConfig(
        **{
            "output_dir": str(directory / "output"),
            "target": TargetRequest(board="Ultra96", period_ns=5.0, shell="pynq"),
            "settings": io_settings(directory),
            "enable_build_pdb_debug": False,
            "save_intermediate_models": False,
            **fields,
        }
    )


def run_steps(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """``model`` through the steps ``cfg`` names (its plan,
    finn.builder.build_dataflow.build_plan), each with its settings, in memory:
    no model read or saved between them."""
    plan = build_plan(cfg)
    for step in plan.steps[plan.run]:
        model = step.function(model, cfg, **step.settings)
    return model


def prepared(export: ModelWrapper, directory: Path) -> ModelWrapper:
    """``export``, exported into ``directory`` with its preprocessing beside it (TFC by
    ``exported``, or CNV by ``support.cnv.exported``), through the builder's
    graph-preparation phase (``phase_prepare_model``) as TFC's build states it: the
    streamlined graph."""
    return run_steps(export, preparation_config(directory, steps=["phase_prepare_model"]))


def reference(export: ModelWrapper, directory: Path) -> ModelWrapper:
    """What ``export``'s prepared graph must compute (``prepared``'s arguments): the
    export through the builder's graph-preparation phase as TFC's build states it, up to
    and including step_add_pre_post_processing, then annotated (``InferDataTypes``), its
    Quant nodes and float arithmetic as exported. The equivalence analysis
    (finn.analysis.equivalence) checks the prepared graph against it."""
    cfg = preparation_config(
        directory, steps=["phase_prepare_model"], stop_step="step_add_pre_post_processing"
    )
    return run_steps(export, cfg).transform(InferDataTypes())  # type: ignore[no-untyped-call]


def streamlined(directory: Path) -> ModelWrapper:
    """TFC_W2A2 exported into ``directory`` and prepared: the streamlined graph."""
    return prepared(exported(directory), directory)


def cut(source: ModelWrapper, directory: Path) -> tuple[ModelWrapper, ModelWrapper, str]:
    """The streamlined ``source`` as KernelOps for Ultra96 at 5 ns in the Zynq shell
    (``ULTRA96``), cut once into ``directory``: the parent graph (Reshape, the partition,
    TopK), the partition's body, every choice at its baseline (a KernelOp's own created
    at conversion, a channel's at the cut), and the body's file."""
    model = source.transform(ToKernelOps(ULTRA96)).transform(InferKernelTensors())
    parent = model.transform(CutKernelPartition(directory))
    body_file = getCustomOp(parent.graph.node[1]).get_nodeattr("model")
    return parent, ModelWrapper(body_file), body_file


def partition(source: ModelWrapper, directory: Path) -> tuple[ModelWrapper, ModelWrapper]:
    """The streamlined ``source`` by hand through the kernel path, as the builder's
    kernel-path phases run it: the parent graph (Reshape, the partition, TopK) and the
    partition's body, cut first, then every choice committed at 16 lanes (``LANES``), the
    body verified (``support.packaging.verified``) and saved again in its file."""
    parent, body, body_file = cut(source, directory)
    explore_kernel_choices(body, [Ranked(LANES)])
    verified(body, parent.graph.node[1].name)
    body.save(body_file)
    return parent, body


def partitioned(directory: Path) -> tuple[ModelWrapper, ModelWrapper, ModelWrapper]:
    """The streamlined source, the parent graph (Reshape, the partition, TopK) and the
    partition's body, every choice committed at 16 lanes (``LANES``)."""
    source = streamlined(directory)
    return (source, *partition(source, directory))


def kernel_ops(source: ModelWrapper, directory: Path) -> ModelWrapper:
    """The streamlined ``source``'s partition of KernelOps for Ultra96 at 5 ns in the Zynq
    shell (``ULTRA96``), cut once into ``directory``: its body, every choice at its
    baseline, as exploration reads it."""
    return cut(source, directory)[1]


def built(path: Path, build: Callable[[Path], ModelWrapper]) -> Path:
    """``path``, the model ``build`` makes (given ``path``'s directory), saved there by the
    first process to ask for it: every process that names the same path waits on one
    lock and reads the one build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            build(path.parent).save(str(path))
    return path
