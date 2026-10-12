# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""CNV_W2A2 for the kernel path's tests: the BNN-PYNQ CNV at two bits (its weights from the
torch hub cache), exported by Brevitas with its preprocessing beside it, as
``kernel_ops.tfc`` exports TFC. Its build states its inputs and outputs as TFC's does
(``kernel_ops.tfc.io_options``), so ``kernel_ops.tfc.prepared`` prepares it.

Exporting and preparing CNV takes about a minute; ``conftest.py`` builds it once a run
(``cnv_export``, ``cnv_streamlined``), for the tests marked slow that read it.
"""

from __future__ import annotations

from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper

from kernel_ops.tfc import PREPROCESSING

SHAPE = (1, 3, 32, 32)
EXPORT = "cnv_w2a2.onnx"
"""The export's file name, in the directory ``exported`` writes."""


def exported(directory: Path) -> ModelWrapper:
    """CNV_W2A2 as Brevitas exports it (QONNX, opset 13) into ``directory``, and beside it
    its preprocessing (``kernel_ops.tfc.PREPROCESSING``), exported alike."""
    # Imported here: torch and brevitas are only for exporting it.
    import torch  # noqa: PLC0415
    from brevitas.export import export_qonnx  # type: ignore[import-untyped]  # noqa: PLC0415

    from finn.util.pytorch import ToTensor  # noqa: PLC0415
    from finn.util.test import get_test_model_trained  # noqa: PLC0415

    network = directory / EXPORT
    directory.mkdir(parents=True, exist_ok=True)
    trained = get_test_model_trained("CNV", 2, 2)  # type: ignore[no-untyped-call]
    export_qonnx(trained, torch.randn(SHAPE), network, opset_version=13)
    preprocessing = ToTensor()  # type: ignore[no-untyped-call]
    export_qonnx(preprocessing, torch.randn(SHAPE), directory / PREPROCESSING, opset_version=13)
    return ModelWrapper(str(network))
