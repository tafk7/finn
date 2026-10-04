# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 for the kernel path's tests, from the trained network to a partition of KernelOps.

The BNN-PYNQ TFC at two bits (``brevitas_examples``, its weights from the torch
hub cache) through finn-dev's own end-to-end steps to the streamlined graph
(``tests/end2end/test_end2end_bnn_pynq.py``: export, tidy, pre- and
post-processing, streamline), then ``ToKernelOps``, ``InferKernelTensors`` and
``CreateDataflowPartition``: the input flatten (a Reshape) before the partition
and the label select (TopK) after it, both on the host.

Every open kernel choice (folding, memories, adapters) is committed before
partitioning by ``CommitKernelChoices(PlaceholderPolicy())``, the DSE seam's
placeholder (G4): 16 lanes where they divide, the whole extent otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.kernels import (
    CommitKernelChoices,
    InferKernelTensors,
    PlaceholderPolicy,
    ToKernelOps,
)
from finn.kernels.target import resolve_target

SHAPE = (1, 1, 28, 28)
ULTRA96 = resolve_target("xczu3eg-sbva484-1-e", 5.0, "vivado_zynq")
"""TFC's target: Ultra96 in the Zynq shell (no UltraRAM, no doubled clock)."""


def _tidy(model: ModelWrapper) -> ModelWrapper:
    for step in (
        InferShapes(),
        FoldConstants(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferDataTypes(),
        RemoveStaticGraphInputs(),
    ):
        model = model.transform(step)
    return model


def streamlined(directory: Path) -> ModelWrapper:
    """TFC_W2A2 after finn-dev's end-to-end steps up to and including streamlining."""
    # Imported here: torch, brevitas and the baseline flow are only for building it.
    import torch  # noqa: PLC0415
    from brevitas.export import export_qonnx  # noqa: PLC0415
    from qonnx.core.datatype import DataType  # noqa: PLC0415
    from qonnx.transformation.bipolar_to_xnor import (  # noqa: PLC0415
        ConvertBipolarMatMulToXnorPopcount,
    )
    from qonnx.transformation.general import RemoveUnusedTensors  # noqa: PLC0415
    from qonnx.transformation.infer_data_layouts import InferDataLayouts  # noqa: PLC0415
    from qonnx.transformation.insert_topk import InsertTopK  # noqa: PLC0415
    from qonnx.transformation.merge_onnx_models import MergeONNXModels  # noqa: PLC0415
    from qonnx.util.cleanup import cleanup as qonnx_cleanup  # noqa: PLC0415

    import finn.transformation.streamline.absorb as absorb  # noqa: PLC0415
    from finn.transformation.qonnx.convert_qonnx_to_finn import (  # noqa: PLC0415
        ConvertQONNXtoFINN,
    )
    from finn.transformation.streamline import Streamline  # noqa: PLC0415
    from finn.transformation.streamline.reorder import (  # noqa: PLC0415
        MoveScalarLinearPastInvariants,
    )
    from finn.util.pytorch import ToTensor  # noqa: PLC0415
    from finn.util.test import get_test_model_trained  # noqa: PLC0415

    network = directory / "tfc_w2a2.onnx"
    export_qonnx(get_test_model_trained("TFC", 2, 2), torch.randn(SHAPE), network, opset_version=13)
    qonnx_cleanup(str(network), out_file=str(network))
    model = _tidy(ModelWrapper(str(network)).transform(ConvertQONNXtoFINN()))
    pre = directory / "preproc.onnx"
    export_qonnx(ToTensor(), torch.randn(SHAPE), pre, opset_version=13)
    qonnx_cleanup(str(pre), out_file=str(pre))
    pre_model = ModelWrapper(str(pre)).transform(ConvertQONNXtoFINN())
    pre_model = pre_model.transform(InferShapes()).transform(FoldConstants())
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
    model = _tidy(model.transform(InsertTopK(k=1)))
    model = model.transform(absorb.AbsorbScalarBiasIntoMultiThreshold())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    return model.transform(RemoveUnusedTensors())


def partitioned(directory: Path) -> tuple[ModelWrapper, ModelWrapper, ModelWrapper]:
    """The streamlined source, the parent graph (Reshape, the partition, TopK) and the
    partition's body, every choice committed by the placeholder policy."""
    source = streamlined(directory)
    model = (
        source.transform(ToKernelOps(ULTRA96))
        .transform(InferKernelTensors())
        .transform(CommitKernelChoices(PlaceholderPolicy()))
    )
    parent = model.transform(CreateDataflowPartition(partition_model_dir=str(directory)))
    sdp = getCustomOp(parent.graph.node[1])
    body: Any = ModelWrapper(sdp.get_nodeattr("model"))
    return source, parent, body
