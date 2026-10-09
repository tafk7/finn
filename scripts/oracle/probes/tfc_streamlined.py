# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 streamlined by finn-dev's own end-to-end steps: ``TestEnd2End``'s
``test_export``, ``test_import_and_tidy``, ``test_add_pre_and_postproc`` and
``test_streamline`` (``tests/end2end/test_end2end_bnn_pynq.py``) for ``tfc``, w2a2,
each step's model saved and read again as the test's checkpoints are. The model is
captured as ``tfc_w2a2_streamlined.onnx``; the values name its nodes and boundary."""

import torch
from _probe import arguments, write
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.transformation.streamline.absorb as absorb
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
from finn.util.pytorch import ToTensor
from finn.util.test import get_trained_network_and_ishape


def tidy(model):
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


raw, _ = arguments()
# test_export
network, ishape = get_trained_network_and_ishape("tfc", 2, 2)
export_qonnx(network, torch.randn(ishape), "export.onnx", opset_version=13)
qonnx_cleanup("export.onnx", out_file="export.onnx")
ModelWrapper("export.onnx").transform(ConvertQONNXtoFINN()).save("export.onnx")
# test_import_and_tidy
tidy(ModelWrapper("export.onnx")).save("import_and_tidy.onnx")
# test_add_pre_and_postproc
model = ModelWrapper("import_and_tidy.onnx")
ishape = model.get_tensor_shape(model.get_first_global_in())
export_qonnx(ToTensor(), torch.randn(ishape), "preproc.onnx", opset_version=13)
qonnx_cleanup("preproc.onnx", out_file="preproc.onnx")
ModelWrapper("preproc.onnx").transform(ConvertQONNXtoFINN()).save("preproc.onnx")
pre_model = ModelWrapper("preproc.onnx").transform(InferShapes()).transform(FoldConstants())
model = model.transform(MergeONNXModels(pre_model))
model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
tidy(model.transform(InsertTopK(k=1))).save("pre_post.onnx")
# test_streamline
model = ModelWrapper("pre_post.onnx")
model = model.transform(absorb.AbsorbScalarBiasIntoMultiThreshold())
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(Streamline())
model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(Streamline())
model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save(str(raw / "tfc_w2a2_streamlined.onnx"))


def tensor(name):
    return {
        "name": name,
        "shape": list(model.get_tensor_shape(name)),
        "datatype": model.get_tensor_datatype(name).name,
    }


write(
    raw,
    {
        "model": "tfc_w2a2_streamlined.onnx",
        "nodes": [[node.name, node.op_type] for node in model.graph.node],
        "inputs": [tensor(each.name) for each in model.graph.input],
        "outputs": [tensor(each.name) for each in model.graph.output],
    },
)
