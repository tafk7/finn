# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveTransposePastEltwise

# The transposes between NCHW and NHWC around a ResNet-50's per-channel bias and scale:
# (input shape, perm, channel axis of the Transpose's output).
NETWORK_CASES = [
    ([1, 64, 56, 56], [0, 2, 3, 1], 3),
    ([1, 256, 14, 14], [0, 2, 3, 1], 3),
    ([1, 14, 14, 256], [0, 3, 1, 2], 1),
]


def make_const_shape(out_shape, ch_axis, const_mode):
    if const_mode == "scalar":
        return [1]
    if const_mode == "per_channel":
        shape = [1] * len(out_shape)
        shape[ch_axis] = out_shape[ch_axis]
        return shape
    if const_mode == "per_channel_low_rank":
        # Broadcast from the channel axis on: (C,) on NHWC, (C, 1, 1) on NCHW.
        return [out_shape[ch_axis]] + [1] * (len(out_shape) - ch_axis - 1)
    return list(out_shape)


def make_model(in_shape, perm, op_type, const_shapes):
    """A Transpose feeding one Add/Mul per constant shape, each with its own constant;
    with two constants the Transpose forks."""
    out_shape = [in_shape[p] for p in perm]
    nodes = [helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)]
    outputs = []
    for i, _ in enumerate(const_shapes):
        nodes.append(helper.make_node(op_type, ["transp_out", f"a{i}"], [f"outp{i}"]))
        outputs.append(helper.make_tensor_value_info(f"outp{i}", TensorProto.FLOAT, out_shape))
    graph = helper.make_graph(
        nodes=nodes,
        name="mv-transpose-eltwise-graph",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=outputs,
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise"))
    for i, shape in enumerate(const_shapes):
        model.set_initializer(f"a{i}", gen_finn_dt_tensor(DataType["FLOAT32"], shape))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model.transform(GiveUniqueNodeNames())


def input_dict(model, in_shape):
    return {model.get_first_global_in(): gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)}


@pytest.mark.streamline
@pytest.mark.parametrize("in_shape, perm, ch_axis", NETWORK_CASES)
@pytest.mark.parametrize("op_type", ["Add", "Mul"])
@pytest.mark.parametrize("const_mode", ["scalar", "per_channel", "per_channel_low_rank", "full"])
def test_move_transpose_past_eltwise(in_shape, perm, ch_axis, op_type, const_mode):
    out_shape = [in_shape[p] for p in perm]
    model = make_model(in_shape, perm, op_type, [make_const_shape(out_shape, ch_axis, const_mode)])

    model_transformed = model.transform(MoveTransposePastEltwise())

    assert oxe.compare_execution(model, model_transformed, input_dict(model, in_shape))
    # The elementwise op reads the graph input; the Transpose follows it, in order.
    nodes = model_transformed.graph.node
    assert [n.op_type for n in nodes] == [op_type, "Transpose"]
    assert nodes[0].input[0] == model.get_first_global_in()
    assert nodes[1].input[0] == nodes[0].output[0]
    assert model_transformed.get_tensor_shape(nodes[0].output[0]) == in_shape


@pytest.mark.streamline
def test_move_transpose_past_eltwise_shared_constant():
    # Two Transposes read by two Adds that share one constant: each move transposes its
    # own copy, and the other Add still reads the constant as it was.
    in_shape = [1, 8, 4, 4]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]
    nodes = [
        helper.make_node("Transpose", ["inp"], ["t0"], perm=perm),
        helper.make_node("Transpose", ["inp"], ["t1"], perm=perm),
        helper.make_node("Add", ["t0", "a"], ["outp0"]),
        helper.make_node("Add", ["t1", "a"], ["outp1"]),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="mv-transpose-eltwise-shared",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[
            helper.make_tensor_value_info(f"outp{i}", TensorProto.FLOAT, out_shape)
            for i in range(2)
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_shared"))
    model.set_initializer("a", gen_finn_dt_tensor(DataType["FLOAT32"], [1, 1, 1, 8]))
    model = model.transform(InferShapes())

    model_transformed = model.transform(MoveTransposePastEltwise())

    assert oxe.compare_execution(model, model_transformed, input_dict(model, in_shape))
    adds = [n for n in model_transformed.graph.node if n.op_type == "Add"]
    assert all(n.input[0] == "inp" for n in adds)
    assert all(model_transformed.get_initializer(n.input[1]).shape == (1, 8, 1, 1) for n in adds)


@pytest.mark.streamline
@pytest.mark.parametrize("op_type", ["Add", "Mul"])
def test_move_transpose_past_eltwise_dynamic_noop(op_type):
    # Without a constant input (a residual join) the transform leaves the graph as it is.
    in_shape = [1, 64, 56, 56]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm),
            helper.make_node(op_type, ["transp_out", "other"], ["outp"]),
        ],
        name="mv-transpose-eltwise-noop",
        inputs=[
            helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape),
            helper.make_tensor_value_info("other", TensorProto.FLOAT, out_shape),
        ],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_noop"))
    model = model.transform(InferShapes())

    model_transformed = model.transform(MoveTransposePastEltwise())

    assert [n.op_type for n in model_transformed.graph.node] == ["Transpose", op_type]


@pytest.mark.streamline
def test_move_transpose_past_eltwise_fork_noop():
    # A Transpose read by two nodes is MoveTransposePastFork's to split first.
    in_shape = [1, 8, 4, 4]
    perm = [0, 2, 3, 1]
    model = make_model(in_shape, perm, "Add", [[1, 1, 1, 8], [1, 1, 1, 8]])

    model_transformed = model.transform(MoveTransposePastEltwise())

    assert [n.op_type for n in model_transformed.graph.node] == ["Transpose", "Add", "Add"]
