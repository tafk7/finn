# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.data_layout import NCHW, NHWC
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveTransposePastEltwise

# ResNet-50-like cases: after LowerConvsToMatMul the residual path is littered
# with NCHW<->NHWC transposes feeding the per-channel bias/scale ops between
# convolutions. Each case is (input feature-map shape, transpose perm, channel
# axis of the transpose *output*), covering both transpose directions.
NETWORK_CASES = [
    # NCHW -> NHWC (channel moves to last axis)
    ([1, 64, 56, 56], [0, 2, 3, 1], 3),
    ([1, 256, 14, 14], [0, 2, 3, 1], 3),
    # NHWC -> NCHW (channel moves back to axis 1)
    ([1, 14, 14, 256], [0, 3, 1, 2], 1),
]


def make_const_shape(out_shape, ch_axis, const_mode):
    if const_mode == "scalar":
        # a single broadcast scalar (e.g. a global requant scale)
        return [1]
    if const_mode == "per_channel":
        # a per-channel affine vector broadcasting on the channel axis
        shape = [1] * len(out_shape)
        shape[ch_axis] = out_shape[ch_axis]
        return shape
    if const_mode == "per_channel_low_rank":
        # a per-channel vector of lower rank, e.g. (C,) on NHWC or (C, 1, 1) on NCHW
        return [out_shape[ch_axis]] + [1] * (len(out_shape) - ch_axis - 1)
    # full per-element constant
    return list(out_shape)


@pytest.mark.streamline
@pytest.mark.parametrize("in_shape, perm, ch_axis", NETWORK_CASES)
@pytest.mark.parametrize("op_type", ["Add", "Mul"])
@pytest.mark.parametrize("const_mode", ["scalar", "per_channel", "per_channel_low_rank", "full"])
def test_move_transpose_past_eltwise(in_shape, perm, ch_axis, op_type, const_mode):
    out_shape = [in_shape[p] for p in perm]
    const_shape = make_const_shape(out_shape, ch_axis, const_mode)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, const_shape)

    transp_node = helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)
    eltwise_node = helper.make_node(op_type, ["transp_out", "a0"], ["outp"])

    graph = helper.make_graph(
        nodes=[transp_node, eltwise_node],
        name="mv-transpose-eltwise-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise"))

    # the addend/scale must be a constant initializer for the transform to fire
    model.set_initializer("a0", gen_finn_dt_tensor(DataType["FLOAT32"], const_shape))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    idict = {model.get_first_global_in(): gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)}

    # no cleanup: the transform itself must leave a sorted, executable graph
    model_transformed = model.transform(MoveTransposePastEltwise(), cleanup=False)

    # functional equivalence: moving the transpose (and transposing the constant
    # by the inverse permutation) must not change the numerical result
    assert oxe.compare_execution(model, model_transformed, idict)

    # order swapped: the elementwise op now runs first, the transpose second
    assert model_transformed.graph.node[0].op_type == op_type
    assert model_transformed.graph.node[1].op_type == "Transpose"
    # the elementwise op now consumes the pre-transpose input directly
    assert model_transformed.graph.node[0].input[0] == model.get_first_global_in()
    # the transpose now consumes the elementwise output
    assert model_transformed.graph.node[1].input[0] == model_transformed.graph.node[0].output[0]
    assert model_transformed.get_tensor_shape(model_transformed.graph.node[0].output[0]) == in_shape


@pytest.mark.streamline
@pytest.mark.parametrize("op_type", ["Add", "Mul"])
def test_move_transpose_past_eltwise_dynamic_noop(op_type):
    # when the other elementwise input is dynamic (no constant initializer), the
    # transform must not fire -- this is the residual-join case that is handled by
    # the dedicated MoveTransposePastJoinAdd pass instead.
    in_shape = [1, 64, 56, 56]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    other = helper.make_tensor_value_info("other", TensorProto.FLOAT, out_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    transp_node = helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)
    eltwise_node = helper.make_node(op_type, ["transp_out", "other"], ["outp"])

    graph = helper.make_graph(
        nodes=[transp_node, eltwise_node],
        name="mv-transpose-eltwise-noop",
        inputs=[inp, other],
        outputs=[outp],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_noop"))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model_transformed = model.transform(MoveTransposePastEltwise())

    # unchanged: transpose still first, elementwise second
    assert model_transformed.graph.node[0].op_type == "Transpose"
    assert model_transformed.graph.node[1].op_type == op_type


@pytest.mark.streamline
def test_move_transpose_past_eltwise_shared_constant():
    # two Adds read the same constant: each move transposes its own copy, and
    # the other Add still reads the constant unchanged
    in_shape = [1, 8, 4, 4]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Transpose", ["inp"], ["t0"], perm=perm),
            helper.make_node("Transpose", ["inp"], ["t1"], perm=perm),
            helper.make_node("Add", ["t0", "a"], ["outp0"]),
            helper.make_node("Add", ["t1", "a"], ["outp1"]),
        ],
        name="mv-transpose-eltwise-shared",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[
            helper.make_tensor_value_info("outp0", TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info("outp1", TensorProto.FLOAT, out_shape),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_shared"))
    model.set_initializer("a", gen_finn_dt_tensor(DataType["FLOAT32"], [1, 1, 1, 8]))
    # no cleanup here: GiveUniqueParameterTensors would split the shared constant
    model = model.transform(InferShapes(), cleanup=False)

    idict = {"inp": gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)}
    model_transformed = model.transform(MoveTransposePastEltwise())

    assert oxe.compare_execution(model, model_transformed, idict)
    adds = model_transformed.get_nodes_by_op_type("Add")
    assert all(n.input[0] == "inp" for n in adds)
    assert all(model_transformed.get_initializer(n.input[1]).shape == (1, 8, 1, 1) for n in adds)


@pytest.mark.streamline
@pytest.mark.parametrize("annotated", ["input", "outputs", "transpose_output"])
def test_move_transpose_past_eltwise_annotations(annotated):
    # the layout comes from the input, else from the outputs through the perm, else
    # the stale layout of the old transpose output is dropped
    in_shape = [1, 8, 4, 4]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm),
            helper.make_node("Add", ["transp_out", "a0"], ["outp"]),
        ],
        name="mv-transpose-eltwise-annotations",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_ann"))
    model.set_initializer("a0", gen_finn_dt_tensor(DataType["UINT2"], [1, 1, 1, 8]))
    model.set_tensor_datatype("a0", DataType["UINT2"])
    model.set_tensor_datatype("inp", DataType["INT4"])
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    if annotated == "input":
        model.set_tensor_layout("inp", NCHW)
    else:
        model.set_tensor_layout("transp_out", NHWC)
        if annotated == "outputs":
            model.set_tensor_layout("outp", NHWC)
    out_dt = model.get_tensor_datatype("outp")
    assert out_dt != DataType["INT4"]

    model_transformed = model.transform(MoveTransposePastEltwise(), cleanup=False)

    add = model_transformed.get_nodes_by_op_type("Add")[0]
    assert model_transformed.get_tensor_datatype(add.output[0]) == out_dt
    expected_layout = None if annotated == "transpose_output" else NCHW
    assert model_transformed.get_tensor_layout(add.output[0]) == expected_layout
    assert model_transformed.get_tensor_layout("outp") == (NHWC if annotated == "outputs" else None)


@pytest.mark.streamline
def test_move_transpose_past_eltwise_higher_rank_noop():
    # a constant of higher rank than the Transpose widens the output, so the
    # Transpose cannot follow the elementwise op
    in_shape = [1, 8, 4, 4]
    perm = [0, 2, 3, 1]
    out_shape = [2] + [in_shape[p] for p in perm]
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm),
            helper.make_node("Add", ["transp_out", "a0"], ["outp"]),
        ],
        name="mv-transpose-eltwise-higher-rank",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_rank"))
    model.set_initializer("a0", gen_finn_dt_tensor(DataType["FLOAT32"], [2, 1, 1, 1, 8]))
    model = model.transform(InferShapes())

    model_transformed = model.transform(MoveTransposePastEltwise())

    assert [n.op_type for n in model_transformed.graph.node] == ["Transpose", "Add"]


@pytest.mark.streamline
@pytest.mark.parametrize("const_shape", [[1], [1, 1, 1, 3]])
def test_move_transpose_past_eltwise_constant_graph_output(const_shape):
    # a constant that is also a graph output must keep its value and shape, so a
    # constant that needs a transpose stops the move
    in_shape = [1, 3, 2, 4]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm),
            helper.make_node("Add", ["transp_out", "a0"], ["outp"]),
        ],
        name="mv-transpose-eltwise-const-output",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[
            helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info("a0", TensorProto.FLOAT, const_shape),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_const_out"))
    model.set_initializer("a0", gen_finn_dt_tensor(DataType["FLOAT32"], const_shape))
    model = model.transform(InferShapes())

    model_transformed = model.transform(MoveTransposePastEltwise())

    moved = const_shape == [1]
    expected_ops = ["Add", "Transpose"] if moved else ["Transpose", "Add"]
    assert [n.op_type for n in model_transformed.graph.node] == expected_ops
    idict = {"inp": gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)}
    expected = oxe.execute_onnx(model, idict)
    produced = oxe.execute_onnx(model_transformed, idict)
    for name in ["outp", "a0"]:
        assert produced[name].shape == expected[name].shape
        assert (produced[name] == expected[name]).all()


@pytest.mark.streamline
@pytest.mark.parametrize("const_shape", [[4, 1, 1], [4, 4, 8, 1]])
def test_move_transpose_past_eltwise_without_perm(const_shape):
    # a Transpose without perm reverses the axes
    in_shape = [1, 8, 4, 4]
    out_shape = in_shape[::-1]
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Transpose", ["inp"], ["transp_out"]),
            helper.make_node("Mul", ["transp_out", "a0"], ["outp"]),
        ],
        name="mv-transpose-eltwise-no-perm",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_no_perm"))
    model.set_initializer("a0", np.asarray(np.random.randn(*const_shape), dtype=np.float32))
    model = model.transform(InferShapes())

    model_transformed = model.transform(MoveTransposePastEltwise())

    assert [n.op_type for n in model_transformed.graph.node] == ["Mul", "Transpose"]
    idict = {"inp": gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)}
    assert oxe.compare_execution(model, model_transformed, idict)
