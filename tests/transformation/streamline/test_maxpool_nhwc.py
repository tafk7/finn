import pytest

import onnx
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC


def create_maxpool(ifm_dim, ifm_ch, kernel_shape, pads, strides, ceil_mode, idt):
    ofm_dim_h = compute_pool_output_dim(ifm_dim[0], kernel_shape[0], strides[0], pads[0], ceil_mode)
    ofm_dim_w = compute_pool_output_dim(ifm_dim[1], kernel_shape[1], strides[1], pads[1], ceil_mode)
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim[0], ifm_dim[1]])
    outp_mp = oh.make_tensor_value_info(
        "outp_mp", TensorProto.FLOAT, [1, ifm_ch, ofm_dim_h, ofm_dim_w]
    )
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch])

    maxpool_node = oh.make_node(
        "MaxPool",
        inputs=["inp"],
        outputs=["outp_mp"],
        ceil_mode=ceil_mode,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=["outp_mp"],
        outputs=["outp"],
        name="Transpose1",
        perm=[0, 2, 3, 1],
    )

    graph = oh.make_graph(
        nodes=[maxpool_node, transpose_node],
        name="maxpool_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[outp_mp],
    )

    model = qonnx_make_model(graph, producer_name="maxpool_model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    model = model.transform(InferShapes())

    return model


@pytest.mark.streamline
# input dimension
@pytest.mark.parametrize("ifm_dim", [[8, 8], [9, 9]])
# input channels
@pytest.mark.parametrize("ifm_ch", [3])
# kernel shape
@pytest.mark.parametrize("kernel_shape", [[2, 2]])
# padding
@pytest.mark.parametrize("pads", [[0, 0, 0, 0], [1, 1, 1, 1]])
# strides
@pytest.mark.parametrize("strides", [[2, 2]])
# ceil_mode
@pytest.mark.parametrize("ceil_mode", [0, 1])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
def test_maxpool_nhwc(ifm_dim, ifm_ch, kernel_shape, pads, strides, ceil_mode, idt):
    # create MaxPool node
    maxpool_model = create_maxpool(ifm_dim, ifm_ch, kernel_shape, pads, strides, ceil_mode, idt)

    # generate input tensor for testing
    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim[0], ifm_dim[1]])
    input_dict = {"inp": input_tensor}

    # execute first model
    output_dict = oxe.execute_onnx(maxpool_model, input_dict)
    expected = output_dict["outp"]

    # transform MaxPool into MaxPoolNHWC
    maxpool_model = maxpool_model.transform(MakeMaxPoolNHWC())

    # execute transformed model
    output_node_name = maxpool_model.get_first_global_out()
    output_dict = oxe.execute_onnx(maxpool_model, input_dict, return_full_exec_context=False)
    output = output_dict[output_node_name]

    # the (MaxPool, Transpose) pair became (Transpose, MaxPoolNHWC)
    assert [n.op_type for n in maxpool_model.graph.node] == ["Transpose", "MaxPoolNHWC"]

    # compare outputs
    assert (expected == output).all()


@pytest.mark.streamline
@pytest.mark.parametrize("nhwc_input", [False, True])
def test_maxpool_nhwc_with_a_forking_output(nhwc_input):
    """A MaxPool whose output feeds a Transpose to NHWC and a second node too: the
    Transpose is not pulled ahead of the pool, which would leave the second node reading
    the unpooled tensor. Fed by a Transpose to NCHW (``nhwc_input``), the pool becomes a
    MaxPoolNHWC ahead of that Transpose, which both branches read."""
    nhwc = [1, 8, 8, 3]
    nchw = [1, 3, 8, 8]
    nodes = []
    pool_in = "inp"
    if nhwc_input:
        nodes.append(oh.make_node("Transpose", ["inp"], ["inp_nchw"], perm=[0, 3, 1, 2]))
        pool_in = "inp_nchw"
    nodes += [
        oh.make_node("MaxPool", [pool_in], ["pooled"], kernel_shape=[2, 2], strides=[2, 2]),
        oh.make_node("Transpose", ["pooled"], ["outp"], perm=[0, 2, 3, 1]),
        oh.make_node("Relu", ["pooled"], ["outp2"]),
    ]
    graph = oh.make_graph(
        nodes=nodes,
        name="maxpool_fork",
        inputs=[oh.make_tensor_value_info("inp", TensorProto.FLOAT, nhwc if nhwc_input else nchw)],
        outputs=[
            oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4, 4, 3]),
            oh.make_tensor_value_info("outp2", TensorProto.FLOAT, [1, 3, 4, 4]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="maxpool_fork"))
    model = model.transform(InferShapes())
    idt = DataType["INT4"]
    input_dict = {"inp": gen_finn_dt_tensor(idt, nhwc if nhwc_input else nchw)}
    expected = oxe.execute_onnx(model, input_dict)

    transformed = model.transform(MakeMaxPoolNHWC())

    got = oxe.execute_onnx(transformed, input_dict)
    assert (got["outp"] == expected["outp"]).all()
    assert (got["outp2"] == expected["outp2"]).all()
    pools = [n.op_type for n in transformed.graph.node if n.op_type.startswith("MaxPool")]
    assert pools == (["MaxPoolNHWC"] if nhwc_input else ["MaxPool"])
