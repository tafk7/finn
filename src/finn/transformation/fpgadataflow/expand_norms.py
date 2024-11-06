import numpy as np
from onnx import TensorProto
from onnx import helper as oh

from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

from finn.util.fpgadataflow import is_fpgadataflow_node


class ExpandNorms(Transformation):
    """Expand any standard LayerNorms/RMSNorms into the functional 
    norm and Mul/Add nodes for affine scale and bias."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "LayerNormalization":
                graph_modified = True
                ln_input = n.input[0]
                ln_output = n.output[0]
                # extract batchnorm parameters as numpy arrays
                scale = model.get_initializer(n.input[1])
                sdt = model.get_tensor_datatype(scale)
                # mean = model.get_initializer(n.output[1])
                # invStdDev = model.get_initializer(n.output[2])
                epsilon = get_by_name(n.attribute, "epsilon")
                print(epsilon)
                print(dir(epsilon))
                epsilon = getattr(epsilon, "f", 1e-5)
                axis = getattr(get_by_name(n.attribute, "axis"), "i", -1)
                print(axis)
                print(dir(axis))
                1/0
                stash_type = get_by_name(n.attribute, "stash_type")
                stash_type = getattr(stash_type, "s", -1)




                mm_input = n.input[0]
                mm_weight = n.input[1]
                # if mm_weight is not constant, skip node
                if model.get_initializer(n.input[1]) is None:
                    continue
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)



                bias = model.get_initializer(n.input[2])





                variance = model.get_initializer(n.input[4])
                epsilon = get_by_name(n.attribute, "epsilon")

                axis = getattr(axis, "f", -1)
                epsilon = getattr(epsilon, "f", 1e-5)
                stash_type = getattr(stash_type, "f", 1e-5)


                # find A and B to compute batchnorm as affine transpose Ax+B
                # TODO is a division by moving avg factor needed for variance?
                A = scale / np.sqrt(epsilon + variance)
                B = bias - (A * mean)
                # see if we have surrounding Unsqueeze/Squeeze nodes we can remove
                producer = model.find_producer(ln_input)
                if producer is not None:
                    if producer.op_type == "Unsqueeze":
                        ln_input = producer.input[0]
                consumer = model.find_consumer(ln_output)
                if consumer is not None:
                    if consumer.op_type == "Squeeze":
                        ln_output = consumer.output[0]
                data_shape = model.get_tensor_shape(ln_input)
                assert A.ndim == B.ndim, "Unexpected mul/add dims in BatchNormToAffine"
                assert len(data_shape) >= A.ndim, "Unexpected number of dims found in BatchNormToAffine"
                # reshape the mul/add constants to match the data shape/dims
                # by adding (1,) dimensions to the right
                n_spatial_dims = len(data_shape) - 2
                target_shape = (1, -1) + tuple(1 for i in range(n_spatial_dims))
                A = A.reshape(target_shape)
                B = B.reshape(target_shape)
                # create value_info and initializers for Mul and Add constants
                mul_const = oh.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, A.shape)
                graph.value_info.append(mul_const)
                model.set_initializer(mul_const.name, A)
                mul_output = oh.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, data_shape)
                graph.value_info.append(mul_output)
                add_const = oh.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, B.shape)
                graph.value_info.append(add_const)
                model.set_initializer(add_const.name, B)


            if n.op_type == "SimplifiedLayerNormFusion":
                # create Mul and Add nodes to replace the batchnorm
                mul_node = oh.make_node("Mul", [ln_input, mul_const.name], [mul_output.name])
                add_node = oh.make_node("Add", [mul_output.name, add_const.name], [ln_output])
                # insert where the batchnorm is to preserve topological ordering
                graph.node.insert(node_ind, mul_node)
                graph.node.insert(node_ind + 1, add_node)
                # remove old nodes
                graph.node.remove(n)
                if consumer is not None:
                    if consumer.op_type == "Squeeze":
                        graph.node.remove(consumer)
                if producer is not None:
                    if producer.op_type == "Unsqueeze":
                        graph.node.remove(producer)
        model = model.transform(InferShapes())
        return (model, graph_modified)