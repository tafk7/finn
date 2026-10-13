# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import logging
from qonnx.transformation.base import Transformation

from finn.transformation.qonnx.qonnx_activation_handlers import (
    QuantActBaseHandler,
    QuantIdentityHandler,
    refused,
)

_LOG = logging.getLogger(__name__)


def default_filter_function_generator(max_multithreshold_bit_width=8):
    """ConvertQuantActToMultiThreshold's default filter function: it leaves a Quant node
    wider than ``max_multithreshold_bit_width`` bits a Quant."""

    def filter_function(model, q_node):
        if q_node.op_type == "Quant":
            bit_width = model.get_initializer(q_node.input[3])
        elif q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        if bit_width is None:
            raise refused(
                "quant-bitwidth-dynamic", q_node, "Quant nodes must have a static bit width."
            )
        return bool(bit_width <= max_multithreshold_bit_width)

    return filter_function


class ConvertQuantActToMultiThreshold(Transformation):
    """Converts the Quant and BipolarQuant nodes of the activation path to
    MultiThresholds, each by the handler of its predecessor's op type, else as an
    identity.

    :param filter_function: called with the model and each candidate node; a node it
        returns False for stays a Quant, warned of once, with its bit width, when no
        other node is left to convert. By default, a Quant wider than 8 bits stays.
    """

    def __init__(
        self,
        filter_function=default_filter_function_generator(max_multithreshold_bit_width=8),
    ):
        super().__init__()
        self._filter_function = filter_function

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        left = []

        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant" or n.op_type == "BipolarQuant":
                inp = model.get_initializer(n.input[0])
                out = model.get_initializer(n.output[0])
                if not (inp is None and out is None):
                    continue
                predecessor = model.find_direct_predecessors(n)
                if predecessor is not None:
                    predecessor_op_type = predecessor[0].op_type
                else:
                    predecessor_op_type = predecessor
                if n.op_type == "Quant" and not model.get_initializer(n.input[2]) == 0:
                    raise refused(
                        "quant-zero-point",
                        n,
                        "Only Quant nodes with zero-point == 0 are currently supported.",
                    )

                if not self._filter_function(model, n):
                    left.append(n)
                    continue

                valid_predecessors = []
                for cls in QuantActBaseHandler.__subclasses__():
                    valid_predecessors.extend(cls.valid_predecessor_op_types())
                if len(valid_predecessors) != len(set(valid_predecessors)):
                    raise RuntimeError(
                        "Two or more activation handlers declare the same "
                        "type of valid predecessor node. "
                        "This leads to ambiguity in the handler selection "
                        "and must thus be avoided."
                    )

                for handler_cls in QuantActBaseHandler.__subclasses__():
                    if predecessor_op_type in handler_cls.valid_predecessor_op_types():
                        handler = handler_cls(model, n, node_ind)
                        break
                else:
                    handler = QuantIdentityHandler(model, n, node_ind)

                model = handler.replace_quant_node()
                graph_modified = True
                return (model, graph_modified)

        # The pass that converts nothing more is the last: each Quant left is warned of once.
        for n in left:
            bit_width = 1.0 if n.op_type == "BipolarQuant" else model.get_initializer(n.input[3])
            _LOG.warning(
                f'The Quant node "{n.name}" (bit width {bit_width}) was not converted to a '
                "MultiThreshold node: the filter function returned False for it."
            )
        return (model, graph_modified)
