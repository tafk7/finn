############################################################################
# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import json
import warnings
from onnx import AttributeProto
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.transformation.base import Transformation


class ApplyConfig(Transformation):
    """Sets custom nodes' attributes from a config dict, or a JSON file of one, by node
    name; ``Defaults`` sets an attribute on every node of the op types it lists::

        {
        # kernel_size = 3 on every Im2Col node
        "Defaults" : {"kernel_size" : [3, ["Im2Col"]]},
        # kernel_size = 7 on the node Im2Col_0
        "Im2Col_0" : {"kernel_size" : 7}
        }

    A node in a subgraph is named by its parents and their graph attributes, joined
    by ``_``. Nodes without an entry, entries for standard nodes and entries no node
    used are warned of.
    """

    def __init__(self, config, node_filter=lambda x: True):
        super().__init__()
        self.config = config
        self.node_filter = node_filter
        self.used_configurations = ["Defaults"]
        self.missing_configurations = []
        self.ignored_non_custom_configurations = []

    def configure_network(self, graph_proto, model_config, subgraph_hier):
        if hasattr(graph_proto, "graph"):
            graph = graph_proto.graph
        else:
            graph = graph_proto

        for node in graph.node:
            if not self.node_filter(node):
                continue

            config_key = (
                node.name if subgraph_hier is None else str(subgraph_hier) + "_" + node.name
            )

            try:
                node_config = model_config[config_key].copy()
            except KeyError:
                self.missing_configurations += [node.name]
                node_config = {}

            if is_custom_op(node.domain, node.op_type):
                inst = getCustomOp(node)

                if "Defaults" in model_config.keys():
                    default_values = []
                    for key, value in model_config["Defaults"].items():
                        assert len(value) % 2 == 0
                        if key not in model_config:
                            for val, op in zip(value[::2], value[1::2]):
                                default_values.append((key, val, op))
                                assert not (op == "all" and len(value) > 2)
                    default_configs = {
                        key: val
                        for key, val, op in default_values
                        if op == "all" or node.op_type in op
                    }
                    for attr_name, value in default_configs.items():
                        inst.set_nodeattr(attr_name, value)

                for attr_name, value in node_config.items():
                    inst.set_nodeattr(attr_name, value)

                if node_config:
                    self.used_configurations += [config_key]
            elif node_config:
                self.ignored_non_custom_configurations += [(config_key, node.op_type)]

            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH:
                    if subgraph_hier is None:
                        new_hier = node.name
                    else:
                        new_hier = str(subgraph_hier) + "_" + node.name
                    new_hier = new_hier + "_" + attr.name
                    self.configure_network(attr.g, model_config, subgraph_hier=new_hier)

    def apply(self, model):
        if isinstance(self.config, dict):
            model_config = self.config
        else:
            with open(self.config, "r") as f:
                model_config = json.load(f)

        self.configure_network(model.model.graph, model_config, subgraph_hier=None)

        # Shared subgraphs (an If node's) report a node more than once.
        unique_missing = list(dict.fromkeys(self.missing_configurations))
        if len(unique_missing) > 0:
            warnings.warn("\nNo HW configuration for nodes: " + ", ".join(unique_missing))

        unique_non_custom = list(dict.fromkeys(self.ignored_non_custom_configurations))
        if len(unique_non_custom) > 0:
            formatted_non_custom = [
                "{} ({})".format(config_key, op_type) for config_key, op_type in unique_non_custom
            ]
            warnings.warn(
                "\nHW configurations for non-custom nodes were ignored: "
                + ", ".join(formatted_non_custom)
                + ". Configs can only be applied to custom ops."
            )

        ignored_configurations = [
            config_key for config_key, _ in self.ignored_non_custom_configurations
        ]
        unused_configs = [
            x
            for x in model_config
            if x not in self.used_configurations
            and x not in ignored_configurations
            and x != "Defaults"
        ]
        if len(unused_configs) > 0:
            warnings.warn("\nUnused HW configurations: " + ", ".join(unused_configs))

        return (model, False)
