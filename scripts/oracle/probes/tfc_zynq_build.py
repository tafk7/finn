# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2's Zynq build in the HWCustomOp flow, Vivado and Vitis HLS stubbed: the folded
dataflow partition (``_tfc.folded``) less its label select, which the kernel path keeps
on the host, through ``ZynqBuild`` for Ultra96 at 5 ns, as Z0's legacy dry run built
its model. Captured: the IODMA_hls nodes ``InsertIODMA`` inserted, the HLS top each
generates (``PrepareIP``), the block design ``MakeZYNQProject`` writes (its
``ip_config.tcl``'s per-partition lines), and the whole ``ip_config.tcl`` as
``tfc_zynq_build.ip_config.tcl``; every path normalised (``_probe.normalised``)."""

from _probe import arguments, normalised, write
from _tfc import BOARD, PERIOD_NS, folded
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow import templates
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild

IODMA_ATTRIBUTES = (
    "numInputVectors",
    "NumChannels",
    "dataType",
    "intfWidth",
    "streamWidth",
    "direction",
)

raw, captures = arguments()
model, _ = folded(captures, Path.cwd())
# The label select on the host: the partition ends at the logits.
(label_select,) = [node for node in model.graph.node if node.op_type.startswith("LabelSelect")]
assert label_select is model.graph.node[-1]
logits = label_select.input[0]
(output,) = model.graph.output
info = model.get_tensor_valueinfo(logits)
output.CopyFrom(info)
model.graph.value_info.remove(info)
model.graph.node.remove(label_select)
model = model.transform(ZynqBuild(BOARD, PERIOD_NS))

partitions, iodmas, code = [], {}, {}
for node in model.graph.node:
    partition = getCustomOp(node)
    body = ModelWrapper(partition.get_nodeattr("model"))
    partitions.append(
        {
            "name": node.name,
            "instance": partition.get_nodeattr("instance_name"),
            "nodes": [[each.name, each.op_type] for each in body.graph.node],
        }
    )
    for each in body.graph.node:
        if each.op_type == "IODMA_hls":
            op = getCustomOp(each)
            attributes = {name: op.get_nodeattr(name) for name in IODMA_ATTRIBUTES}
            iodmas[attributes["direction"]] = {"node": each.name, **attributes}
            top = Path(op.get_nodeattr("code_gen_dir_ipgen")) / f"top_{each.name}.cpp"
            code[each.name] = normalised(top.read_text())

# The block design: the lines MakeZYNQProject puts in place of the template's config.
script = normalised(
    (Path(model.get_metadata_prop("vivado_pynq_proj")) / "ip_config.tcl").read_text()
)
(raw / "tfc_zynq_build.ip_config.tcl").write_text(script)
template = templates.custom_zynq_shell_template.split("\n")
at = template.index("%s")
lines = script.split("\n")
block_design = lines[at : len(lines) - (len(template) - at - 1) - 1]
assert lines[len(lines) - (len(template) - at - 1) - 1] == ""
write(
    raw,
    {
        "dropped": [label_select.name, label_select.op_type],
        "partitions": partitions,
        "iodmas": iodmas,
        "iodma_code": code,
        "block_design": block_design,
        "ip_config": "tfc_zynq_build.ip_config.tcl",
    },
)
