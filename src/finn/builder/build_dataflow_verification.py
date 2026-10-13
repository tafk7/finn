# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition's Python check over a finished build's output directory: a utility a
test or a user runs, not a step.

``verify_partition_python(output_dir, inputs_file)`` compares two models the build
saved (``intermediate_models/``, so ``save_intermediate_models`` on): its last, the
parent graph whose partition node runs the KernelOps, against the one saved before
step_convert_to_hw, both executed on each input of an inputs file. The steps are read
from the build's manifest. The partition's outputs are compared value for value by the
graph comparison (``finn.analysis.comparison``), and its result returned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.comparison import Comparison, compared_with
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.core.graph.partition import partition_body


def verify_partition_python(
    output_dir: str | os.PathLike[str], inputs_file: str | os.PathLike[str]
) -> Comparison:
    """Compare the partition's outputs in the last model the build into ``output_dir``
    saved with the same tensors of the model it saved before step_convert_to_hw, on each
    input in ``inputs_file`` (an .npy file, one input a row)."""
    output = Path(output_dir)
    steps = json.loads((output / MANIFEST_FILE).read_text())["steps"]
    names = [step["name"] for step in steps if step["status"] == "completed"]
    if "step_convert_to_hw" not in names[1:]:
        raise ValueError(
            f"{output}: the build's manifest lists no step_convert_to_hw after another "
            "completed step"
        )
    saved = output / "intermediate_models"
    reference = ModelWrapper(str(saved / f"{names[names.index('step_convert_to_hw') - 1]}.onnx"))
    parent = ModelWrapper(str(saved / f"{names[-1]}.onnx"))
    reference_input = reference.graph.input[0].name
    shape = reference.get_tensor_shape(reference_input)
    if shape is None:
        raise ValueError(f"the reference's input {reference_input} states no shape")
    rows = np.load(inputs_file)
    inputs = [
        {parent.graph.input[0].name: rows[index : index + 1].reshape((1, *shape[1:]))}
        for index in range(rows.shape[0])
    ]
    node, _, _ = partition_body(parent)
    tensors = [(name, name) for name in node.output]
    return compared_with(reference, inputs, tensors=tensors)(parent)


__all__ = ["verify_partition_python"]
