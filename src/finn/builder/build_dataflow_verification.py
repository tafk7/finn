# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition's Python check (step_verify_partition_python): the parent graph, its
partition node running the KernelOps, against the model before conversion, both executed
by FINN's execution on each input of an inputs file, value for value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.partition import partition_body
from finn.core.onnx_exec import execute_onnx

_LOG = logging.getLogger(__name__)

#: Where the check saves the partition's outputs, under the output directory.
VERIFICATION_DIR = Path("verification_output")


def verify_partition_python(
    parent: ModelWrapper,
    reference_file: Path,
    inputs_file: Path,
    output_dir: Path,
) -> bool:
    """Execute the parent graph ``parent`` and the model in ``reference_file`` on each
    input in ``inputs_file`` (an .npy file, one input a row), and compare the partition's
    outputs value for value. Saves each input's outputs as
    verification_output/verify_partition_python_<index>_<SUCCESS|FAIL>.npz under
    ``output_dir``, logs each output that differs as a warning, and returns whether all
    matched."""
    if not inputs_file.is_file():
        raise FileNotFoundError(f"verify_input_npy: {inputs_file} is no file")
    saved = output_dir / VERIFICATION_DIR
    saved.mkdir(parents=True, exist_ok=True)
    node, _, _ = partition_body(parent)
    reference = ModelWrapper(str(reference_file))
    reference_input = reference.graph.input[0].name
    shape = reference.get_tensor_shape(reference_input)
    if shape is None:
        raise ValueError(f"{reference_file}: its input {reference_input} states no shape")
    outputs = list(node.output)
    inputs = np.load(inputs_file)
    all_match = True
    for index in range(inputs.shape[0]):
        frame = inputs[index : index + 1].reshape((1,) + tuple(shape)[1:])
        built = execute_onnx(parent, {parent.graph.input[0].name: frame}, True)
        expected = execute_onnx(reference, {reference_input: frame}, True)
        mismatched = [name for name in outputs if not np.array_equal(built[name], expected[name])]
        for name in mismatched:
            _LOG.warning(
                f"Python check: input {index}: {name} differs from the reference's in "
                f"{int(np.sum(built[name] != expected[name]))} of {built[name].size} values"
            )
        result = "FAIL" if mismatched else "SUCCESS"
        all_match = all_match and not mismatched
        np.savez(
            saved / f"verify_partition_python_{index}_{result}.npz",
            **{name: built[name] for name in outputs},
        )
    _LOG.info(
        f"Python check: the partition's outputs {outputs} on {inputs.shape[0]} "
        f"inputs of {inputs_file}, against {reference_file.name}"
    )
    return all_match


__all__ = ["VERIFICATION_DIR", "verify_partition_python"]
