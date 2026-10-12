# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's Python verification of the partition, ``partition_python``: the parent
graph, its partition node running the KernelOps, against the model the kernel path
started from.

The reference is the model before conversion, which the build's conversion step saves
when the verification is asked (``save_reference``, ``REFERENCE_FILE`` under the output
directory). ``verify_partition_python`` reads it back after the cut, executes both
models with FINN's execution (``finn.core.onnx_exec``) on each input of the inputs file,
and compares the partition's outputs with the reference's, value for value. The
reference is read from the output directory, so a build that starts after the
conversion step reads the one an earlier run left there, if any.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.onnx_exec import execute_onnx
from finn.custom_op.partition.kernel_partitions import partition_body

_LOG = logging.getLogger(__name__)

#: Where the conversion step saves the model before conversion, the reference,
#: under the output directory.
REFERENCE_FILE = Path("intermediate_models") / "kernel_path_source.onnx"

#: Where the verification saves the partition's outputs, under the output directory.
VERIFICATION_DIR = Path("verification_output")


def save_reference(model: ModelWrapper, output_dir: Path) -> None:
    """Save ``model``, the graph the kernel path converts, as the reference
    (``REFERENCE_FILE`` under ``output_dir``)."""
    reference = output_dir / REFERENCE_FILE
    reference.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(reference))


def verify_partition_python(
    parent: ModelWrapper, output_dir: Path, inputs_file: str | os.PathLike[str]
) -> bool:
    """Execute the parent graph ``parent``, its partition node running the KernelOps, on
    each input in ``inputs_file`` (an .npy file, one input a row), and compare the
    partition's outputs value for value with those of the reference saved in
    ``output_dir`` (``save_reference``). Saves each input's partition outputs as
    verification_output/verify_partition_python_<index>_<SUCCESS|FAIL>.npz there, logs
    each output that differs as a warning, and returns whether all matched. A missing
    reference is refused with FileNotFoundError."""
    reference_file = output_dir / REFERENCE_FILE
    if not reference_file.is_file():
        raise FileNotFoundError(
            f"{reference_file}: the kernel path's source, which step_convert_to_hw keeps when "
            "PARTITION_PYTHON is asked, is not there (did the build start after it?)"
        )
    saved = output_dir / VERIFICATION_DIR
    saved.mkdir(parents=True, exist_ok=True)
    node, _, _ = partition_body(parent)
    reference = ModelWrapper(str(reference_file))
    reference_input = reference.graph.input[0].name
    shape = reference.get_tensor_shape(reference_input)
    if shape is None:
        raise ValueError(f"partition_python: the source's input {reference_input} states no shape")
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
                f"partition_python: input {index}: {name} differs from the source's in "
                f"{int(np.sum(built[name] != expected[name]))} of {built[name].size} values"
            )
        result = "FAIL" if mismatched else "SUCCESS"
        all_match = all_match and not mismatched
        np.savez(
            saved / f"verify_partition_python_{index}_{result}.npz",
            **{name: built[name] for name in outputs},
        )
    _LOG.info(
        f"partition_python: the partition's outputs {outputs} on {inputs.shape[0]} "
        f"inputs of {inputs_file}, against the model the kernel path started from"
    )
    return all_match


__all__ = ["VERIFICATION_DIR", "REFERENCE_FILE", "save_reference", "verify_partition_python"]
