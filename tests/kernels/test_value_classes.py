# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The default snapshot's contract: every frozen dataclass of ``finn.kernels`` holds
immutable values, so the engine shares an instance as its own snapshot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from support.value_classes import frozen_dataclasses, mutable_fields

from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.policies import Ranking
from finn.kernels.port_model import ServedPort
from finn.kernels.values.semantics import IntegerTensorValue

ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    """This process's environment, ``src`` alone on the import path."""
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


#: Types outside the checker's rule that are immutable, and why.
IMMUTABLE: dict[object, str] = {
    QONNXDataType: "QONNX's datatypes are interned and refuse attribute assignment",
    Ranking: "a first valid policy's ranking is a view by class access, its declaration, "
    "fixed once its class is made (finn.kernels.policies.FirstValid)",
    IntegerTensorValue: "an integer tensor is immutable, a copy the value itself, and refuses "
    "attribute assignment (finn.kernels.values.semantics); MatMul's known weights hold one",
}


def test_every_frozen_value_class_holds_immutable_values() -> None:
    # A declaration of a kernel (a port) is no value the engine holds: it lives in a class
    # body, and holds the kernel's members it reads.
    # A served port is a declaration too: it holds its child's, which its kernel lowers.
    classes = tuple(cls for cls in frozen_dataclasses("finn.kernels") if cls is not ServedPort)
    assert classes
    assert mutable_fields(classes, IMMUTABLE) == []


def test_a_kernel_s_module_alone_registers_finn_s_value_types() -> None:
    # The engine admits a QONNX datatype only registered (``value_type``): importing one
    # kernel's module, and nothing else of FINN, registers it, so its design space opens.
    script = (
        "from qonnx.core.datatype import DataType\n"
        "from finn.core.space import design_space\n"
        "from finn.core.space._value_types import registered\n"
        "from finn.dataflow.tensor import ScalarEncoding, Tensor\n"
        "from finn.dataflow.traversal import BeatSequence, vector_major\n"
        "from finn.kernels.vpc import VpcKernel\n"
        "tensor = Tensor((4, 6), ScalarEncoding(DataType['INT4']))\n"
        "point = design_space(VpcKernel(input_tensor=tensor,\n"
        "    arriving=BeatSequence(vector_major((4, 6), 2)),\n"
        "    leaving=BeatSequence(vector_major((4, 6), 3))))\n"
        "print(sorted(kind.__name__ for kind in registered()), point.input_tensor == tensor)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT, env=_env()
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["['BaseDataType',", "'IntegerTensorValue']", "True"]
