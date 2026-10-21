# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The default snapshot's contract: every frozen dataclass of ``finn.kernels`` holds
immutable values, so the engine shares an instance as its own snapshot."""

from __future__ import annotations

from support.value_classes import frozen_dataclasses, mutable_fields

from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.base import KernelDeclaration

#: Types outside the checker's rule that are immutable, and why.
IMMUTABLE: dict[object, str] = {
    QONNXDataType: "QONNX's datatypes are interned and refuse attribute assignment",
}


def test_every_frozen_value_class_holds_immutable_values() -> None:
    # A declaration of a kernel (a port) is no value the engine holds: it lives in a class
    # body, and holds the kernel's members it reads.
    classes = tuple(
        cls for cls in frozen_dataclasses("finn.kernels") if not issubclass(cls, KernelDeclaration)
    )
    assert classes
    assert mutable_fields(classes, IMMUTABLE) == []
