# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Containers: the element types an executed ONNX graph holds its tensors' values in.

A tensor's annotation (qonnx's datatype) states which values it may take; its
container is the ONNX element type of its value_info, or an initializer's data
type, which execution allocates and computes in. A container holds every integer
up to a magnitude exactly (``EXACT_UP_TO``): float32 up to 2**24, float64 up to
2**53, an integer container its range. Past it, a float container rounds and an
integer one wraps.

The one statement of that fact: the graph-preparation phase widens the integer
regions that need it (P6) and its checkpoint checks every integer tensor against
it; a KernelOp's domain step (``KernelOp.exact``) reads it to say where its
reference is exact against ONNX executed in its operands' container; the harness
draws a tensor's inputs as its container holds them.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from onnx import TensorProto, helper

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

#: The largest magnitude up to which each container (an ONNX element type) holds every
#: integer exactly.
EXACT_UP_TO: Mapping[int, int] = MappingProxyType(
    {
        TensorProto.FLOAT16: 2**11,
        TensorProto.FLOAT: 2**24,
        TensorProto.DOUBLE: 2**53,
        TensorProto.INT8: 2**7 - 1,
        TensorProto.UINT8: 2**8 - 1,
        TensorProto.INT16: 2**15 - 1,
        TensorProto.UINT16: 2**16 - 1,
        TensorProto.INT32: 2**31 - 1,
        TensorProto.UINT32: 2**32 - 1,
        TensorProto.INT64: 2**63 - 1,
        TensorProto.UINT64: 2**64 - 1,
    }
)


def name(element_type: int) -> str:
    """A container's ONNX name (``FLOAT``, ``DOUBLE``)."""
    return str(TensorProto.DataType.Name(element_type))


def container(model: ModelWrapper, tensor: str) -> int | None:
    """``tensor``'s container: an initializer's data type, else its value_info's
    element type; None when the graph states neither."""
    for init in model.graph.initializer:
        if init.name == tensor:
            return int(init.data_type)
    info = model.get_tensor_valueinfo(tensor)
    if info is None or not info.type.HasField("tensor_type"):
        return None
    element_type = int(info.type.tensor_type.elem_type)
    return element_type or None


def exact_up_to(element_type: int | None) -> int | None:
    """The largest magnitude ``element_type`` holds every integer up to; None for a
    container this module does not know (or none)."""
    return None if element_type is None else EXACT_UP_TO.get(element_type)


def numpy_type(element_type: int) -> np.dtype[Any]:
    """The numpy element type execution holds a container's values in."""
    return np.dtype(helper.tensor_dtype_to_np_dtype(element_type))


def held(values: npt.NDArray[np.int64], element_type: int) -> npt.NDArray[np.int64]:
    """Integer ``values`` as ``element_type`` holds them: each moved to the nearest value
    the container holds toward zero, so a value of a type's range stays in it."""
    dtype = numpy_type(element_type)
    if dtype.kind != "f":
        return values
    stored = values.astype(dtype)
    beyond = np.abs(stored.astype(np.int64)) > np.abs(values)
    stored[beyond] = np.nextafter(stored[beyond], dtype.type(0))
    found: npt.NDArray[np.int64] = stored.astype(np.int64)
    return found


__all__ = [
    "EXACT_UP_TO",
    "container",
    "exact_up_to",
    "held",
    "name",
    "numpy_type",
]
