############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
#
# Inlined verbatim from microsoft/brainsmith:brainsmith/dataflow/spec_helpers.py
# @ 38faaf9 (MIT). Temporary vendored copy — see Stage 3 (source model) in
# kernel-design/kernel-final-design/KERNEL_REFACTOR_PLAN.md for the de-vendoring plan.
############################################################################

"""Datatype range helper for building kernel schemas.

Only ``smallest_datatype_for_range`` survives the prune — it is the single symbol the
built engine imports (``ops/mvau/op.py``). The rest of the vendored spec-helper surface
(dimension/datatype derivation, context-aware binary-op builders) modelled the same
concepts as the live ``space/`` engine and is removed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qonnx.core.datatype import BaseDataType


def smallest_datatype_for_range(min_val: float, max_val: float) -> "BaseDataType":
    """Find smallest integer datatype that fits the given range.

    Consolidates the logic used throughout the codebase for selecting
    optimal datatypes based on value ranges.

    Uses array-based checking to correctly handle signed/unsigned detection,
    matching FINN's proven approach.

    Args:
        min_val: Minimum value to represent
        max_val: Maximum value to represent

    Returns:
        Smallest QONNX DataType that can represent the range

    Example:
        dt = smallest_datatype_for_range(-100, 127)  # Returns DataType["INT8"]
        dt = smallest_datatype_for_range(0, 255)     # Returns DataType["UINT8"]
        dt = smallest_datatype_for_range(-1016, 1024)  # Returns DataType["INT12"]
    """
    import numpy as np
    from qonnx.core.datatype import DataType

    # Create array with both bounds (matches FINN's approach)
    # This ensures we correctly detect when range includes negative values
    vals = np.array([min_val, max_val], dtype=np.float64)

    # Verify values are integers
    for v in vals:
        assert int(v) == v, f"Non-integer value in range: {v}"

    # Iterate through accumulator candidates (sorted by size, prefers unsigned)
    for k in DataType.get_accumulator_dt_cands():
        dt = DataType[k]

        # Skip unsupported types
        if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
            continue

        # Check if datatype can represent BOTH bounds
        if (dt.min() <= vals).all() and (vals <= dt.max()).all():
            return dt

    # Fallback for extreme ranges
    if min_val >= 0:
        return DataType["UINT64"]
    else:
        return DataType["INT64"]
