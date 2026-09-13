# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The production selected-transform authorization catalogue."""

from finn.dataflow.ops.mvau.selected import (
    MVAU_CONSTRUCTION_FAMILY,
    MVAU_CONSTRUCTION_VERSION,
    MVAU_SELECTED_TRANSFORM_AUTHORIZATIONS,
)
from finn.dataflow.ops.replay.selected import (
    REPLAY_CONSTRUCTION_FAMILY,
    REPLAY_CONSTRUCTION_VERSION,
    REPLAY_SELECTED_TRANSFORM_AUTHORIZATIONS,
)
from finn.dataflow.ops.selected_transforms import SelectedTransformRegistry

DEFAULT_SELECTED_TRANSFORMS = SelectedTransformRegistry(
    {
        (
            REPLAY_CONSTRUCTION_FAMILY,
            REPLAY_CONSTRUCTION_VERSION,
        ): REPLAY_SELECTED_TRANSFORM_AUTHORIZATIONS,
        (
            MVAU_CONSTRUCTION_FAMILY,
            MVAU_CONSTRUCTION_VERSION,
        ): MVAU_SELECTED_TRANSFORM_AUTHORIZATIONS,
    }
)

__all__ = ["DEFAULT_SELECTED_TRANSFORMS"]
