# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU graph/build projection and resolved-source boundary."""

from finn.dataflow.mvau.source import (
    MVAU_LOGICAL_SOURCE_NODEATTRS,
    MVAUModelAccessor,
    MVAUProjectionContext,
    MVAUResolvedDesign,
    MVAUSourceAdapterError,
    MVAUSourceMappingEntry,
    MVAUSourceProjection,
    classify_mvau_dsp_block,
    project_mvau_build_problem,
    project_mvau_graph_source,
    project_mvau_source,
    resolve_mvau_point,
    start_mvau_projection,
    tensor_value_fingerprint,
)

__all__ = [
    "MVAU_LOGICAL_SOURCE_NODEATTRS",
    "MVAUModelAccessor",
    "MVAUProjectionContext",
    "MVAUResolvedDesign",
    "MVAUSourceAdapterError",
    "MVAUSourceMappingEntry",
    "MVAUSourceProjection",
    "classify_mvau_dsp_block",
    "project_mvau_build_problem",
    "project_mvau_graph_source",
    "project_mvau_source",
    "resolve_mvau_point",
    "start_mvau_projection",
    "tensor_value_fingerprint",
]
