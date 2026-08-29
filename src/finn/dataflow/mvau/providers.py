# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatch elaboration to the provider the selected Kernel declared.

A Kernel names its providers; this maps the declared provider id to the code
that realizes it.  Callers ask for an elaboration of a resolved point and never
choose an implementation themselves -- choosing one is what the Kernel
selection already did.

The table is keyed on the *provider* id rather than the Kernel id, because that
is the identity the Kernel publishes and the identity the elaboration records
in its origin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.kernels import SelectedKernel
from finn.dataflow.mvau.decomposed import DOT_PRODUCT_PROVIDER
from finn.dataflow.mvau.decomposed_provider import elaborate_mvau_decomposed
from finn.dataflow.mvau.compute_kernels import SOFT_VECTOR_PROVIDER_ID
from finn.dataflow.mvau.elaboration import (
    MVAUElaborationError,
    MVAUPhysicalElaboration,
    elaborate_mvau_rtl_softvec,
)
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.ops.mvau import MVAU_COMPUTE_SELECTION

_DISPATCH_PATH = QualifiedPath("provider.mvau.dispatch")

MVAUElaborator = Callable[[MVAUResolvedDesign], MVAUPhysicalElaboration]

#: Provider id -> the code that realizes it.
MVAU_COMPUTE_ELABORATORS: Mapping[str, MVAUElaborator] = MappingProxyType(
    {
        SOFT_VECTOR_PROVIDER_ID: elaborate_mvau_rtl_softvec,
        DOT_PRODUCT_PROVIDER: elaborate_mvau_decomposed,
    }
)


def elaborate_mvau(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate through whichever provider the selected compute Kernel declares."""

    answer = resolved.engine.query_property(
        resolved.point, MVAU_COMPUTE_SELECTION.paths.selected_kernel
    )
    if not isinstance(answer, Decided) or not isinstance(answer.value, SelectedKernel):
        raise MVAUElaborationError(
            (
                Finding(
                    FindingKind.BLOCKER,
                    "mvau-dispatch-kernel-unresolved",
                    _DISPATCH_PATH,
                    "the selected compute Kernel must resolve before elaboration",
                ),
            )
        )
    kernel_id = answer.value.kernel_id
    declared = tuple(item.id for item in MVAU_COMPUTE_SELECTION.kernel(kernel_id).providers)
    covered = tuple(item for item in declared if item in MVAU_COMPUTE_ELABORATORS)
    if len(covered) != 1:
        raise MVAUElaborationError(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-dispatch-no-covered-provider"
                    if not covered
                    else "mvau-dispatch-ambiguous-provider",
                    _DISPATCH_PATH,
                    f"{kernel_id} declares {len(covered)} providers this build can realize",
                    (("declared", declared), ("covered", covered)),
                ),
            )
        )
    return MVAU_COMPUTE_ELABORATORS[covered[0]](resolved)


__all__ = ["MVAU_COMPUTE_ELABORATORS", "MVAUElaborator", "elaborate_mvau"]
