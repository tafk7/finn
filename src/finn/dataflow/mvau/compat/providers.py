# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Provider-era MVAU elaboration dispatch retained for compatibility tests.

Two paths meet here, because the migration is half done by design:

- the **decomposed** member elaborates through bound physical Kernels.  It
  declares no provider, and nothing here asks it for one.
- the **fused RTL** members still elaborate through the legacy provider
  inventory.  They are retained until the decomposed path is packaged and
  numerically validated, so their dispatch is retained with them.

Callers ask for an elaboration of a resolved point and never choose an
implementation themselves -- choosing one is what the Kernel selection already
did.  When the fused members retire, so does the provider half of this module
and every mention of ``KernelProvider`` in the migrated path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.kernels import SelectedKernel
from finn.dataflow.mvau.compute_kernels import SOFT_VECTOR_PROVIDER_ID
from finn.dataflow.mvau.elaboration import (
    MVAUElaborationError,
    MVAUPhysicalElaboration,
    elaborate_mvau_rtl_softvec,
)
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.mvau.compat.operation import MVAU_COMPUTE_SELECTION

_DISPATCH_PATH = QualifiedPath("mvau.elaboration.dispatch")

MVAUElaborator = Callable[[MVAUResolvedDesign], MVAUPhysicalElaboration]

#: Provider id -> the code that realizes it, for the members that have not.
MVAU_PROVIDER_ELABORATORS: Mapping[str, MVAUElaborator] = MappingProxyType(
    {SOFT_VECTOR_PROVIDER_ID: elaborate_mvau_rtl_softvec}
)


def _fail(code: str, message: str, values: tuple[tuple[str, object], ...] = ()) -> Exception:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _DISPATCH_PATH, message, values),)
    )


def elaborate_mvau(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate the selected compute Kernel, by whichever route it has."""

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
    covered = tuple(item for item in declared if item in MVAU_PROVIDER_ELABORATORS)
    if len(covered) != 1:
        raise _fail(
            "mvau-dispatch-no-covered-provider"
            if not covered
            else "mvau-dispatch-ambiguous-provider",
            f"{kernel_id} declares {len(covered)} providers this build can realize",
            (("declared", declared), ("covered", covered)),
        )
    return MVAU_PROVIDER_ELABORATORS[covered[0]](resolved)


__all__ = [
    "MVAU_PROVIDER_ELABORATORS",
    "MVAUElaborator",
    "elaborate_mvau",
]
