# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU elaboration dispatch over the reviewed design inventory."""

from __future__ import annotations

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.hardware.composition import compose
from finn.dataflow.ops.mvau.physical import MVAUElaborationError, MVAUPhysicalElaboration
from finn.dataflow.ops.mvau.projection import MVAUResolvedDesign

_DISPATCH_PATH = QualifiedPath("mvau.elaboration.dispatch")


def _fail(code: str, message: str) -> MVAUElaborationError:
    return MVAUElaborationError((Finding(FindingKind.LIMITATION, code, _DISPATCH_PATH, message),))


def compose_dot_product_design(
    resolved: MVAUResolvedDesign,
    realization: DesignRealization,
) -> MVAUPhysicalElaboration:
    """Compose an already-realized DotProduct design under dispatch ownership."""

    if resolved.result.network != realization.network:
        raise _fail(
            "mvau-dispatch-network-mismatch",
            "the source envelope and DotProduct realization name different Networks",
        )
    return compose(resolved, realization)


def elaborate_mvau(resolved: MVAUResolvedDesign) -> MVAUPhysicalElaboration:
    """Elaborate the configured Kernels of the selected production design."""

    selected = MVAU_DESIGN_INVENTORY.inventory.selected(resolved.point)
    if not isinstance(selected, Decided):
        raise MVAUElaborationError(selected.findings)
    if selected.value.id != "dot_product":
        raise _fail(
            "mvau-dispatch-design-semantic-only",
            f"{selected.value.id} has no production physical Kernel",
        )
    realization = MVAU_DESIGN_INVENTORY.inventory.realize(resolved.engine, resolved.point)
    if not isinstance(realization, Decided):
        raise MVAUElaborationError(realization.findings)
    return compose_dot_product_design(resolved, realization.value)


__all__ = ["elaborate_mvau"]
