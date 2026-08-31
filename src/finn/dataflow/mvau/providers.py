# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU elaboration dispatch over the reviewed design inventory."""

from __future__ import annotations

from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.mvau.designs.dot_product import compose_dot_product_design
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.physical import MVAUElaborationError, MVAUPhysicalElaboration
from finn.dataflow.mvau.source import MVAUResolvedDesign

_DISPATCH_PATH = QualifiedPath("mvau.elaboration.dispatch")


def _fail(code: str, message: str) -> MVAUElaborationError:
    return MVAUElaborationError((Finding(FindingKind.LIMITATION, code, _DISPATCH_PATH, message),))


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
