# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Binding the decomposed MVAU's two Regions to the hardware that covers them.

This is the only thing entitled to say which node fills which Kernel role: it
owns the Network, so it knows that ``replay`` is the replay node and ``compute``
is the dot-product node.  Neither Kernel is asked to work that out, and neither
could -- inferring it from Region shape is exactly the guess the coverage
contract exists to prevent.

It also resolves named source roots.  A Kernel declares ``finnlib/rtl/...``; a
checkout is what turns that into a path, and where FinnLib lives is a property
of this installation rather than of the hardware.
"""

from __future__ import annotations

from finn.dataflow.authoring.design import DesignRealization
from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.physical import MVAUElaborationError
from finn.dataflow.mvau.hardware.source import (
    FINNLIB_DEFAULT_SUBDIRECTORY,
    FINNLIB_ROOT_VARIABLE,
    finnlib_root,
    resolved_manifest,
    source_roots,
    verify_manifest,
)
from finn.dataflow.mvau.projection import MVAUResolvedDesign

_BINDING_PATH = QualifiedPath("hardware.mvau.decomposed")


def _fail(
    code: str, message: str, values: tuple[tuple[str, object], ...] = ()
) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _BINDING_PATH, message, values),)
    )


def bind_decomposed(resolved: MVAUResolvedDesign) -> DesignRealization:
    """Return the configured Kernels of a resolved production DotProductDesign."""

    design_path = MVAU_DESIGN_INVENTORY.inventory.design_path
    if design_path is None or design_path not in resolved.point.design_space.decisions:
        raise _fail(
            "mvau-decomposed-legacy-point",
            "production binding requires the v6 MVAU DataflowDesign inventory",
        )
    selected_design = MVAU_DESIGN_INVENTORY.inventory.selected(resolved.point)
    if not isinstance(selected_design, Decided) or selected_design.value.id != DotProductDesign.id:
        raise _fail(
            "mvau-decomposed-design-unsupported",
            "this hardware realizes only DotProductDesign",
        )
    realization = MVAU_DESIGN_INVENTORY.inventory.realize(resolved.engine, resolved.point)
    if not isinstance(realization, Decided):
        raise MVAUElaborationError(realization.findings)
    return realization.value


__all__ = [
    "FINNLIB_DEFAULT_SUBDIRECTORY",
    "FINNLIB_ROOT_VARIABLE",
    "bind_decomposed",
    "finnlib_root",
    "resolved_manifest",
    "source_roots",
    "verify_manifest",
]
