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

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.authoring.compiler import CompiledDataflowOperation
from finn.dataflow.authoring.inventory import DataflowDesignInventory
from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.op import realize_resolved_dataflow
from finn.dataflow.op_contracts import DataflowOpError
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.associations import MVAUResolvedDataflowOp
from finn.dataflow.ops.mvau.physical import MVAUElaborationError
from finn.dataflow.ops.mvau.artifacts.roots import (
    FINNLIB_DEFAULT_SUBDIRECTORY,
    FINNLIB_ROOT_VARIABLE,
    finnlib_root,
    resolved_manifest,
    source_roots,
    verify_manifest,
)

_BINDING_PATH = QualifiedPath("hardware.mvau.decomposed")


def _fail(
    code: str, message: str, values: tuple[tuple[str, object], ...] = ()
) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _BINDING_PATH, message, values),)
    )


def bind_decomposed(resolved: MVAUResolvedDataflowOp) -> DesignRealization:
    """Return the configured Kernels of a resolved production DotProductDesign."""

    compiled = resolved.compiled
    if isinstance(compiled, CompiledDataflowOperation) and compiled.inventory is not None:
        if resolved.selected_design_id != DotProductDesign.id:
            raise _fail(
                "mvau-decomposed-design-unsupported",
                "this hardware realizes only DotProductDesign",
            )
        try:
            return realize_resolved_dataflow(resolved)
        except DataflowOpError as error:
            raise MVAUElaborationError(error.findings) from error

    # Historical tests may still build a pre-adapter resolved value directly.
    # Keep that bridge lazy so no production MVAU import loads or depends on the
    # retired hand-assembled inventory.
    from finn.dataflow.ops.mvau.inventory import MVAU_DESIGN_INVENTORY  # noqa: PLC0415

    inventory: DataflowDesignInventory = MVAU_DESIGN_INVENTORY.inventory
    design_path = inventory.design_path
    if design_path is None or design_path not in resolved.point.design_space.decisions:
        raise _fail(
            "mvau-decomposed-legacy-point",
            "production binding requires the compiled MVAU DataflowDesign inventory",
        )
    selected_design = inventory.selected(resolved.point)
    if not isinstance(selected_design, Decided) or selected_design.value.id != DotProductDesign.id:
        raise _fail(
            "mvau-decomposed-design-unsupported",
            "this hardware realizes only DotProductDesign",
        )
    realization = inventory.realize(resolved.engine, resolved.point)
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
