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

import os
from pathlib import Path

from finn.dataflow.authoring.design import DesignRealization
from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.elaboration import MVAUElaborationError
from finn.dataflow.mvau.hardware.dotp_axi import FINNLIB_ROOT
from finn.dataflow.mvau.hardware.replay_buffer import FINN_ROOT
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.ops.mvau import NetworkRef

_BINDING_PATH = QualifiedPath("hardware.mvau.decomposed")

#: Where ``fetch-repos.sh`` places the pinned FinnLib checkout.
FINNLIB_DEFAULT_SUBDIRECTORY = "deps/finnlib"

#: Environment override for a local working clone.
FINNLIB_ROOT_VARIABLE = "FINNLIB_ROOT"


def _fail(
    code: str, message: str, values: tuple[tuple[str, object], ...] = ()
) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _BINDING_PATH, message, values),)
    )


def finnlib_root(finn_root: str | Path) -> Path:
    """The FinnLib checkout this build should compile against.

    ``FINNLIB_ROOT`` wins so a working clone can be used during development;
    otherwise it is the revision ``fetch-repos.sh`` pinned.
    """

    override = os.environ.get(FINNLIB_ROOT_VARIABLE)
    if override:
        return Path(override).resolve()
    return (Path(finn_root) / FINNLIB_DEFAULT_SUBDIRECTORY).resolve()


def source_roots(finn_root: str | Path, finnlib: str | Path | None = None) -> dict[str, Path]:
    """Resolve every named root the decomposed Kernels declare sources under."""

    root = Path(finn_root).resolve()
    return {
        FINN_ROOT: root,
        FINNLIB_ROOT: Path(finnlib).resolve() if finnlib is not None else finnlib_root(root),
    }


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
    if not isinstance(resolved.result, NetworkRef):
        raise _fail(
            "mvau-decomposed-result-not-a-network",
            "the decomposed Region must resolve to a replay-plus-compute Network",
        )
    realization = MVAU_DESIGN_INVENTORY.inventory.realize(resolved.engine, resolved.point)
    if not isinstance(realization, Decided):
        raise MVAUElaborationError(realization.findings)
    return realization.value


def resolved_manifest(
    realization: DesignRealization, roots: dict[str, Path]
) -> tuple[tuple[str, str], ...]:
    """Every declared source as ``(id, absolute path)``, in compile order.

    Ids stay stable across the migration -- ``compute.finn.0`` and so on -- so a
    staged build directory and the Phase 0 baseline both keep reading.
    """

    entries: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for placement in ("replay", "compute"):
        for source in realization.kernel(placement).sources:
            index = counts.get(source.root, 0)
            counts[source.root] = index + 1
            entries.append(
                (f"compute.{source.root}.{index}", str(roots[source.root] / source.path))
            )
    return tuple(entries)


def verify_manifest(entries: tuple[tuple[str, str], ...]) -> None:
    """Refuse a manifest that names files this checkout does not have.

    Reported here, with the paths in the finding, rather than surfacing later as
    an ``xelab`` error with no provenance.
    """

    missing = tuple(path for _, path in entries if not Path(path).is_file())
    if missing:
        raise _fail(
            "mvau-decomposed-source-missing",
            "declared decomposed RTL sources do not exist; is FinnLib fetched?",
            (("paths", missing),),
        )


__all__ = [
    "FINNLIB_DEFAULT_SUBDIRECTORY",
    "FINNLIB_ROOT_VARIABLE",
    "bind_decomposed",
    "finnlib_root",
    "resolved_manifest",
    "source_roots",
    "verify_manifest",
]
