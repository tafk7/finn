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
from finn.dataflow.hardware import HardwareKernel, bind_hardware_kernel, bound_regions
from finn.dataflow.hardware.kernel import HardwareKernelDeclaration
from finn.dataflow.mvau.compute_kernels import DECOMPOSED_MVAU_KERNELS, MVAU_COMPUTE_SELECTION
from finn.dataflow.mvau.compute_pool import MVAUComputeKernelId
from finn.dataflow.mvau.decomposed import DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.mvau.designs.dot_product import MVAU_DOT_PRODUCT_DESIGN
from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.elaboration import MVAUElaborationError
from finn.dataflow.mvau.hardware.dotp_axi import FINNLIB_ROOT
from finn.dataflow.mvau.hardware.replay_buffer import FINN_ROOT
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau import NetworkRef

_BINDING_PATH = QualifiedPath("hardware.mvau.decomposed")

#: Where ``fetch-repos.sh`` places the pinned FinnLib checkout.
FINNLIB_DEFAULT_SUBDIRECTORY = "deps/finnlib"

#: Environment override for a local working clone.
FINNLIB_ROOT_VARIABLE = "FINNLIB_ROOT"

#: Which Kernel role each Network node fills.  Stated here, once.
COMPUTE_ROLE = "compute"
REPLAY_ROLE = "replay"


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


def _bind(
    resolved: MVAUResolvedDesign,
    declaration: HardwareKernelDeclaration,
    role: str,
    node_id: str,
    network: DataflowNetwork,
) -> HardwareKernel:
    """Bind one Kernel to the node this assembly says fills its role."""

    answer = bind_hardware_kernel(
        resolved.engine,
        declaration,
        resolved.point,
        bound_regions(((role, node_id, network.node(node_id).region),)),
    )
    if not isinstance(answer, Decided):
        raise MVAUElaborationError(
            answer.findings
            or (
                Finding(
                    FindingKind.BLOCKER,
                    "mvau-decomposed-kernel-unbound",
                    _BINDING_PATH,
                    f"{declaration.id} did not bind to the {role} node",
                ),
            )
        )
    return answer.value


def bind_decomposed(resolved: MVAUResolvedDesign) -> DesignRealization:
    """Bind the decomposed slice, refusing anything this hardware does not cover.

    The Region checks that used to live here are gone: ``bind_hardware_kernel``
    now verifies that the node's Region is the one the Kernel's coverage handle
    derives, which is the same claim made once instead of twice.
    """

    design_path = MVAU_DESIGN_INVENTORY.inventory.design_path
    if design_path is not None and design_path in resolved.point.design_space.decisions:
        selected_design = MVAU_DESIGN_INVENTORY.inventory.selected(resolved.point)
        if not isinstance(selected_design, Decided) or selected_design.value.id != "dot_product":
            raise _fail(
                "mvau-decomposed-design-unsupported",
                "this hardware realizes only DotProductDesign",
            )
        realization = MVAU_DESIGN_INVENTORY.inventory.realize(resolved.engine, resolved.point)
        if not isinstance(realization, Decided):
            raise MVAUElaborationError(realization.findings)
        return realization.value

    selected = resolved.engine.query_property(
        resolved.point, MVAU_COMPUTE_SELECTION.paths.selected_kernel
    )
    if not isinstance(selected, Decided):
        raise _fail(
            "mvau-decomposed-compute-unresolved",
            "the selected compute Kernel must resolve before binding",
        )
    if getattr(selected.value, "kernel_id", None) != MVAUComputeKernelId.DOT_PRODUCT.value:
        raise _fail(
            "mvau-decomposed-kernel-unsupported",
            "this hardware realizes only the decomposed dot-product Region",
        )
    if not isinstance(resolved.result, NetworkRef):
        raise _fail(
            "mvau-decomposed-result-not-a-network",
            "the decomposed Region must resolve to a replay-plus-compute Network",
        )
    network = resolved.result.network
    node_ids = {node.id for node in network.nodes}
    if node_ids != {REPLAY_NODE, DOT_PRODUCT_NODE}:
        raise _fail(
            "mvau-decomposed-network-unsupported",
            "this hardware builds the two-node replay-plus-compute Network only",
            (("nodes", tuple(sorted(node_ids))),),
        )
    configured = {
        "compute": _bind(
            resolved,
            DECOMPOSED_MVAU_KERNELS.dot_product_hardware,
            COMPUTE_ROLE,
            DOT_PRODUCT_NODE,
            network,
        ),
        "replay": _bind(
            resolved,
            DECOMPOSED_MVAU_KERNELS.replay_hardware,
            REPLAY_ROLE,
            REPLAY_NODE,
            network,
        ),
    }
    validated = MVAU_DOT_PRODUCT_DESIGN.design.validate_realization(
        network,
        configured,
        active_placements=("compute", "replay"),
    )
    if not isinstance(validated, Decided):
        raise MVAUElaborationError(validated.findings)
    return validated.value


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
    "COMPUTE_ROLE",
    "FINNLIB_DEFAULT_SUBDIRECTORY",
    "FINNLIB_ROOT_VARIABLE",
    "REPLAY_ROLE",
    "bind_decomposed",
    "finnlib_root",
    "resolved_manifest",
    "source_roots",
    "verify_manifest",
]
