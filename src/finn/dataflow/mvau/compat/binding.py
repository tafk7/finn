# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Provider-era binding of the decomposed semantic-Kernel selection."""

from __future__ import annotations

from finn.dataflow.authoring.design import DesignRealization
from finn.dataflow.design import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.hardware import HardwareKernel, bind_hardware_kernel, bound_regions
from finn.dataflow.hardware.kernel import HardwareKernelDeclaration
from finn.dataflow.mvau.compat.operation import MVAU_COMPUTE_SELECTION
from finn.dataflow.mvau.compute_kernels import DECOMPOSED_MVAU_KERNELS
from finn.dataflow.mvau.compute_pool import MVAUComputeKernelId
from finn.dataflow.mvau.decomposed import DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.ops.mvau.designs.dot_product import MVAU_DOT_PRODUCT_DESIGN
from finn.dataflow.mvau.compat.elaboration import MVAUElaborationError
from finn.dataflow.mvau.compat.source import MVAULegacyResolvedDesign as MVAUResolvedDesign
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau import NetworkRef

_BINDING_PATH = QualifiedPath("compat.hardware.mvau.decomposed")


def _fail(code: str, message: str) -> MVAUElaborationError:
    return MVAUElaborationError((Finding(FindingKind.LIMITATION, code, _BINDING_PATH, message),))


def _bind(
    resolved: MVAUResolvedDesign,
    declaration: HardwareKernelDeclaration,
    role: str,
    node_id: str,
    network: DataflowNetwork,
) -> HardwareKernel:
    answer = bind_hardware_kernel(
        resolved.engine,
        declaration,
        resolved.point,
        bound_regions(((role, node_id, network.node(node_id).region),)),
    )
    if not isinstance(answer, Decided):
        raise MVAUElaborationError(answer.findings)
    return answer.value


def bind_legacy_decomposed(resolved: MVAUResolvedDesign) -> DesignRealization:
    """Bind the retired dot-product semantic member for comparison tests."""

    selected = resolved.engine.query_property(
        resolved.point, MVAU_COMPUTE_SELECTION.paths.selected_kernel
    )
    if not isinstance(selected, Decided):
        raise MVAUElaborationError(selected.findings)
    if getattr(selected.value, "kernel_id", None) != MVAUComputeKernelId.DOT_PRODUCT.value:
        raise _fail(
            "mvau-legacy-decomposed-kernel-unsupported",
            "compatibility binding requires the old dot-product semantic member",
        )
    if not isinstance(resolved.result, NetworkRef):
        raise _fail(
            "mvau-legacy-decomposed-result-not-a-network",
            "compatibility binding requires the old replay-plus-compute Network",
        )
    network = resolved.result.network
    configured = {
        "compute": _bind(
            resolved,
            DECOMPOSED_MVAU_KERNELS.dot_product_hardware,
            "compute",
            DOT_PRODUCT_NODE,
            network,
        ),
        "replay": _bind(
            resolved,
            DECOMPOSED_MVAU_KERNELS.replay_hardware,
            "replay",
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


__all__ = ["bind_legacy_decomposed"]
