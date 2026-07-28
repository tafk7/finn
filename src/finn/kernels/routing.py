############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Taxonomy routing for ``finn.kernels`` nodes (Seam C).

FINN's flow routes on ``is_hls_node``/``is_rtl_node`` (``finn.util.fpgadataflow``), which
classify a node as an HLS or RTL hardware node. For a kernel node BOTH facts those
predicates need — is it HW-ready, and in which language — are pure functions of the ONE
genuine stored value, the ``implementation`` nodeattr (resolve's committed backend
selection):

- HW-ready  = ``implementation`` is set (resolve committed a backend).
- language  = the selected :class:`~finn.kernels.space.backend.Backend`'s ``language`` field.

Nothing extra is stored (no ``codegen_ready``/``language`` nodeattr that could go stale on
re-resolve); routing is a QUERY on committed state. :func:`kernel_hw_language` is the
derivation, and it is BARE-NODE by construction — a single nodeattr read plus a static
pool lookup, NO model, NO ``getCustomOp``, NO op instantiation — because the predicates it
feeds are called hot across the whole flow.
"""

from __future__ import annotations

from qonnx.util.basic import get_by_name

KERNEL_DOMAIN = "finn.kernels"

# {op_type: {backend.name: language}} — static per op class, built once on first use.
_LANGUAGE_TABLE: dict[str, dict[str, str | None]] | None = None


def _language_table() -> dict[str, dict[str, str | None]]:
    """The op_type → {implementation-name → language} map, built once from the kernel op
    registry. Reads each op class's pool via the ``kernel()`` classmethod — bare-node, no op
    instantiation. Cached module-side; the registry + pools are static per process."""
    global _LANGUAGE_TABLE
    if _LANGUAGE_TABLE is None:
        from finn.kernels import custom_op

        _LANGUAGE_TABLE = {
            op_type: {b.name: b.language for b in cls.kernel().pool}
            for op_type, cls in custom_op.items()
        }
    return _LANGUAGE_TABLE


def kernel_hw_language(node) -> str | None:
    """The resolved backend's language (``"hls"``/``"rtl"``) for a ``finn.kernels`` node, or
    ``None`` when the node is not a kernel node, is unresolved (``implementation`` empty or
    absent — i.e. NOT yet HW-ready), or names an unknown implementation.

    Bare-node: reads only the ``implementation`` nodeattr + a static pool lookup. No model,
    no ``getCustomOp``, no op instantiation — safe to call from the hot routing predicates.
    """
    if node is None or node.domain != KERNEL_DOMAIN:
        return None  # domain alone uniquely identifies kernel nodes (Seam A co-stamps it)
    impl_attr = get_by_name(node.attribute, "implementation")
    impl = impl_attr.s.decode("UTF-8") if impl_attr is not None else ""
    if not impl:
        return None  # unresolved -> not HW-ready
    return _language_table().get(node.op_type, {}).get(impl)
