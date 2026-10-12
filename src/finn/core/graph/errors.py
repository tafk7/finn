# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The errors of FINN's view of a graph and of the KernelOps that read it.

``KernelOpError`` is a node or a channel that cannot be bound or configured: a fact
refused, or a choice refused. ``FactUnstated`` is a fact the graph does not state, which
conversion reports and does not stop on.
"""

from __future__ import annotations

from collections.abc import Iterable

from finn.core.space import Finding


class KernelOpError(ValueError):
    """A node or a channel the KernelOps cannot bind or configure: a missing or refused
    fact, or a refused choice. ``keys`` names what is refused, as its owner states it,
    when there is something to name: a node's choice attributes (``compute.packed.pe``),
    a channel's keys (``transport``, or ``<tensor>.<key>`` where a node's point or
    packaging names the owner), or the tensors whose channel choices are refused
    whole. ``findings`` are the refusal's reasons where they have codes (the kernels'
    findings, the shell's, a conversion's): what a reader of the build asserts on."""

    def __init__(
        self, message: str, keys: tuple[str, ...] = (), findings: Iterable[Finding] = ()
    ) -> None:
        super().__init__(message)
        self.keys = keys
        self.findings = tuple(findings)


class FactUnstated(KernelOpError):
    """A fact binding reads that the graph does not state (``tensor``'s shape or its
    datatype annotation): not a contradiction, so conversion leaves the node on the
    host with a ``fact-unstated`` finding instead of stopping the build."""

    def __init__(self, message: str, tensor: str) -> None:
        super().__init__(message)
        self.tensor = tensor


__all__ = [
    "FactUnstated",
    "KernelOpError",
]
