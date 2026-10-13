# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The errors the graph readers and the KernelOps raise."""

from __future__ import annotations

from collections.abc import Iterable

from finn.core.space import Finding


class KernelOpError(ValueError):
    """A node or channel the KernelOps cannot bind or configure: a wrong fact or a
    refused choice. ``keys`` names what is refused, if anything: a node's choice
    attributes (``compute.packed.pe``), a channel's keys (``transport``, or
    ``<tensor>.<key>``), or tensors whose channel choices are refused whole.
    ``findings`` holds the coded reasons, which tests and the build's reports read."""

    def __init__(
        self, message: str, keys: tuple[str, ...] = (), findings: Iterable[Finding] = ()
    ) -> None:
        super().__init__(message)
        self.keys = keys
        self.findings = tuple(findings)


class FactUnstated(KernelOpError):
    """The graph lacks ``tensor``'s shape or datatype annotation, which binding needs.
    Conversion leaves the node on the host (``fact-unstated``) instead of stopping the
    build."""

    def __init__(self, message: str, tensor: str) -> None:
        super().__init__(message)
        self.tensor = tensor


__all__ = [
    "FactUnstated",
    "KernelOpError",
]
