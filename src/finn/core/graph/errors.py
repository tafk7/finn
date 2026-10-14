# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The errors the graph readers and the KernelOps raise."""

from __future__ import annotations

from collections.abc import Iterable

from finn.core.reporting.findings import Finding, FindingKind, Refusal


class KernelOpError(Refusal):
    """A node or channel the KernelOps cannot bind or configure: a wrong fact or a
    refused choice. ``findings`` holds the coded reasons, which tests and the build's
    reports read, each owned by who refused (a KernelOp by its op type, a transformation,
    reader or check by its name) and naming the node or tensor it is about in its
    details; the error's message is its own. ``keys`` names what is refused, if
    anything: a node's choice attributes (``compute.packed.pe``), a channel's keys
    (``transport``, or ``<tensor>.<key>``), or tensors whose channel choices are
    refused whole. ``refused`` makes the error of one rejection."""

    def __init__(
        self, message: str, keys: tuple[str, ...] = (), *, findings: Iterable[Finding]
    ) -> None:
        super().__init__(findings, message)
        self.keys = keys

    @staticmethod
    def refused(
        code: str,
        owner: str,
        message: str,
        keys: tuple[str, ...] = (),
        *,
        node: str | None = None,
        tensor: str | None = None,
    ) -> KernelOpError:
        """The error of one rejection, ``code`` by ``owner``, its message the error's,
        with ``keys`` and the ``node`` or ``tensor`` it is about in its details."""
        subjects = (("node", node), ("tensor", tensor))
        details = ((("keys", keys),) if keys else ()) + tuple(
            (name, value) for name, value in subjects if value is not None
        )
        found = Finding(FindingKind.REJECTION, code, owner, message, details)
        return KernelOpError(message, keys, findings=(found,))


class FactUnstated(KernelOpError):
    """The graph lacks ``tensor``'s shape or datatype annotation, which binding needs
    (``fact-unstated``, by ``owner``, the reader that needs it). Conversion leaves the
    node on the host instead of stopping the build."""

    def __init__(self, owner: str, message: str, tensor: str, node: str) -> None:
        details = (("node", node), ("tensor", tensor))
        unstated = Finding(FindingKind.REJECTION, "fact-unstated", owner, message, details)
        super().__init__(message, findings=(unstated,))
        self.tensor = tensor


__all__ = [
    "FactUnstated",
    "KernelOpError",
]
