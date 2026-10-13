# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Read a refused or unresolved answer's findings, and a node's dependencies, in the space
tests."""

from __future__ import annotations

from typing import Any

from finn.core.space import Model, Rejected, Unresolved
from finn.core.space.ir import Node


def codes(result: object) -> set[str]:
    """The codes of the findings of ``result``, which must be refused or unresolved."""
    assert isinstance(result, (Rejected, Unresolved)), result
    return {finding.code for finding in result.findings}


def dependency_nodes(model: Model[Any], reference: object) -> list[Node]:
    """The linked nodes ``reference`` (at the model's root) names as its known
    structural and explicit dependencies, as compiled."""
    linked = model.linked
    return [linked.nodes[index] for index in linked.nodes[model.resolve(0, reference)].dependencies]
