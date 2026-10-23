# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The depth guard: how deep evaluation recurses on the Python stack, and the exception
that unwinds it past that depth.

Evaluation recurses on the Python stack to ``DEPTH`` evaluations in flight. Past it,
``Deeper`` unwinds to the run (``evaluate.Session.run``), which evaluates the deep node
first, then asks again what it abandoned: authored functions are pure, so asking again
answers the same. A method's reader lets it pass through untouched
(``receiver.MethodReader.read_value``): it abandons the call, it is not the call's failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._scopes import Address

# How many evaluations deep a run goes on the Python stack before the depth guard
# unwinds it. Each level through a method's read takes about a dozen Python frames.
DEPTH = 32


class Deeper(BaseException):
    """The depth guard: ``node`` was demanded deeper than ``DEPTH``. Every evaluation and
    call it passes through is abandoned (nothing kept); the run evaluates the node first,
    then again what it abandoned. An evaluator's own guard says where (``evaluate``'s)."""

    def __init__(self, node: Address) -> None:
        super().__init__(node)
        self.node = node


__all__ = ["DEPTH", "Deeper"]
