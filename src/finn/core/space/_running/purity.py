# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The purity check's switch, for every model in the process (a test mode).

The node cache, the domains' outcomes kept on a facts layer and the depth guard's restart
all rest on purity: an authored function answers from what it reads alone. With the check
on, a cache hit is evaluated anyway, a kept domain outcome computed again, and a node the
depth guard abandoned evaluated again once it completes; where the two answers differ (a
value by type and ``==``), the function is impure, its ``EvaluationError``.

The switch is this module's attribute ``_PURITY``, read at each ask
(``purity._PURITY``, never imported by name, which would freeze it): on from the start
where the environment variable ``FINN_SPACE_CHECK_PURITY`` is 1, and switched by
``check_purity``.
"""

from __future__ import annotations

import os

_PURITY = os.environ.get("FINN_SPACE_CHECK_PURITY") == "1"


def check_purity(on: bool) -> bool:
    """Switch the purity check on or off, for every model; the setting it replaces."""
    global _PURITY  # noqa: PLW0603 - a test mode, switched for the process
    previous, _PURITY = _PURITY, on
    return previous


__all__ = ["check_purity"]
