# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Read a refused or unresolved answer's findings in the space tests."""

from __future__ import annotations

from finn.core.space import Rejected, Unresolved


def codes(result: object) -> set[str]:
    """The codes of the findings of ``result``, which must be refused or unresolved."""
    assert isinstance(result, (Rejected, Unresolved)), result
    return {finding.code for finding in result.findings}
