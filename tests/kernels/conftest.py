# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``--strict-rtl``: a conformance sample the RTL checker declines fails."""

from __future__ import annotations

import pytest

from kernels import conformance


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--strict-rtl",
        action="store_true",
        help="fail a conformance sample whose module the RTL checker declines",
    )


def pytest_configure(config: pytest.Config) -> None:
    conformance.STRICT_RTL = bool(config.getoption("--strict-rtl", default=False))
