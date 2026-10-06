# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.util._legacy_build_env import toolchain as legacy_toolchain


@pytest.fixture(scope="session")
def hls_toolchain():
    """The toolchain this environment selects, as HLS C++ simulation takes it by
    default; the test is skipped when it names no HLS installation."""
    toolchain = legacy_toolchain()
    try:
        toolchain.hls_installation()
    except LookupError as exc:
        pytest.skip(str(exc))
    return toolchain
