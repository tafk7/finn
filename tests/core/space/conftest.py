# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every Space operation must restore the caller's evaluation context."""

from collections.abc import Iterator

import pytest

from finn.core.space import _execution


@pytest.fixture(autouse=True)
def evaluation_context_is_reset() -> Iterator[None]:
    assert _execution.current() is None
    yield
    assert _execution.current() is None
