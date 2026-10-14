# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each test is one build: it runs in one member cache (``finn.kernels.members``),
dropped at its end. A conftest imports the fixture and the hook; every conftest that a
gate's ``--confcutdir`` stops at does so itself."""

from __future__ import annotations

import pytest

import sys
from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _member_cache() -> Iterator[None]:
    from finn.kernels.members import member_cache  # noqa: PLC0415 - imported once used

    with member_cache():
        yield


def pytest_sessionfinish(session: pytest.Session) -> None:
    """In the member cache's verifying mode, write this process's counts
    (``finn.kernels.members.write_counts``)."""
    members = sys.modules.get("finn.kernels.members")
    if members is not None:
        members.write_counts()
