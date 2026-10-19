# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Space engine's purity check off for one test (``purity_unchecked``): for a test that
counts the calls the engine's memo saves, which the check makes again. The check is on in
``tests/core/space`` and wherever ``FINN_SPACE_CHECK_PURITY=1`` is set. A conftest imports
the fixture; every conftest that a gate's ``--confcutdir`` stops at does so itself."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def purity_unchecked() -> Iterator[None]:
    """The purity check off for this test, and back as it was after it."""
    from finn.core.space._world import check_purity  # noqa: PLC0415 - imported once used

    previous = check_purity(False)
    yield
    check_purity(previous)
