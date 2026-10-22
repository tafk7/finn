# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Space engine's purity check off for one test (``purity_unchecked``): for a test that
counts the calls the engine's cache hits save, which the check makes again. The check is on
in ``tests/core/space`` and wherever ``FINN_SPACE_CHECK_PURITY=1`` is set. A conftest imports
the fixture; every conftest that a gate's ``--confcutdir`` stops at does so itself.

``check_purity`` is the engine's switch, ``_running.purity.check_purity``, read by every
evaluation at each ask."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


def check_purity(on: bool) -> bool:
    """Switch the purity check on or off; the setting it replaces."""
    from finn.core.space._running import purity  # noqa: PLC0415 - imported once used

    return purity.check_purity(on)


def restore_purity(previous: bool) -> None:
    """The switch back as ``check_purity`` found it."""
    check_purity(previous)


@pytest.fixture
def purity_unchecked() -> Iterator[None]:
    """The purity check off for this test, and back as it was after it."""
    previous = check_purity(False)
    yield
    restore_purity(previous)
