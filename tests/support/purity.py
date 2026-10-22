# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Space engine's purity check off for one test (``purity_unchecked``): for a test that
counts the calls the engine's cache saves, which the check makes again. The check is on in
``tests/core/space`` and wherever ``FINN_SPACE_CHECK_PURITY=1`` is set. A conftest imports
the fixture; every conftest that a gate's ``--confcutdir`` stops at does so itself.

``check_purity`` switches both switches the engine has while the world engine is kept:
``_world.check_purity``, the one the running type reads today, and
``_running.purity.check_purity``, the one it reads once the world engine is deleted."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


def check_purity(on: bool) -> tuple[bool, bool]:
    """Switch the purity check on or off, both switches; the settings it replaces."""
    from finn.core.space import _world  # noqa: PLC0415 - imported once used
    from finn.core.space._running import purity  # noqa: PLC0415

    return _world.check_purity(on), purity.check_purity(on)


def restore_purity(previous: tuple[bool, bool]) -> None:
    """Both switches back as ``check_purity`` found them."""
    from finn.core.space import _world  # noqa: PLC0415 - imported once used
    from finn.core.space._running import purity  # noqa: PLC0415

    _world.check_purity(previous[0])
    purity.check_purity(previous[1])


@pytest.fixture
def purity_unchecked() -> Iterator[None]:
    """The purity check off for this test, and back as it was after it."""
    previous = check_purity(False)
    yield
    restore_purity(previous)
