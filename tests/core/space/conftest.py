# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every Space operation must restore the caller's evaluation context (no authored call
left active); every memo hit runs its function again and compares (the purity check),
but in a test that counts the calls the memo saves; and the reference model's
comparisons against the engine sit out until the model rewrite's adapter lands."""

from collections.abc import Iterator

import pytest

from finn.core.space._failure import current
from finn.core.space._world import check_purity

# The reference harness compares the engine through an adapter that still reads the removed
# forcing and unchecked answers; the model's rewrite brings the adapter to the new engine.
collect_ignore = [
    "test_reference_cases.py",
    "test_reference_rules.py",
    "test_reference_scopes.py",
    "test_reference_harness.py",
    "test_reference_walks.py",
]  # until the model rewrite's adapter lands on the joint tree


@pytest.fixture(autouse=True)
def evaluation_context_is_reset() -> Iterator[None]:
    assert current() is None
    yield
    assert current() is None


@pytest.fixture(autouse=True)
def purity_checked() -> Iterator[None]:
    """Every memo hit, and every method the depth guard abandoned, runs its function again
    and compares the answers (``_world.check_purity``)."""
    previous = check_purity(True)
    yield
    check_purity(previous)


@pytest.fixture
def purity_unchecked(purity_checked: None) -> Iterator[None]:
    """For a test that counts the calls the memo saves, which the purity check makes."""
    check_purity(False)
    yield
    check_purity(True)
