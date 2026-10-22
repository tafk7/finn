# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every Space operation must restore the caller's evaluation context (no authored call
left active); every cache hit, and every node the depth guard abandoned, is evaluated again
and compared (the purity check), but in a test that counts the calls the cache saves
(``purity_unchecked``, from ``support.purity``).

``--space-implementation MODULE:ATTRIBUTE`` (or a registered name; ``public``, the public
API, by default, loaded by ``support.space_oracle.load``) names the implementation the
reference harness asks wherever a test names none (``reference_harness.implementation``),
and the report's header names it:

    python -m pytest -p no:xdist -q --space-implementation=core.space._running_type:RUNNING \\
        core/space/test_reference_cases.py ...

(from ``tests``). Before the paths, the option and its value are one argument (``=``): pytest
reads this file's options only once it has found the paths, which a separate value would
be taken for."""

from collections.abc import Iterator

import pytest
from support.purity import (
    check_purity,
    purity_unchecked,  # noqa: F401
    restore_purity,
)

from core.space.reference_harness.implementation import OPTION, Option
from finn.core.space._failure import current


@pytest.fixture(autouse=True)
def evaluation_context_is_reset() -> Iterator[None]:
    assert current() is None
    yield
    assert current() is None


@pytest.fixture(autouse=True)
def purity_checked() -> Iterator[None]:
    """Every cache hit, and every node the depth guard abandoned, is evaluated again and the
    answers compared (``support.purity.check_purity``)."""
    previous = check_purity(True)
    yield
    restore_purity(previous)


# -- the reference harness's implementation ----------------------------------------------------

_OPTION = pytest.StashKey[Option]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        OPTION,
        default="public",
        metavar="MODULE:ATTRIBUTE",
        help="the implementation the reference harness asks wherever a test names none: a "
        "registered name, or module:attribute (support.space_oracle.load)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_OPTION] = Option.start(config.getoption("space_implementation", "public"))


def pytest_unconfigure(config: pytest.Config) -> None:
    option = config.stash.get(_OPTION, None)
    if option is not None:
        option.stop()


def pytest_report_header(config: pytest.Config) -> str:
    return config.stash[_OPTION].header()
