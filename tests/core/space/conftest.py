# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every Space operation must restore the caller's evaluation context (no authored call
left active); every memo hit runs its function again and compares (the purity check),
but in a test that counts the calls the memo saves (``purity_unchecked``, from
``support.purity``).

``--space-implementation MODULE:ATTRIBUTE`` (or a registered name; ``today`` by default,
loaded by ``support.space_oracle.load``) names the implementation the reference harness
asks wherever a test names none (``reference_harness.implementation``). Named otherwise
than today's engine, the session fails if today's is opened or made, and its summary names
the implementation, each test deselected and today's opens:

    python -m pytest -p no:xdist -q --space-implementation=core.space._running_type:RUNNING \\
        core/space/test_reference_cases.py ...

(from ``tests``). Before the paths, the option and its value are one argument (``=``): pytest
reads this file's options only once it has found the paths, which a separate value would
be taken for."""

from collections.abc import Generator, Iterator, Sequence
from typing import Any

import pytest
from support.purity import purity_unchecked  # noqa: F401

from core.space.reference_harness.implementation import OPTION, Option
from finn.core.space._failure import current
from finn.core.space._world import check_purity


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


# -- the reference harness's implementation ----------------------------------------------------

_OPTION = pytest.StashKey[Option]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        OPTION,
        default="today",
        metavar="MODULE:ATTRIBUTE",
        help="the implementation the reference harness asks wherever a test names none: a "
        "registered name, or module:attribute (support.space_oracle.load); named otherwise "
        "than today's, the session fails if today's engine is opened or made",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_OPTION] = Option.start(config.getoption("space_implementation", "today"))


def pytest_unconfigure(config: pytest.Config) -> None:
    option = config.stash.get(_OPTION, None)
    if option is not None:
        option.stop()


def pytest_report_header(config: pytest.Config) -> str | None:
    option = config.stash[_OPTION]
    return option.header() if option.opens is not None else None


def pytest_deselected(items: Sequence[pytest.Item]) -> None:
    if items:
        items[0].config.stash[_OPTION].deselected.extend(item.nodeid for item in items)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, None, None]:
    with item.config.stash[_OPTION].running(item.nodeid):
        yield


def pytest_sessionfinish(session: pytest.Session) -> None:
    """The session fails where today's engine was opened or made while another
    implementation is named."""
    if session.config.stash[_OPTION].failed():
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    for line in config.stash[_OPTION].summary():
        terminalreporter.write_line(line)
