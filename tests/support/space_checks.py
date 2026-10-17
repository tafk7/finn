# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every Space snapshot's cache, in every tree, refuses what the engine's rule forbids
(``support.checked_cache``); production keeps exact dicts. Importing this module installs
the checked cache as ``_runtime.cache_factory``. It also holds the ``index_path`` fixture
and the terminal summary of the evaluations the cache kept and dropped that differ in their
reads. A conftest imports the fixture and the hook; every conftest that a gate's
``--confcutdir`` stops at does so itself (a tree that never runs the engine needs neither)."""

from __future__ import annotations

import sys
from typing import Protocol

import pytest

from finn.core.space import _runtime
from support.checked_cache import DIFFERING_READS, CheckedCache

_runtime.cache_factory = CheckedCache


class _Reporter(Protocol):
    """What the summary writes with: pytest's terminal reporter."""

    def section(self, title: str) -> None: ...

    def write_line(self, line: str) -> None: ...


@pytest.fixture(params=["index", "scan"])
def index_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run a test with the Space's readers index used for every cache, then for none: the
    answers are the same either way (``_runtime.INDEX_FROM``)."""
    monkeypatch.setattr(_runtime, "INDEX_FROM", 0 if request.param == "index" else sys.maxsize)
    return str(request.param)


def pytest_terminal_summary(terminalreporter: _Reporter) -> None:
    if DIFFERING_READS:
        terminalreporter.section(
            "cache setdefault: kept and dropped evaluations differing in reads"
        )
        for (site, kind), count in sorted(DIFFERING_READS.items()):
            terminalreporter.write_line(f"{count:6d}  {site}: {kind}")
        DIFFERING_READS.clear()
