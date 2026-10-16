# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Assert a refusal by its code, not its words.

``refused`` expects a ``Refusal`` (``finn.core.reporting.findings``) carrying a finding
with a given code, and optionally its owner, its message and its details. A test that
names the code fails when the code under test raises another one, and survives a
rewording of the message: each message's text is pinned once, in the test of the module
that writes it (``message=``), and every other site asserts the code only.

A test helper, imported as ``support.refusals`` (the gates put ``tests`` on
``PYTHONPATH``); it depends on ``finn.core.reporting`` and pytest only, so a test of any
tree may import it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from finn.core.reporting.findings import Finding, FindingKind, Refusal


@dataclass
class Caught:
    """What ``refused`` caught, set when the block has raised: ``error`` is the refusal,
    ``finding`` the one of its top-level findings that matched."""

    error: Refusal = field(init=False)
    finding: Finding = field(init=False)


def _stated(details: dict[str, object]) -> dict[str, object]:
    """``details`` as a finding holds them (frozen: a list as a tuple)."""
    return dict(Finding(FindingKind.REJECTION, "stated", "", "", tuple(details.items())).details)


def _describe(finding: Finding) -> str:
    stated = f" {dict(finding.details)}" if finding.details else ""
    return f"{finding.owner or '(no owner)'}: {finding.code}: {finding.message}{stated}"


@contextmanager
def refused(
    code: str,
    *,
    owner: str | None = None,
    message: str | None = None,
    error: type[Refusal] = Refusal,
    **details: object,
) -> Iterator[Caught]:
    """Expect ``error`` carrying a finding with ``code`` (among its top-level findings),
    owned by ``owner`` if given, its message matching ``message`` (a regex, ``re.search``) if
    given, its details holding ``details``. ``Caught.finding`` is the finding matched,
    ``Caught.error`` the error.

    Fails where nothing is raised, and where no finding matches, naming the type of the
    error and each finding's owner, code, message and details. An error that is not
    an ``error`` propagates."""
    caught = Caught()
    wanted = _stated(details)
    try:
        yield caught
    except error as raised:
        for finding in raised.findings:
            if (
                finding.code == code
                and (owner is None or finding.owner == owner)
                and (message is None or re.search(message, finding.message))
                and wanted.items() <= dict(finding.details).items()
            ):
                caught.error = raised
                caught.finding = finding
                return
        asked = [f"code {code!r}"]
        asked += [] if owner is None else [f"owner {owner!r}"]
        asked += [] if message is None else [f"message matching {message!r}"]
        asked += [] if not wanted else [f"details {wanted}"]
        found = "\n".join(f"  {_describe(finding)}" for finding in raised.findings)
        codes = [finding.code for finding in raised.findings]
        pytest.fail(
            f"{type(raised).__name__} carries no finding with {', '.join(asked)}; "
            f"it caught the codes {codes}:\n{found}",
            pytrace=False,
        )
    else:
        pytest.fail(f"DID NOT RAISE {error.__name__} with code {code!r}", pytrace=False)
