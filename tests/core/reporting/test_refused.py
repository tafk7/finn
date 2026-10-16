# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``refused`` (``tests/support/refusals.py``): a test asserts a refusal by its code, and the
helper fails, naming what it found, when the code, the owner, the message or the details
differ."""

from __future__ import annotations

import pytest
from support.refusals import refused

from finn.core.reporting.findings import Finding, FindingKind, Refusal


class Tooling(Refusal):
    """A refusal of its own, to tell the types apart."""


def finding(
    code: str, owner: str = "tool.owner", message: str = "it is refused", **details: object
) -> Finding:
    return Finding(FindingKind.REJECTION, code, owner, message, tuple(details.items()))


def raising(*findings: Finding, error: type[Refusal] = Refusal) -> None:
    raise error(findings)


def test_a_refusal_with_the_code_is_caught_and_its_finding_given() -> None:
    first, second = finding("first-code"), finding("second-code", "other.owner")
    with refused("second-code") as caught:
        raising(first, second)
    assert caught.finding is second
    assert caught.error.findings == (first, second)


def test_a_wrong_code_fails_naming_the_codes_found() -> None:
    with pytest.raises(pytest.fail.Exception) as failed:
        with refused("wanted-code"):
            raising(finding("first-code", message="the first words"), finding("second-code"))
    text = str(failed.value)
    assert "carries no finding with code 'wanted-code'" in text
    assert "['first-code', 'second-code']" in text
    assert "tool.owner: first-code: the first words" in text


def test_the_owner_narrows_the_match() -> None:
    with refused("a-code", owner="this.owner") as caught:
        raising(finding("a-code", "that.owner"), finding("a-code", "this.owner"))
    assert caught.finding.owner == "this.owner"
    with pytest.raises(pytest.fail.Exception, match="owner 'no.owner'"):
        with refused("a-code", owner="no.owner"):
            raising(finding("a-code", "this.owner"))


def test_the_message_is_a_regex_searched_in_the_finding_s_message() -> None:
    with refused("a-code", message=r"is \w+ on") as caught:
        raising(finding("a-code", message="the tool is not on the route"))
    assert caught.finding.message.startswith("the tool")
    with pytest.raises(pytest.fail.Exception, match="message matching 'on the PATH'"):
        with refused("a-code", message="on the PATH"):
            raising(finding("a-code", message="not on the route"))


def test_the_details_must_hold_and_a_list_is_read_as_a_tuple() -> None:
    with refused("a-code", tool="vivado", pins=["a", "b"]):
        raising(finding("a-code", tool="vivado", pins=("a", "b"), more="x"))
    with pytest.raises(pytest.fail.Exception, match="details"):
        with refused("a-code", tool="hls"):
            raising(finding("a-code", tool="vivado"))
    with pytest.raises(pytest.fail.Exception, match="details"):
        with refused("a-code", absent="x"):
            raising(finding("a-code", tool="vivado"))


def test_every_narrowing_must_hold_on_one_finding() -> None:
    """The owner of one finding and the message of another do not make a match."""
    with pytest.raises(pytest.fail.Exception, match="carries no finding"):
        with refused("a-code", owner="this.owner", message="second"):
            raising(
                finding("a-code", "this.owner", "first words"),
                finding("a-code", "that.owner", "second words"),
            )


def test_the_error_type_is_the_one_asked_for() -> None:
    with refused("a-code", error=Tooling) as caught:
        raising(finding("a-code"), error=Tooling)
    assert isinstance(caught.error, Tooling)
    with pytest.raises(Refusal):
        with refused("a-code", error=Tooling):
            raising(finding("a-code"))


def test_an_error_that_is_not_a_refusal_is_not_caught() -> None:
    with pytest.raises(ValueError, match="plain"):
        with refused("a-code"):
            raise ValueError("plain")


def test_no_error_fails() -> None:
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE Refusal with code 'a-code'"):
        with refused("a-code"):
            pass
