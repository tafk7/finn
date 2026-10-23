# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Reading what the Vivado tools print.

Vivado, Vitis HLS, xvlog, xelab and xsim print their messages in one form:
``SEVERITY: [Tool N-M] text``, for example ``ERROR: [Synth 8-439] module 'foo' not found``.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Tuple, Union

# Severity, optional message ID (some messages have none) and text.
_MESSAGE = re.compile(
    r"^\s*(FATAL_ERROR|FATAL|ERROR|CRITICAL WARNING|WARNING|INFO)\s*:\s*(?:\[([^\]]+)\]\s*)?(.*)$"
)

ERROR_SEVERITIES = ("FATAL_ERROR", "FATAL", "ERROR")

# Vivado's [Common 17-345] and its wording, and the wording of FlexNet (FlexLM).
_NO_LICENSE = re.compile(
    r"\[Common 17-345\]|valid licen[cs]e was not found|licen[cs]e checkout failed|"
    r"failed to get a licen[cs]e|no licen[cs]e (is )?available|FLEX(net|lm) licensing error",
    re.IGNORECASE,
)

LICENSE_UNAVAILABLE = "license-unavailable"
TOOL_ERROR = "tool-error"


@dataclass(frozen=True)
class Messages:
    """The messages of one tool run, as read by ``messages``.

    Attributes:
        errors: the error lines (ERROR, FATAL, FATAL_ERROR), in the order printed.
        critical_warnings: the critical warnings, in the order printed.
        license_unavailable: whether the output says that no license was found. This
            is a hint from the wording of the output, not a diagnosis.
    """

    errors: Tuple[str, ...]
    critical_warnings: Tuple[str, ...]
    license_unavailable: bool

    @property
    def code(self) -> str:
        """``LICENSE_UNAVAILABLE`` if no license was found, else ``TOOL_ERROR``."""
        return LICENSE_UNAVAILABLE if self.license_unavailable else TOOL_ERROR


def messages(lines: Union[str, Iterable[str]]) -> Messages:
    """Read the messages in the output of a Vivado tool: a transcript's text, or its lines."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    errors = []
    critical = []
    license_unavailable = False
    for line in lines:
        line = line.strip()
        license_unavailable = license_unavailable or _NO_LICENSE.search(line) is not None
        found = _MESSAGE.match(line)
        if found is None:
            continue
        if found.group(1) in ERROR_SEVERITIES:
            errors.append(line)
        elif found.group(1) == "CRITICAL WARNING":
            critical.append(line)
    return Messages(tuple(errors), tuple(critical), license_unavailable)
