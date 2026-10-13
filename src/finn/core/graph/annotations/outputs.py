# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build made of a partition of KernelOps: the ``finn.outputs`` namespace.

On the partition's body: the packaged IP and the digest of the design it was made from.
On the parent graph: the bitfile, its hardware handoff, and the digest and shell options
it was built with. A reader compares a recorded digest with the body's current one
(``finn.core.graph.partition.design_digest``), so an artifact is never cleared when the
design changes.

It imports only qonnx, so the builder and the partition's readers use it without
loading the kernels.
"""

from __future__ import annotations

import re

from qonnx.core.metadata import JSON, Namespace

OUTPUTS = Namespace("finn.outputs", version=1)
"""What a build made of a partition of KernelOps (typed graph metadata): the packaged
IP on the partition's body, the shell's integration on the build's parent graph."""


def _named(item: object) -> bool:
    return isinstance(item, str) and bool(item)


def _interface_names(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(names, list) for names in value.values())


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _options(value: object) -> bool:
    return isinstance(value, dict)


OUTPUT_IP = OUTPUTS.key("ip", str, check=_named, expect="a directory")
"""The packaged IP's directory (its ``component.xml``)."""
OUTPUT_VLNV = OUTPUTS.key("vlnv", str, check=_named, expect="a VLNV")
"""The packaged IP's VLNV."""
OUTPUT_INTERFACES = OUTPUTS.key(
    "interfaces",
    JSON,
    check=_interface_names,
    expect="interface names by kind (clk, rst, s_axis, m_axis, axilite, ...)",
)
"""The packaged IP's interface names, by kind."""
OUTPUT_MADE_FROM = OUTPUTS.key(
    "made_from", str, check=_digest, expect="a design digest (sha256:<64 hex digits>)"
)
"""The digest of the design the packaged IP was made from (``design_digest`` of the
body it was packaged from)."""
OUTPUT_BITFILE = OUTPUTS.key("bitfile", str, check=_named, expect="a file")
"""The bitfile the shell's integration made."""
OUTPUT_HWH = OUTPUTS.key("hwh", str, check=_named, expect="a file")
"""The bitfile's hardware handoff."""
OUTPUT_BITFILE_MADE_FROM = OUTPUTS.key(
    "bitfile_made_from", str, check=_digest, expect="a design digest (sha256:<64 hex digits>)"
)
"""The digest of the design of the partition's IP the bitfile was built from (the body's
``made_from`` when the shell's build ran)."""
OUTPUT_BITFILE_OPTIONS = OUTPUTS.key(
    "bitfile_options", JSON, check=_options, expect="the shell's build options by name"
)
"""The shell's build options the bitfile was built with (``hw_debug``)."""


__all__ = [
    "OUTPUTS",
    "OUTPUT_BITFILE",
    "OUTPUT_BITFILE_MADE_FROM",
    "OUTPUT_BITFILE_OPTIONS",
    "OUTPUT_HWH",
    "OUTPUT_INTERFACES",
    "OUTPUT_IP",
    "OUTPUT_MADE_FROM",
    "OUTPUT_VLNV",
]
