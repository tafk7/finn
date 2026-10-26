# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build made of a partition of KernelOps: the ``finn.outputs`` namespace.

On the partition's body: the digest of the design its check verified, the search's
record, the packaged IP, its key and the options it was packaged with. On the parent
graph: the bitfile, its hardware handoff, the digest of the design and the key of the IP it
was built from, and its shell options. Two ties, two facts: a design digest
(``finn.core.graph.partition.design_digest``) says which design an artifact or a check
describes, an IP key (``finn.toolchain.ip.packaging_key``) which packaged IP it holds. A
reader compares a recorded value with the current one, so an artifact is never cleared
when the design changes.

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


def _keys_by_owner(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(keys, list) and all(isinstance(key, str) for key in keys)
        for keys in value.values()
    )


def _search(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"design", "held", "objective_fps"}:
        return False
    objective = value["objective_fps"]
    return (
        _digest(value["design"])
        and _keys_by_owner(value["held"])
        and (
            objective is None
            or (isinstance(objective, int | float) and not isinstance(objective, bool))
            and objective > 0
        )
    )


OUTPUT_VERIFIED = OUTPUTS.key(
    "verified", str, check=_digest, expect="a design digest (sha256:<64 hex digits>)"
)
"""The digest of the design the partition's check verified (``design_digest`` of the body
when ``finn.custom_op.kernels.shell.verify_partition`` passed it). Emission refuses a body
whose design is not that one (``refuse_unverified``). Not the IP key: what Vivado was
given is one fact, that this design passed its check another."""
OUTPUT_SEARCH = OUTPUTS.key(
    "search",
    JSON,
    check=_search,
    expect=(
        "the search's record: {design: a design digest, held: {graph name: [key, ...]}, "
        "objective_fps: frames a second or null}"
    ),
)
"""What the search that made the body's choices left to later readers, written with them
(``finn.transformation.kernels.explore_kernel_choices``): ``design``, the digest of the
design the search persisted (``design_digest``), so a reader takes the record as its own
only while the body is that design; ``held``, the keys the search held, by owner (``{graph
name: [key, ...]}``, each owner's keys sorted; the values are the ones the nodes and tensors
state), which a verification holds again; ``objective_fps``, the frames a second a target
throughput explored the body to, or null where none did or an entry changed a kernel's
choice after it, which a shell's build reports its delivered clock against. A record whose
``design`` is not the body's is stale: it holds nothing and states no objective
(``finn.core.graph.partition.search_record``)."""
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
OUTPUT_IP_KEY = OUTPUTS.key(
    "ip_key", str, check=_digest, expect="an IP key (sha256:<64 hex digits>)"
)
"""The packaged IP's key: what Vivado was given to package it, Vivado and ``run_synth``
(``finn.toolchain.ip.packaging_key``)."""
OUTPUT_IP_OPTIONS = OUTPUTS.key(
    "ip_options", JSON, check=_options, expect="the packaging options by name"
)
"""The options the IP was packaged with (``run_synth``), so a reader can ask for the key
the body has now."""
OUTPUT_BITFILE = OUTPUTS.key("bitfile", str, check=_named, expect="a file")
"""The bitfile the shell's integration made."""
OUTPUT_HWH = OUTPUTS.key("hwh", str, check=_named, expect="a file")
"""The bitfile's hardware handoff."""
OUTPUT_BITFILE_MADE_FROM = OUTPUTS.key(
    "bitfile_made_from", str, check=_digest, expect="a design digest (sha256:<64 hex digits>)"
)
"""The digest of the design the bitfile was built from (``design_digest`` of the
partition's body when the shell's build ran)."""
OUTPUT_BITFILE_IP_KEY = OUTPUTS.key(
    "bitfile_ip_key", str, check=_digest, expect="an IP key (sha256:<64 hex digits>)"
)
"""The key of the partition's IP the bitfile was built from (the body's ``ip_key`` when
the shell's build ran)."""
OUTPUT_BITFILE_OPTIONS = OUTPUTS.key(
    "bitfile_options", JSON, check=_options, expect="the shell's build options by name"
)
"""The shell's build options the bitfile was built with (``hw_debug``)."""


__all__ = [
    "OUTPUTS",
    "OUTPUT_BITFILE",
    "OUTPUT_BITFILE_IP_KEY",
    "OUTPUT_BITFILE_MADE_FROM",
    "OUTPUT_BITFILE_OPTIONS",
    "OUTPUT_HWH",
    "OUTPUT_INTERFACES",
    "OUTPUT_IP",
    "OUTPUT_IP_KEY",
    "OUTPUT_IP_OPTIONS",
    "OUTPUT_SEARCH",
    "OUTPUT_VERIFIED",
    "OUTPUT_VLNV",
]
