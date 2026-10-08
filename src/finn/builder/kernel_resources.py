# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's resources per member of the shell (``report/resources.json``).

``shell_resources_report`` states, for each member of the partition's shell root (the
partition, each end by its boundary channel, each IP of the static region) and their
total, up to three columns:

- ``model``: what the shell root states (``finn.custom_op.kernels.shell.shell_resources``):
  the partition's kernels from their RTL's fits, each end and each static IP from its
  out-of-context characterisation;
- ``out_of_context``: Vivado's synthesis of each IP alone. On a shell with an
  integration, the integration's per-IP runs (``<run>_utilization_synth.rpt``, read by
  ``utilization_synth``), each run its instance's (``member_instances``); on ``ip``, the
  partition packaged with OOC_SYNTH, its hierarchical report's top
  (``package.hierarchical_utilization``);
- ``placed``: the routed design's hierarchical report (``placed_hierarchy``), each
  instance below the block design's top its member's. ``ip`` is not placed.

A column the build did not make is ``null`` for every member, and ``absent`` says why.
``caveat`` is the shell row's for the board (``finn.platform.ShellRow.caveat``): on a
board the shell was not built and timed on, the ends' and static region's ``model``
counts are carried over from where they were characterised; ``null`` otherwise.
A member a column does not list (the placed design lists no row for the processor and
its reset) is ``null`` there. ``unattributed`` is each column's total less its members'
sum: what the report lists outside the members, or what no member's row holds. The
placed column's can be negative: Vivado's hierarchical report can count its children
above its top row (TFC on Ultra96: the members' LUTs 3 above it), and the difference is
stated as it falls, not clipped.

When the partition was synthesized out of context (OOC_SYNTH), ``partition_members``
is its split by member of the shell root (``package.ooc_member_resources``):
synthesis flattens small instances into the module's top, and what no reported
instance holds is stated ``unattributed`` there, the instances with no row
``unreported_instances``; the hierarchy is not kept.

Every count is in ``finn.kernels.utilization.Resources``' units: LUTs, FFs, RAMB18
halves (a RAMB36 is two), URAMs and DSPs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels.base import read_target
from finn.custom_op.kernels.partition import member
from finn.custom_op.kernels.shell import ShellResources, shell_resources
from finn.kernels.explore import Completion
from finn.kernels.utilization import SHELL_CHARACTERISED, Resources
from finn.platform import shell_row
from finn.transformation.fpgadataflow.kernel_partitions import OUTPUT_REPORTS, partition_body
from finn.transformation.fpgadataflow.pynq_runner import STATIC_INSTANCES
from finn.transformation.kernels.integration import integration
from finn.transformation.kernels.package import (
    configured_root,
    hierarchical_utilization,
    ooc_member_resources,
)

#: The report's file, under the output directory.
RESOURCES_FILE = "report/resources.json"

#: A flat utilization report's rows by the resource they count; the first row of each
#: name counts (later sections repeat primitive names).
_SITE_TYPES = {
    "CLB LUTs": "lut",
    "Slice LUTs": "lut",
    "CLB Registers": "ff",
    "Slice Registers": "ff",
    "RAMB36/FIFO": "bram36",
    "RAMB18": "bram18",
    "URAM": "uram",
    "DSPs": "dsp",
}


def _resources(counts: Mapping[str, int]) -> Resources:
    found = dict(counts)
    bram18 = 2 * found.pop("bram36", 0) + found.pop("bram18", 0)
    return Resources(**found, bram18=bram18)


def utilization_synth(text: str) -> Resources:
    """The resources of Vivado's flat ``report_utilization`` of one synthesized IP (a
    ``<run>_utilization_synth.rpt``): its LUTs, registers, block RAM tiles, URAMs and
    DSPs, each the first row of its site type."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 2:
            continue
        site = cells[0].rstrip("*").strip()
        key = _SITE_TYPES.get(site) or ("dsp" if site.startswith("DSP48") else None)
        if key is not None and key not in counts and cells[1].isdigit():
            counts[key] = int(cells[1])
    if "lut" not in counts:
        raise ValueError("no LUT row in the utilization report")
    return _resources(counts)


#: A hierarchical report's columns by the resource they count.
_HIERARCHY_COLUMNS = {
    "Total LUTs": "lut",
    "FFs": "ff",
    "RAMB36": "bram36",
    "RAMB18": "bram18",
    "URAM": "uram",
}


def placed_hierarchy(text: str) -> list[tuple[int, str, Resources]]:
    """The rows of Vivado's ``report_utilization -hierarchical -format xml``, in order:
    each instance's depth (the top 0), its name and its resources."""
    table = ET.fromstring(text).find(".//table")
    if table is None:
        raise ValueError("no hierarchical utilization table in the report")
    header: list[str] = []
    rows: list[tuple[int, str, Resources]] = []
    for row in table.findall("tablerow"):
        heads = [cell.get("contents", "") for cell in row.findall("tableheader")]
        if heads:
            header = heads
            continue
        cells = [cell.get("contents", "") for cell in row.findall("tablecell")]
        if not cells:
            continue
        counts: dict[str, int] = {}
        for column, cell in zip(header, cells, strict=True):
            key = "dsp" if column.startswith("DSP") else _HIERARCHY_COLUMNS.get(column)
            if key is not None:
                counts[key] = counts.get(key, 0) + int(cell)
        name = cells[0]
        rows.append(((len(name) - len(name.lstrip(" "))) // 2, name.strip(), _resources(counts)))
    if not rows:
        raise ValueError("no rows in the hierarchical utilization table")
    return rows


def _counts(resources: Resources | None) -> dict[str, int] | None:
    return (
        None
        if resources is None
        else {f.name: getattr(resources, f.name) for f in fields(resources)}
    )


def _less(total: Resources, parts: list[Resources]) -> dict[str, int]:
    """``total`` less the sum of ``parts``, each count as it falls (synthesis and
    placement may count a parent below its children's sum)."""
    claimed = sum(parts, Resources())
    return {f.name: getattr(total, f.name) - getattr(claimed, f.name) for f in fields(total)}


def _instance_of(name: str, instances: Mapping[str, str]) -> str | None:
    """The member whose instance ``name`` is, or begins (an IP's own runs and
    sub-instances: ``axi_interconnect_0_imp_auto_ds``), by the longest instance."""
    found = [
        (len(instance), key)
        for key, instance in instances.items()
        if name == instance or name.startswith(instance.rstrip("_") + "_")
    ]
    return max(found)[1] if found else None


def member_instances(parent: ModelWrapper, completion: Completion | None) -> dict[str, str]:
    """Each member of the shell's block design by its key in the report (``partition``,
    ``ends.<channel>``, ``static_region.<ip>``) and its instance name, or the instance's
    prefix where Vivado names it (the reset's, by its clock). A shell without an
    integration names its partition alone, by its node."""
    node, body, _ = partition_body(parent)
    target = read_target(body)
    row = shell_row(target.shell, target.board)
    if row.integration is None:
        return {"partition": node.name}
    export = integration(parent, completion)
    instances = {"partition": export.partition}
    instances.update({f"ends.{member(end.tensor)}": end.instance for end in export.ends})
    static = row.static_region
    if static is not None:
        instances.update(
            {
                f"static_region.{getattr(static, role)}": instance
                for role, instance in STATIC_INSTANCES.items()
            }
        )
    return instances


def _members(split: Mapping[str, Any]) -> dict[str, Any]:
    """The report's members, ``partition``, ``ends`` by channel and ``static_region`` by
    IP, each with its columns (``model``, ``out_of_context``, ``placed``)."""
    return {
        "partition": split["partition"],
        "ends": {key[len("ends.") :]: row for key, row in split.items() if key.startswith("ends.")},
        "static_region": {
            key[len("static_region.") :]: row
            for key, row in split.items()
            if key.startswith("static_region.")
        },
    }


def shell_resources_report(
    parent: ModelWrapper, completion: Completion | None = None
) -> dict[str, Any]:
    """``report/resources.json`` of the kernel path's parent graph ``parent``, its
    partition completed by ``completion``: see the module docstring. The reports are the
    ones the parent graph states (``finn.outputs`` ``reports``): ``out_of_context`` and
    ``placed`` of the shell's integration, ``ooc_synth`` of the packaged partition."""
    node, body, _ = partition_body(parent)
    target = read_target(body)
    reports = parent.get(OUTPUT_REPORTS) or {}
    point, _ = configured_root(body, node.name, completion)
    modelled = shell_resources(point)
    instances = member_instances(parent, completion)
    columns: dict[str, dict[str, Resources | None]] = {key: {} for key in instances}
    totals: dict[str, Resources | None] = {}
    absent: dict[str, str] = {}

    # The model: the shell root's statements, by member.
    if isinstance(modelled, ShellResources):
        stated: dict[str, Resources | None] = {"partition": modelled.partition}
        stated.update({f"ends.{key}": value for key, value in modelled.ends})
        stated.update({f"static_region.{key}": value for key, value in modelled.static_region})
        for key in instances:
            columns[key]["model"] = stated.get(key)
        totals["model"] = modelled.total
    else:
        absent["model"] = modelled
        totals["model"] = None

    # Out of context: the integration's per-IP runs, or the packaged partition's.
    runs = reports.get("out_of_context")
    packaged = reports.get("ooc_synth")
    if runs is not None:
        per_member: dict[str, Resources] = {}
        unclaimed = Resources()
        for path in sorted(Path(runs).glob("*_utilization_synth.rpt")):
            name = path.name[: -len("_utilization_synth.rpt")]
            name = name[len("top_") :] if name.startswith("top_") else name
            counted = utilization_synth(path.read_text())
            owner = _instance_of(name.removesuffix("_0"), instances)
            if owner is None:
                unclaimed = unclaimed + counted
            else:
                held = per_member.get(owner)
                per_member[owner] = counted if held is None else held + counted
        for key in instances:
            columns[key]["out_of_context"] = per_member.get(key)
        totals["out_of_context"] = sum(per_member.values(), unclaimed)
    elif packaged is not None:
        rows = hierarchical_utilization(Path(packaged).read_text())
        columns["partition"]["out_of_context"] = rows[0][2]
        totals["out_of_context"] = rows[0][2]
    else:
        totals["out_of_context"] = None
        absent["out_of_context"] = (
            "no out-of-context synthesis: ask bitfile on a shell with an integration, or ooc_synth"
        )

    # Placed: the routed design's hierarchy, each instance below the block design's top.
    placed = reports.get("placed")
    if placed is not None:
        rows = placed_hierarchy(Path(placed).read_text())
        for _, name, counted in [row for row in rows if row[0] == 2]:
            owner = _instance_of(name, instances)
            if owner is not None:
                columns[owner]["placed"] = counted
        totals["placed"] = rows[0][2]
    else:
        totals["placed"] = None
        absent["placed"] = (
            f"the {target.shell!r} shell is not placed by the build"
            if shell_row(target.shell, target.board).integration is None
            else "no placed design: bitfile not asked"
        )

    names = ("model", "out_of_context", "placed")
    split = {
        key: {
            "instance": instances[key],
            **{column: _counts(held.get(column)) for column in names},
        }
        for key, held in columns.items()
    }
    unattributed = {
        column: None
        if totals[column] is None
        else _less(
            totals[column],  # type: ignore[arg-type]
            [held[column] for held in columns.values() if held.get(column) is not None],  # type: ignore[misc]
        )
        for column in names
    }
    report: dict[str, Any] = {
        "shell": target.shell,
        "board": target.board,
        "part": target.part,
        "caveat": shell_row(target.shell, target.board).caveat,
        "columns": {
            "model": "the shell root's statements (RESOURCES): the partition's kernels from "
            f"their RTL's fits; each end and each static IP {SHELL_CHARACTERISED}, which "
            "overstate the placed shell",
            "out_of_context": "Vivado's synthesis of each IP alone",
            "placed": "the routed design's hierarchical utilization",
        },
        "members": _members(split),
        "unattributed": unattributed,
        "total": {column: _counts(totals[column]) for column in names},
        "absent": absent,
    }
    if packaged is not None:
        report["partition_members"] = ooc_member_resources(body, Path(packaged).parent, completion)
    return report


__all__ = [
    "RESOURCES_FILE",
    "member_instances",
    "placed_hierarchy",
    "shell_resources_report",
    "utilization_synth",
]
