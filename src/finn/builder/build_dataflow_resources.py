# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's resources report, ``report/resources.json``: resources per member of the
shell.

``shell_resources_report`` lists each member of the partition's shell (the partition,
each mover by its boundary channel, each IP of the static region) and their total, in up
to three columns:

- ``model``: the partition's network's own figures
  (``finn.custom_op.kernels.shell.shell_resources``): the partition's kernels from their
  RTL's fitted models, each mover and each static IP from its out-of-context
  characterisation;
- ``out_of_context``: Vivado's synthesis of each IP alone. On a shell with an
  integration, these are the integration's per-IP runs (``<run>_utilization_synth.rpt``,
  read by ``finn.toolchain.vivado.utilization_synth``), matched to members by instance
  name (``_member_instances``). On ``ip``, it is the top row of the hierarchical report
  of the partition packaged with ``ooc_synth`` (``vivado.hierarchical_utilization``);
- ``placed``: the routed design's hierarchical report (``vivado.placed_hierarchy``),
  each instance directly below the block design's top counted to its member. ``ip`` is
  not placed.

A column the build did not produce is ``null`` for every member, and ``absent`` says
why. A member a column has no row for (the placed design has none for the processor and
its reset) is ``null`` there. ``caveat`` is the shell row's note for the board
(``finn.platform.shells.ShellRow.caveat``), or ``null``: on a board the shell was not built and
timed on, the ``model`` figures of the movers and static region are carried over from
where they were characterised.

``unattributed`` is each column's total minus the sum of its members: what the report
counts outside the members. It is not clipped, and can be negative in the placed
column, because Vivado's hierarchical report can count children above their top row
(TFC on Ultra96: the members' LUTs exceed it by 3).

When the partition was synthesized out of context (``ooc_synth``), ``partition_members``
splits it by member of the partition's network (``package.ooc_member_resources``).
Synthesis flattens small instances into the module's top and does not keep the
hierarchy: what no reported instance holds is listed as ``unattributed`` there, and the
instances without a row as ``unreported_instances``.

All counts are in ``finn.platform.capabilities.Resources`` units: LUTs, FFs, RAMB18
halves (a RAMB36 counts two), URAMs and DSPs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.annotations.outputs import OUTPUT_IP, OUTPUT_IP_KEY, OUTPUT_IP_OPTIONS
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import partition_body
from finn.custom_op.kernels.shell import ShellResources, configured_network, shell_resources
from finn.kernels.composer import member_name
from finn.kernels.utilization import SHELL_CHARACTERISED
from finn.platform.capabilities import Resources
from finn.platform.shells import shell_row
from finn.shells.pynq.runner import REPORTS, STATIC_INSTANCES, bitfile_directory
from finn.toolchain import vivado
from finn.toolchain.launcher import Toolchain, ToolFailed
from finn.transformation.kernels.integration import integration
from finn.transformation.kernels.package import (
    ip_key,
    ooc_member_resources,
    partition_utilization_report,
    resources_of,
)

#: The report's file, under the output directory.
RESOURCES_FILE = "report/resources.json"


def _counts(resources: Resources | None) -> dict[str, int] | None:
    return (
        None
        if resources is None
        else {f.name: getattr(resources, f.name) for f in fields(resources)}
    )


def _less(total: Resources, parts: list[Resources]) -> dict[str, int]:
    """``total`` minus the sum of ``parts``, per resource and not clipped (synthesis and
    placement may count a parent below its children's sum)."""
    claimed = sum(parts, Resources())
    return {f.name: getattr(total, f.name) - getattr(claimed, f.name) for f in fields(total)}


def _instance_of(name: str, instances: Mapping[str, str]) -> str | None:
    """The member whose instance is ``name``, or a prefix of it (an IP's own runs and
    sub-instances, such as ``axi_interconnect_0_imp_auto_ds``); the longest match wins."""
    found = [
        (len(instance), key)
        for key, instance in instances.items()
        if name == instance or name.startswith(instance.rstrip("_") + "_")
    ]
    return max(found)[1] if found else None


def _member_instances(parent: ModelWrapper) -> dict[str, str]:
    """Each member of the shell's block design by its report key (``partition``,
    ``movers.<channel>``, ``static_region.<ip>``), with its instance name, or the prefix of
    a name Vivado chooses (the reset's, after its clock). A shell without an integration
    has the partition alone, by its node's name."""
    node, body, _ = partition_body(parent)
    target = read_target(body)
    row = shell_row(target.shell, target.board)
    if row.integration is None:
        return {"partition": node.name}
    export = integration(parent)
    instances = {"partition": export.partition}
    instances.update(
        {f"movers.{member_name(mover.tensor)}": mover.instance for mover in export.movers}
    )
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
    """The report's members, ``partition``, ``movers`` by channel and ``static_region`` by
    IP, each with its columns (``model``, ``out_of_context``, ``placed``)."""
    return {
        "partition": split["partition"],
        "movers": {
            key[len("movers.") :]: row for key, row in split.items() if key.startswith("movers.")
        },
        "static_region": {
            key[len("static_region.") :]: row
            for key, row in split.items()
            if key.startswith("static_region.")
        },
    }


def built_reports(parent: ModelWrapper, *, toolchain: Toolchain) -> dict[str, Path]:
    """The reports the build made that ``shell_resources_report`` reads, by name: those
    collected beside the bitfile the parent graph ``parent`` states (``placed``,
    ``out_of_context``; none without a bitfile), and the utilization report of the IP
    its partition's body states, packaged with ``ooc_synth`` and of the key the body has
    now (``ip_key``, staged in ``toolchain``) (``ooc_synth``; none when the body states
    no such IP)."""
    reports: dict[str, Path] = {}
    built = bitfile_directory(parent)
    if built is not None:
        reports.update({name: built / REPORTS[name] for name in ("placed", "out_of_context")})
    node, body, _ = partition_body(parent)
    ip, packaging = body.get(OUTPUT_IP), body.get(OUTPUT_IP_OPTIONS)
    if ip is None or packaging is None or not packaging["run_synth"]:
        return reports
    if body.get(OUTPUT_IP_KEY) == ip_key(body, node.name, run_synth=True, toolchain=toolchain):
        try:
            reports["ooc_synth"] = partition_utilization_report(Path(ip).parent)
        except ToolFailed:
            pass  # no report: the column is absent
    return reports


def shell_resources_report(parent: ModelWrapper, reports: Mapping[str, Path]) -> dict[str, Any]:
    """``report/resources.json`` for the parent graph ``parent`` (see the module
    docstring), from the ``reports`` the build made, by name: the shell integration's
    ``out_of_context`` (a directory) and ``placed`` (``finn.shells.pynq.runner.REPORTS``),
    and the packaged partition's ``ooc_synth``. A column whose report is not given is
    absent."""
    node, body, _ = partition_body(parent)
    target = read_target(body)
    point, _ = configured_network(body, node.name)
    modelled = shell_resources(point)
    instances = _member_instances(parent)
    columns: dict[str, dict[str, Resources | None]] = {key: {} for key in instances}
    totals: dict[str, Resources | None] = {}
    absent: dict[str, str] = {}

    # The model column: the partition's network's figures, by member.
    if isinstance(modelled, ShellResources):
        stated: dict[str, Resources | None] = {"partition": modelled.partition}
        stated.update({f"movers.{key}": value for key, value in modelled.movers})
        stated.update({f"static_region.{key}": value for key, value in modelled.static_region})
        for key in instances:
            columns[key]["model"] = stated.get(key)
        totals["model"] = modelled.total
    else:
        absent["model"] = "; ".join(map(str, modelled))
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
            counted = resources_of(vivado.utilization_synth(path.read_text()))
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
        rows = vivado.hierarchical_utilization(Path(packaged).read_text())
        columns["partition"]["out_of_context"] = resources_of(rows[0][2])
        totals["out_of_context"] = resources_of(rows[0][2])
    else:
        totals["out_of_context"] = None
        absent["out_of_context"] = (
            "no out-of-context synthesis: list step_build_bitfile on a shell with an "
            "integration, or step_package_partition with ooc_synth"
        )

    # Placed: the routed design's hierarchy, each instance directly below the top.
    placed = reports.get("placed")
    if placed is not None:
        rows = vivado.placed_hierarchy(Path(placed).read_text())
        for _, name, counts in [row for row in rows if row[0] == 2]:
            owner = _instance_of(name, instances)
            if owner is not None:
                columns[owner]["placed"] = resources_of(counts)
        totals["placed"] = resources_of(rows[0][2])
    else:
        totals["placed"] = None
        absent["placed"] = (
            f"the {target.shell!r} shell is not placed by the build"
            if shell_row(target.shell, target.board).integration is None
            else "no placed design: step_build_bitfile not listed"
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
            "model": "the network's statements (its members' resources): the partition's kernels "
            f"from their RTL's fits; each mover and each static IP {SHELL_CHARACTERISED}, which "
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
        report["partition_members"] = ooc_member_resources(body, Path(packaged).parent)
    return report


__all__ = ["RESOURCES_FILE", "built_reports", "shell_resources_report"]
