# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build's plan cannot see of its configuration as a whole, checked before the
first step (``configuration_findings``): the settings the run leaves unread, the tool
releases the target's shell row builds with, and a boundary the build states where the shell's mover
states it. A rejection refuses the build (``ConfigurationRefused``); a limitation is
logged as a warning, and the build runs. A step's own rules, an output's shell and the
outputs it is made from among them, are the step's, refused when it runs."""

from finn.builder.build_dataflow_config import DataflowBuildConfig, config_finding
from finn.builder.build_dataflow_plan import Plan
from finn.core.reporting.findings import Finding, FindingKind
from finn.platform.shells import shell_row
from finn.platform.target import Target
from finn.toolchain import vivado
from finn.toolchain.launcher import ToolUnavailable


def _tool_version(cfg: DataflowBuildConfig, target: Target) -> list[Finding]:
    """The rejection of a toolchain whose Vivado release the target's shell row does not
    build with (``ShellRow.versions``: its linker's, narrowed for its board), or that has
    none. Vivado is asked its release only for a row that states releases."""
    versions = shell_row(target.shell, target.board).versions
    if versions is None:
        return []
    if versions.tool != "vivado":
        message = f"the {target.shell!r} shell's linker runs {versions.tool}, which FINN cannot ask"
        return [config_finding("tool-version", message, "configuration.toolchain")]
    try:
        found = vivado.release(cfg.prepared_toolchain)
    except (ToolUnavailable, RuntimeError, OSError) as missing:
        stated = f"no Vivado detected: {missing}"
    else:
        if versions.admits(found):
            return []
        stated = "found %d.%d" % found
    on = f"{target.board} on the {target.shell!r} shell"
    message = f"{on} builds with Vivado {versions.spec}, {stated}"
    return [config_finding("tool-version", message, "configuration.toolchain")]


def _stated_twice(cfg: DataflowBuildConfig, target: Target) -> list[Finding]:
    """The rejection of each port the build states (``boundary``) on a shell whose row
    offers a mover (``boundary-stated-twice``): the mover states every boundary port it is
    offered on (``MoverOffer.states``), so a build's statement would be a second source.
    Decided from the row and the statements alone; an empty ``boundary`` states nothing."""
    row = shell_row(target.shell, target.board)
    movers = list(dict.fromkeys(mover.name for offer in row.offers for mover in offer.movers))
    if not movers:
        return []
    offered = " and ".join(f"an {mover}" for mover in movers)
    return [
        config_finding(
            "boundary-stated-twice",
            f"the {target.shell!r} shell's row offers {offered} mover on every boundary "
            f"port, which states {port} (row-major, one pass a frame, its pace the "
            "mover's): a build statement would be a second source; state no boundary on "
            "this shell",
            f"configuration.boundary.{port}",
        )
        for port in cfg.boundary
    ]


def configuration_findings(cfg: DataflowBuildConfig, plan: Plan, target: Target) -> list[Finding]:
    """The findings of the checks the module docstring names, for a configuration, its
    ``plan`` and its resolved ``target``: rejections and limitations."""
    findings = [*_tool_version(cfg, target), *_stated_twice(cfg, target)]
    run = plan.steps[plan.run]
    read = {name for step in run for name in step.settings}
    for name in sorted(name for name in cfg.settings if name not in read):
        readers = ", ".join(plan.settings[name].readers)
        message = f"only {readers} read it, and start_step/stop_step leave them out"
        place = f"configuration.{name}"
        findings.append(config_finding("setting-unread", message, place, FindingKind.LIMITATION))
    return findings
