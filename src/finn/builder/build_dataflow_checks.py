# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build's plan cannot see of its configuration as a whole, checked before the
first step (``configuration_findings``): the settings the run leaves unread, and the
Vivado release a board requires. A rejection refuses the build
(``ConfigurationRefused``); a limitation is logged as a warning, and the build runs. A
step's own rules, an output's shell and the outputs it is made from among them, are the
step's, refused when it runs."""

from finn.builder.build_dataflow_config import DataflowBuildConfig, config_finding
from finn.builder.build_dataflow_plan import Plan
from finn.core.space import Finding, FindingKind
from finn.platform.boards import BOARDS
from finn.platform.target import Target
from finn.toolchain import vivado
from finn.toolchain.launcher import ToolUnavailable


def _board_vivado(cfg: DataflowBuildConfig, target: Target) -> list[Finding]:
    """The rejection of a toolchain whose Vivado is older than the release the target's
    board requires, or that has none. Vivado is asked its release only for such a
    board."""
    required = None if target.board is None else BOARDS[target.board].vivado
    if required is None:
        return []
    needed = "%d.%d" % required
    try:
        found = vivado.release(cfg.prepared_toolchain)
    except (ToolUnavailable, RuntimeError, OSError) as missing:
        stated = f"no Vivado detected: {missing}"
    else:
        if found >= required:
            return []
        stated = "found %d.%d" % found
    message = f"{target.board} requires Vivado {needed} or later, {stated}"
    return [config_finding("board-vivado", message, "configuration.toolchain")]


def configuration_findings(cfg: DataflowBuildConfig, plan: Plan, target: Target) -> list[Finding]:
    """The findings of the checks the module docstring names, for a configuration, its
    ``plan`` and its resolved ``target``: rejections and limitations."""
    findings = _board_vivado(cfg, target)
    run = plan.steps[plan.run]
    read = {name for step in run for name in step.settings}
    for name in sorted(name for name in cfg.settings if name not in read):
        readers = ", ".join(plan.settings[name].readers)
        message = f"only {readers} read it, and start_step/stop_step leave them out"
        place = f"configuration.{name}"
        findings.append(config_finding("setting-unread", message, place, FindingKind.LIMITATION))
    return findings
