# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build's plan cannot see of its configuration, checked before the first step
(``configuration_findings``): the outputs and the shell they need, ``hw_debug``, the
settings the run leaves unread, the verification inputs and steps, and the Vivado
release a board requires. A rejection refuses the build (``ConfigurationRefused``); a
limitation is logged as a warning, and the build runs."""

import os

from finn.builder.build_dataflow_config import (
    OUTPUT_NEEDS,
    SHELL_OUTPUTS,
    VERIFIED_BY,
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
    config_finding,
)
from finn.builder.build_dataflow_plan import BuildPlan
from finn.core.space import Finding, FindingKind
from finn.kernels.target import Target
from finn.platform import BOARDS, VIVADO_BLOCK_DESIGN, shell_row
from finn.toolchain import vivado
from finn.toolchain.launcher import ToolUnavailable


def _warning(code: str, message: str, place: str) -> Finding:
    """A limitation: logged as a warning, and the build runs."""
    return config_finding(code, message, place, FindingKind.LIMITATION)


def required_vivado(target: Target) -> tuple[int, int] | None:
    """The Vivado release the target's board requires (``Board.vivado``), if any."""
    return None if target.board is None else BOARDS[target.board].vivado


def _board_vivado(cfg: DataflowBuildConfig, target: Target) -> list[Finding]:
    """The rejection of a toolchain whose Vivado is older than the release the target's
    board requires, or that has none. Vivado is asked its release only for such a
    board."""
    required = required_vivado(target)
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


def configuration_findings(
    cfg: DataflowBuildConfig, plan: BuildPlan, target: Target
) -> list[Finding]:
    """The findings of the checks the module docstring names, for a configuration, its
    ``plan`` and its resolved ``target``: rejections and limitations."""
    findings = _board_vivado(cfg, target)
    row = shell_row(target.shell, target.board)
    shell_outputs = [output for output in cfg.generate_outputs if output in SHELL_OUTPUTS]
    if shell_outputs and row.integration is None:
        asked_shell = ", ".join(output.value for output in shell_outputs)
        message = (
            f"{asked_shell}: the {target.shell!r} shell does not integrate the partition; "
            "its outputs are the packaged IP's (stitched_ip, ooc_synth)"
        )
        findings.append(config_finding("shell-outputs", message, "configuration.generate_outputs"))
    asked = {DataflowOutputType(output) for output in cfg.generate_outputs}
    for output, needs in OUTPUT_NEEDS.items():
        missing = [need.value for need in needs if output in asked and need not in asked]
        if missing:
            message = (
                f"{output.value} needs {', '.join(missing)}: it is made from them, in the "
                "same build"
            )
            findings.append(
                config_finding("output-needs", message, "configuration.generate_outputs")
            )
    hw_debug = plan.settings.get("hw_debug")
    if hw_debug is not None and hw_debug.value and row.integration != VIVADO_BLOCK_DESIGN:
        message = f"the {target.shell!r} shell has no build to take it"
        findings.append(config_finding("hw-debug-shell", message, "configuration.hw_debug"))
    for name in plan.unread:
        readers = ", ".join(plan.settings[name].readers)
        message = f"only {readers} read it, and start_step/stop_step leave them out"
        findings.append(_warning("setting-unread", message, f"configuration.{name}"))
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps and not os.path.isfile(
        cfg.verify_input_npy
    ):
        message = f"not found: {cfg.verify_input_npy}"
        findings.append(
            config_finding("verify-input-missing", message, "configuration.verify_input_npy")
        )
    testbench = VerificationStepType.STITCHED_IP_TESTBENCH
    if testbench in cfg.verify_steps and DataflowOutputType.STITCHED_IP not in asked:
        message = (
            f"{testbench.value} runs the testbench {DataflowOutputType.STITCHED_IP.value} "
            "writes, and generate_outputs does not ask for it"
        )
        findings.append(config_finding("testbench-output", message, "configuration.verify_steps"))
    resolved = {step.name for step in plan.steps}
    for each in cfg.verify_steps:
        vstep = VerificationStepType(each)
        *saving, concluding = VERIFIED_BY[vstep]
        unsaved = [step for step in saving if step not in resolved]
        if concluding not in resolved:
            message = (
                f"{vstep.value}, which {concluding} runs, and the resolved build steps "
                "(steps/start_step/stop_step) leave it out: that verification does not run, "
                "and has no verdict"
            )
            findings.append(_warning("verification-not-run", message, "configuration.verify_steps"))
        elif unsaved:
            message = (
                f"{vstep.value}, whose reference {', '.join(unsaved)} saves, and the resolved "
                f"build steps (steps/start_step/stop_step) leave it out: {concluding} reads "
                "the reference an earlier run left in the output directory, and fails "
                "(FileNotFoundError) where there is none"
            )
            findings.append(
                _warning("verification-unreferenced", message, "configuration.verify_steps")
            )
    return findings
