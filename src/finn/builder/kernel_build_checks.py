# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's configuration checks: what a KernelBuildConfig's build refuses or
warns of before its first step (``kernel_path_checks``).

``run_all_config_checks`` (``finn.builder.build_dataflow_checks``) runs these and adds
the Vivado release checks. A check is a ``Check`` of a ``Severity``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum

from finn.builder.kernel_build_config import (
    OUTPUT_NEEDS,
    SHELL_OUTPUTS,
    VERIFIED_BY,
    KernelBuildConfig,
    KernelOutputType,
    KernelVerificationStepType,
)
from finn.platform import TargetRefused, shell_row


class Severity(Enum):
    """How a failed check counts: an error stops the build."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Check:
    """One check of a build's configuration and its outcome."""

    name: str
    severity: Severity
    passed: bool
    message: str
    suggestion: str | None = None


def _resolved_step_names(cfg: KernelBuildConfig) -> set[str] | None:
    """The names of the steps the build runs (steps, start_step, stop_step), or None
    when they cannot be resolved (the build then fails on its own when it starts)."""
    try:
        # imported lazily via importlib (rather than a top-level import) since
        # build_dataflow imports the checks, and a top-level import back
        # would be circular
        build_dataflow_mod = importlib.import_module("finn.builder.build_dataflow")
        return {fn.__name__ for fn in build_dataflow_mod.resolve_build_steps(cfg, partial=True)}
    except (ValueError, AttributeError):
        return None


def kernel_path_checks(cfg: KernelBuildConfig) -> list[Check]:
    """The checks of a kernel-path build that its configuration's type does not make
    impossible: its target resolves (finn.platform.resolve_target); the shell outputs
    it asks for (SHELL_OUTPUTS) need a shell that integrates the partition (not
    ``ip``), and each output those it is made from (OUTPUT_NEEDS); the shell's build
    takes its shell_options; the verification input exists, the testbench a
    verification runs is asked for, and the steps each verification needs run
    (VERIFIED_BY)."""
    checks = []
    try:
        target = cfg._resolve_target()
    except TargetRefused as refused:
        return [
            Check(
                "kernel_target",
                Severity.ERROR,
                False,
                str(refused),
                "State a target the platform registry resolves (finn.platform)",
            )
        ]
    row = shell_row(target.shell, target.board)
    shell_outputs = [output for output in cfg.generate_outputs if output in SHELL_OUTPUTS]
    if shell_outputs and row.integration is None:
        asked_shell = ", ".join(output.value for output in shell_outputs)
        checks.append(
            Check(
                "kernel_path_shell",
                Severity.ERROR,
                False,
                f"{asked_shell}: the {target.shell!r} shell does not integrate the partition; "
                "its outputs are the packaged IP's (stitched_ip, ooc_synth)",
                "State a shell with ends that integrates it (pynq, for a board), or "
                "remove them from generate_outputs",
            )
        )
    asked = {KernelOutputType(output) for output in cfg.generate_outputs}
    for output, needs in OUTPUT_NEEDS.items():
        missing = [need.value for need in needs if output in asked and need not in asked]
        if missing:
            checks.append(
                Check(
                    "kernel_output_needs",
                    Severity.ERROR,
                    False,
                    f"{output.value} needs {', '.join(missing)}: it is made from them, "
                    "in the same build",
                    f"Add {', '.join(missing)} to generate_outputs, or remove {output.value}",
                )
            )
    try:
        cfg._resolve_shell_options()
    except ValueError as refused:
        checks.append(
            Check(
                "kernel_shell_options",
                Severity.ERROR,
                False,
                str(refused),
                "State only the options the target's shell's build takes (pynq: "
                "enable_hw_debug), or none",
            )
        )
    if KernelVerificationStepType.PARTITION_PYTHON in cfg.verify_steps and not os.path.isfile(
        cfg.verify_input_npy
    ):
        checks.append(
            Check(
                "verify_files",
                Severity.ERROR,
                False,
                f"verify_input_npy not found: {cfg.verify_input_npy}",
                "Provide a valid verification input .npy file or disable verification",
            )
        )
    testbench = KernelVerificationStepType.STITCHED_IP_TESTBENCH
    if testbench in cfg.verify_steps and KernelOutputType.STITCHED_IP not in asked:
        checks.append(
            Check(
                "kernel_testbench_output",
                Severity.ERROR,
                False,
                f"verify_steps includes {testbench.value}, which runs the testbench "
                f"{KernelOutputType.STITCHED_IP.value} writes, and generate_outputs does "
                "not ask for it",
                f"Add {KernelOutputType.STITCHED_IP.value} to generate_outputs, or remove "
                f"{testbench.value} from verify_steps",
            )
        )
    resolved = _resolved_step_names(cfg)
    if resolved is None:
        return checks
    for each in cfg.verify_steps:
        vstep = KernelVerificationStepType(each)
        unmet = [" or ".join(either) for either in VERIFIED_BY[vstep] if not resolved & set(either)]
        if unmet:
            checks.append(
                Check(
                    "verify_step_prereq",
                    Severity.WARNING,
                    False,
                    f"verify_steps includes {vstep.value}, which needs {'; '.join(unmet)} "
                    "in the resolved build steps (steps/start_step/stop_step). That "
                    "verification would silently never run",
                    "Include those steps in steps, adjust start_step/stop_step so they run, "
                    "or remove it from verify_steps",
                )
            )
    return checks


__all__ = ["Check", "Severity", "kernel_path_checks"]
