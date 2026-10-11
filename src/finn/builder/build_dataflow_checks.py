# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration check system for FINN builds - catches incompatibilities early.

What a DataflowBuildConfig's build refuses or warns of before its first step: the kernel
path's own checks (``kernel_path_checks``) and the Vivado release its board's row
requires, against the release its toolchain runs (``run_all_config_checks`` runs both).
A check is a ``Check`` of a ``Severity``.
"""

import importlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from finn.builder.build_dataflow_config import (
    OUTPUT_NEEDS,
    SHELL_OUTPUTS,
    VERIFIED_BY,
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.platform import BOARDS, Board, TargetRefused, shell_row
from finn.toolchain import vivado
from finn.toolchain.launcher import ToolUnavailable


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


@dataclass
class Report:
    timestamp: str
    vivado_version: Optional[Tuple[int, int]]
    checks: List[Check] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(c.severity == Severity.ERROR and not c.passed for c in self.checks)


def _check(
    name: str,
    severity: Severity,
    condition: object,
    msg_fail: str,
    suggestion: Optional[str] = None,
) -> Check:
    """Helper to create a Check result."""
    if condition:
        return Check(name, severity, True, "OK")
    return Check(name, severity, False, msg_fail, suggestion)


def _resolved_step_names(cfg: DataflowBuildConfig) -> set[str] | None:
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


def kernel_path_checks(cfg: DataflowBuildConfig) -> list[Check]:
    """The checks of a kernel-path build that its configuration's type does not make
    impossible: its target resolves (finn.platform.resolve_target); the shell outputs
    it asks for (SHELL_OUTPUTS) need a shell that integrates the partition (not
    ``ip``), and each output those it is made from (OUTPUT_NEEDS); enable_hw_debug
    only on a shell with a build; the verification input exists, the testbench a
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
    asked = {DataflowOutputType(output) for output in cfg.generate_outputs}
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
        cfg._resolve_shell_build_options()
    except ValueError as refused:
        checks.append(
            Check(
                "kernel_hw_debug",
                Severity.ERROR,
                False,
                str(refused),
                "State enable_hw_debug only for a shell with a build (pynq), as a bool",
            )
        )
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps and not os.path.isfile(
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
    testbench = VerificationStepType.STITCHED_IP_TESTBENCH
    if testbench in cfg.verify_steps and DataflowOutputType.STITCHED_IP not in asked:
        checks.append(
            Check(
                "testbench_output",
                Severity.ERROR,
                False,
                f"verify_steps includes {testbench.value}, which runs the testbench "
                f"{DataflowOutputType.STITCHED_IP.value} writes, and generate_outputs does "
                "not ask for it",
                f"Add {DataflowOutputType.STITCHED_IP.value} to generate_outputs, or remove "
                f"{testbench.value} from verify_steps",
            )
        )
    resolved = _resolved_step_names(cfg)
    if resolved is None:
        return checks
    for each in cfg.verify_steps:
        vstep = VerificationStepType(each)
        unmet = [step for step in VERIFIED_BY[vstep] if step not in resolved]
        if unmet:
            checks.append(
                Check(
                    "verify_step_prereq",
                    Severity.WARNING,
                    False,
                    f"verify_steps includes {vstep.value}, which needs {', '.join(unmet)} "
                    "in the resolved build steps (steps/start_step/stop_step). That "
                    "verification would silently never run",
                    "Include those steps in steps, adjust start_step/stop_step so they run, "
                    "or remove it from verify_steps",
                )
            )
    return checks


def _vivado_release(cfg: DataflowBuildConfig) -> Tuple[Optional[Tuple[int, int]], str]:
    """The Vivado release the build's own toolchain runs (``finn.toolchain.vivado.release``),
    or None and why when none is detected: no Vivado on its route, or its toolchain not
    prepared."""
    try:
        return vivado.release(cfg._resolve_toolchain()), ""
    except (ToolUnavailable, RuntimeError, OSError) as missing:
        return None, str(missing)


def _vivado_checks(board: Optional[Board], v: Optional[Tuple[int, int]], why: str) -> List[Check]:
    """Whether ``v``, the Vivado release detected (None: none, for the reason ``why``),
    meets the release ``board``'s row requires (``Board.vivado``); no check for a board
    without a requirement, or no board."""
    if board is None or board.vivado is None:
        return []
    needed = f"{board.vivado[0]}.{board.vivado[1]}"
    found = f"found {v[0]}.{v[1]}" if v else f"no Vivado detected: {why}"
    return [
        _check(
            "board_vivado",
            Severity.ERROR,
            v is not None and v >= board.vivado,
            f"{board.name} requires Vivado {needed} or later, {found}",
            f"Select Vivado {needed} or later in the build's toolchain",
        )
    ]


def run_all_config_checks(cfg: DataflowBuildConfig) -> Report:
    """Run all configuration checks of a kernel-path build and return the report: the
    Vivado release its board needs, and the kernel path's own checks
    (kernel_path_checks). The report states the release the build's toolchain runs."""
    v, why = _vivado_release(cfg)
    board = None
    try:
        name = cfg._resolve_target().board
        board = BOARDS[name] if name is not None else None
    except TargetRefused:
        pass  # kernel_path_checks names it
    checks = _vivado_checks(board, v, why) + kernel_path_checks(cfg)
    return Report(timestamp=datetime.now().isoformat(), vivado_version=v, checks=checks)


def format_report(report: Report) -> str:
    """Format report for human-readable output."""
    lines = ["=" * 70, "FINN Build Configuration Check Report", "=" * 70]
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(
        f"Vivado: {report.vivado_version[0]}.{report.vivado_version[1]}"
        if report.vivado_version
        else "Vivado: Not detected"
    )
    lines.append("-" * 70)

    errors = [c for c in report.checks if c.severity == Severity.ERROR and not c.passed]
    warnings = [c for c in report.checks if c.severity == Severity.WARNING and not c.passed]
    infos = [c for c in report.checks if c.severity == Severity.INFO and not c.passed]

    for label, symbol, items in [
        ("ERRORS", "X", errors),
        ("WARNINGS", "!", warnings),
        ("INFO", "i", infos),
    ]:
        if items:
            lines.append(f"\n{label}:")
            for c in items:
                lines.append(f"  [{symbol}] {c.name}: {c.message}")
                if c.suggestion:
                    lines.append(f"      -> {c.suggestion}")

    lines.append(f"\nSUMMARY: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")
    lines.append("=" * 70)
    return "\n".join(lines)


def save_report(report: Report, output_dir: str) -> str:
    """Save report to output_dir as .txt and .json. Returns path to text report."""
    txt_path = os.path.join(output_dir, "config_check_report.txt")
    json_path = os.path.join(output_dir, "config_check_report.json")

    with open(txt_path, "w") as f:
        f.write(format_report(report))

    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "vivado_version": list(report.vivado_version) if report.vivado_version else None,
                "checks": [
                    {
                        "name": c.name,
                        "severity": c.severity.value,
                        "passed": c.passed,
                        "message": c.message,
                        "suggestion": c.suggestion,
                    }
                    for c in report.checks
                ],
                "summary": {
                    "errors": sum(
                        1 for c in report.checks if c.severity == Severity.ERROR and not c.passed
                    ),
                    "warnings": sum(
                        1 for c in report.checks if c.severity == Severity.WARNING and not c.passed
                    ),
                    "info": sum(
                        1 for c in report.checks if c.severity == Severity.INFO and not c.passed
                    ),
                },
            },
            f,
            indent=2,
        )

    return txt_path
