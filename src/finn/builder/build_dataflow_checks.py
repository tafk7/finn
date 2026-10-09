# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration check system for FINN builds - catches incompatibilities early."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from finn.builder.kernel_build_checks import Check, Severity, kernel_path_checks
from finn.builder.kernel_build_config import KernelBuildConfig
from finn.platform import TargetRefused
from finn.util.basic import get_vivado_version


@dataclass
class Report:
    timestamp: str
    vivado_version: Optional[Tuple[int, int]]
    checks: List[Check] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(c.severity == Severity.ERROR and not c.passed for c in self.checks)


def _check(name, severity, condition, msg_fail, suggestion=None):
    """Helper to create a Check result."""
    if condition:
        return Check(name, severity, True, "OK")
    return Check(name, severity, False, msg_fail, suggestion)


def _vivado_checks(board: Optional[str], v: Optional[Tuple[int, int]]) -> List[Check]:
    """The Vivado release a build for ``board`` needs, and whether ``v`` is one FINN
    recommends."""
    checks = []
    if board == "V80":
        checks.append(
            _check(
                "v80_vivado",
                Severity.ERROR,
                v and v >= (2024, 2),
                "V80 board requires Vivado 2024.2 or later, "
                f"found {v[0]}.{v[1] if v else 'unknown'}",
                "Upgrade to Vivado 2024.2 or later to use V80",
            )
        )
    if board == "AUP-ZU3_8GB":
        checks.append(
            _check(
                "aupzu3_vivado",
                Severity.ERROR,
                v and v >= (2024, 1),
                f"AUP-ZU3_8GB board requires Vivado 2024.1 or later, "
                f"found {v[0]}.{v[1] if v else 'unknown'}",
                "Upgrade to Vivado 2024.1 or later to use AUP-ZU3_8GB",
            )
        )
    if v and v not in [(2022, 2), (2024, 2)]:
        checks.append(
            _check(
                "vivado_stable",
                Severity.INFO,
                False,
                f"Vivado {v[0]}.{v[1]} is used. Recommended versions are "
                "2022.2 and 2024.2, other versions can be used but may have unexpected issues.",
            )
        )
    return checks


def run_all_config_checks(cfg: KernelBuildConfig) -> Report:
    """Run all configuration checks of a kernel-path build and return the report: the
    Vivado release its board needs, and the kernel path's own checks
    (finn.builder.kernel_build_checks)."""
    v = get_vivado_version()
    board = None
    try:
        board = cfg._resolve_target().board
    except TargetRefused:
        pass  # kernel_path_checks names it
    checks = _vivado_checks(board, v) + kernel_path_checks(cfg)
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
