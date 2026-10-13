# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one resolution of a build target: a part or a board, a clock period and a shell
to the ``finn.platform.target.Target`` its kernels are built for."""

from __future__ import annotations

from dataclasses import fields

from finn.platform.boards import BOARDS
from finn.platform.capabilities import Platform
from finn.platform.catalog import part as catalog_part
from finn.platform.refusal import TargetRefused
from finn.platform.shells import IP, built_shell, shell_row
from finn.platform.target import Target


def resolve_target(
    *, period_ns: float, part: str | None = None, board: str | None = None, shell: str = IP
) -> Target:
    """Resolve a build's target. A board gives its part, and a part given with it must
    match. On the ``ip`` shell the board is dropped from the target, so naming a board
    or its part gives the same model. The shell's own facts (doubled clock, budgets)
    stay in its row and are not copied into the target. Refusals are
    ``TargetRefused``."""
    built_shell(shell)
    if not period_ns > 0:
        raise TargetRefused(
            "period-invalid",
            "configuration.target.period_ns",
            f"a clock period is > 0 ns, not {period_ns!r}",
        )
    if board is not None:
        if board not in BOARDS:
            raise TargetRefused(
                "unknown-board",
                "configuration.target.board",
                f"{board!r} is not a board (one of {sorted(BOARDS)})",
            )
        if part is not None and catalog_part(part).name != BOARDS[board].part:
            raise TargetRefused(
                "board-part-mismatch",
                "configuration.target.part",
                f"board {board!r} carries {BOARDS[board].part!r}, not {part!r}",
            )
        part = BOARDS[board].part
    if part is None:
        raise TargetRefused(
            "target-unstated", "configuration.target", "a target names a part or a board"
        )
    found = catalog_part(part)
    device = found.device
    if device.fabric is None or device.dsp is None:
        raise TargetRefused(
            "unsupported-architecture",
            "configuration.target.part",
            f"{found.name} ({device.name}, {device.architecture}/{device.family}): "
            f"{device.unsupported}",
        )
    shell_row(shell, board)  # refuses a board the shell has no row for
    resources = device.resources
    platform = Platform(
        period_ns=float(period_ns),
        dsp=device.dsp,
        fabric=device.fabric,
        uram=resources.uram > 0,
        uram_init=device.uram_init,
        resources=resources,
    )
    return Target(
        part=found.name, platform=platform, shell=shell, board=None if shell == IP else board
    )


def refuse_drift(stated: Target, built: Target, build: str) -> None:
    """Refuse a build whose target (``built``) differs from the model's (``stated``),
    naming each differing field; ``build`` names the build in the message. A model's
    target changes only by converting it again. A resolved target spells its part as
    Vivado does, so parts compare exactly."""
    pairs = [
        ("part", stated.part, built.part),
        ("shell", stated.shell, built.shell),
        ("board", stated.board, built.board),
    ] + [
        (field.name, getattr(stated.platform, field.name), getattr(built.platform, field.name))
        for field in fields(Platform)
    ]
    differing = [
        f"{name}: the model states {model_value!r}, the {build} {build_value!r}"
        for name, model_value, build_value in pairs
        if model_value != build_value
    ]
    if differing:
        raise TargetRefused(
            "target-drift",
            "configuration.target",
            f"the model's target is not the {build}'s: " + "; ".join(differing),
        )


__all__ = ["refuse_drift", "resolve_target"]
