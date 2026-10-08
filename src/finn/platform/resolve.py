# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one resolution of a build target: a part or a board, a clock period and a shell
to the ``finn.kernels.target.Target`` its kernels are built for."""

from __future__ import annotations

from dataclasses import fields

from finn.kernels.target import Platform, Target
from finn.platform.boards import BOARDS
from finn.platform.catalog import part as catalog_part
from finn.platform.refusal import TargetRefused
from finn.platform.shells import IP, built_shell, shell_row


def resolve_target(
    *, period_ns: float, part: str | None = None, board: str | None = None, shell: str = IP
) -> Target:
    """The target of a build at ``period_ns`` for ``board`` or ``part``, integrated by
    ``shell`` (``ip`` unless one is stated).

    A board names its part; a part stated beside it is an assertion, refused when it
    is not the board's (``board-part-mismatch``). On the ``ip`` shell a board only
    names its part, and the target states none: the packaged IP is built for the part
    and integrated by its user, wherever the part is, so a build that names the board
    and one that names its part make the same model. The platform is the part's
    facts in the catalog (``finn.platform.catalog``: its device's fabric, DSP block,
    UltraRAM and totals) and the clock period; a part the catalog does not have is
    refused (``unknown-part``), and so is one of a device FINN does not build for
    (``unsupported-architecture``). The target names the part as Vivado spells it. What the
    shell gives (its doubled clock, its budgets) is its row's, for the shell and board
    the target names (``finn.platform.shell_row``), and no copy of it is the target's.
    Every refusal is named (``TargetRefused``)."""
    built_shell(shell)
    if not period_ns > 0:
        raise TargetRefused("period-invalid", f"a clock period is > 0 ns, not {period_ns!r}")
    if board is not None:
        if board not in BOARDS:
            raise TargetRefused(
                "unknown-board", f"{board!r} is not a board (one of {sorted(BOARDS)})"
            )
        if part is not None and catalog_part(part).name != BOARDS[board].part:
            raise TargetRefused(
                "board-part-mismatch",
                f"board {board!r} carries {BOARDS[board].part!r}, not {part!r}",
            )
        part = BOARDS[board].part
    if part is None:
        raise TargetRefused("target-unstated", "a target names a part or a board")
    found = catalog_part(part)
    device = found.device
    if device.fabric is None or device.dsp is None:
        raise TargetRefused(
            "unsupported-architecture",
            f"{found.name} ({device.name}, {device.architecture}/{device.family}): "
            f"{device.unsupported}",
        )
    shell_row(shell, board)  # a shell with no row for the board is refused (no-shell-row)
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
    """Refuse a ``build`` (what it is, for the message) whose target, ``built``, is not
    the one a model states (``stated``), each differing field named
    (``target-drift``): a model's target is changed only by converting it again. A
    resolved target names its part as Vivado spells it, so parts compare exactly."""
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
            "target-drift", f"the model's target is not the {build}'s: " + "; ".join(differing)
        )


__all__ = ["refuse_drift", "resolve_target"]
