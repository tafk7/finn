# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build target in a model's graph metadata: the ``finn.platform`` namespace.

Every key is stated: the part, the shell, the board, and the platform's fields (the
clock period and the capabilities, with the part's resources where they are known).
``read_target`` is its one reader and ``write_target`` its one writer. The namespace
inherits, so a partition's body opened through its parent reads the parent's target.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any

from qonnx.core.metadata import JSON, Key, MetadataError, Namespace

from finn.core.graph.errors import KernelOpError
from finn.kernels.target import DspBlock, Platform, Target
from finn.kernels.utilization import Fabric, Resources

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

PLATFORM = Namespace("finn.platform", version=1, inherit=True)
"""The build target, typed graph metadata (qonnx's ``qonnx.core.metadata``): the part,
the shell, the board (``null`` when none is stated) and the platform
(``finn.kernels.target.Platform``: the clock period and the capabilities, its resources
``null`` when the part's are not known), every key stated. It inherits: a subgraph body
reads its parent's. A model stating another shape of it (a key it does not declare, or
one missing) is refused, each key named: it is converted again (``ToKernelOps``)."""

RESOURCE_COUNTS = tuple(field.name for field in fields(Resources))


def _resources_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, dict)
        and set(value) == set(RESOURCE_COUNTS)
        and all(type(count) is int and count >= 0 for count in value.values())
    )


PLATFORM_KEYS: dict[str, Key[Any]] = dict(
    part=PLATFORM.key("part", str, check=bool, expect="a part name"),
    shell=PLATFORM.key("shell", str, check=bool, expect="a shell name"),
    board=PLATFORM.key(
        "board",
        JSON,
        check=lambda v: v is None or (isinstance(v, str) and bool(v)),
        expect="a board name, or null when none is stated",
    ),
    period_ns=PLATFORM.key("period_ns", float, check=lambda v: v > 0, expect="a period > 0"),
    dsp=PLATFORM.key("dsp", DspBlock),
    fabric=PLATFORM.key("fabric", Fabric),
    uram=PLATFORM.key("uram", bool),
    uram_init=PLATFORM.key("uram_init", bool),
    resources=PLATFORM.key(
        "resources",
        JSON,
        check=_resources_or_none,
        expect=f"the counts {', '.join(RESOURCE_COUNTS)}, or null when unknown",
    ),
)

PLATFORM_FIELDS = tuple(field.name for field in fields(Platform))
"""The ``finn.platform`` keys that are ``Platform``'s fields: all but the part, the
shell and the board."""


def read_target(model: ModelWrapper) -> Target:
    """The build target, from the model's ``finn.platform`` metadata (a subgraph body
    opened through its parent reads the parent's).

    The one reader of the target; ``write_target`` is the one writer. A key missing,
    malformed or not declared (another shape's) is refused, by name.
    """
    try:
        stated = model.namespace(PLATFORM)
    except MetadataError as error:
        raise KernelOpError(
            f"the model's target is not one: {error}; run ToKernelOps to state it again"
        ) from error
    missing = [name for name in PLATFORM_KEYS if name not in stated]
    if missing:
        raise KernelOpError(
            f"the model states no target (finn.platform: {', '.join(missing)} missing; "
            "run ToKernelOps)"
        )
    values = {name: stated[name] for name in PLATFORM_FIELDS}
    if values["resources"] is not None:
        values["resources"] = Resources(**values["resources"])
    platform = Platform(**values)
    return Target(
        part=stated["part"], platform=platform, shell=stated["shell"], board=stated["board"]
    )


def write_target(model: ModelWrapper, target: Target) -> None:
    """State the build target in the model's ``finn.platform`` metadata, every key,
    where ``read_target`` reads it."""
    platform = target.platform
    if platform.dsp is None:
        raise KernelOpError("a target states its DSP block (finn.platform.resolve_target)")
    values: dict[str, object] = dict(
        part=target.part,
        shell=target.shell,
        board=target.board,
        **{name: getattr(platform, name) for name in PLATFORM_FIELDS},
    )
    if platform.resources is not None:
        values["resources"] = asdict(platform.resources)
    try:
        for name, key in PLATFORM_KEYS.items():
            key.encode(values[name])  # every key checked before any is written
    except MetadataError as error:
        raise KernelOpError(f"the target is not one: {error}") from error
    for name, key in PLATFORM_KEYS.items():
        model.set(key, values[name])


__all__ = [
    "PLATFORM",
    "PLATFORM_FIELDS",
    "PLATFORM_KEYS",
    "RESOURCE_COUNTS",
    "read_target",
    "write_target",
]
