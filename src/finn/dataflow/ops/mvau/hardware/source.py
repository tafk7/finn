# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Named source-root resolution and manifest verification for MVAU artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.design import Finding, FindingKind, QualifiedPath
from finn.dataflow.ops.mvau.hardware.dotp_axi import FINNLIB_ROOT
from finn.dataflow.ops.mvau.hardware.replay_buffer import FINN_ROOT
from finn.dataflow.ops.mvau.physical import MVAUElaborationError

_SOURCE_PATH = QualifiedPath("hardware.mvau.decomposed")

FINNLIB_DEFAULT_SUBDIRECTORY = "deps/finnlib"
FINNLIB_ROOT_VARIABLE = "FINNLIB_ROOT"


def _fail(
    code: str, message: str, values: tuple[tuple[str, object], ...] = ()
) -> MVAUElaborationError:
    return MVAUElaborationError(
        (Finding(FindingKind.LIMITATION, code, _SOURCE_PATH, message, values),)
    )


def finnlib_root(finn_root: str | Path) -> Path:
    """Return the configured or pinned FinnLib checkout."""

    override = os.environ.get(FINNLIB_ROOT_VARIABLE)
    if override:
        return Path(override).resolve()
    return (Path(finn_root) / FINNLIB_DEFAULT_SUBDIRECTORY).resolve()


def source_roots(finn_root: str | Path, finnlib: str | Path | None = None) -> dict[str, Path]:
    """Resolve every named root the decomposed Kernels declare sources under."""

    root = Path(finn_root).resolve()
    return {
        FINN_ROOT: root,
        FINNLIB_ROOT: Path(finnlib).resolve() if finnlib is not None else finnlib_root(root),
    }


def resolved_manifest(
    realization: DesignRealization, roots: dict[str, Path]
) -> tuple[tuple[str, str], ...]:
    """Return every declared source in compile order."""

    entries: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for placement in ("replay", "compute"):
        for source in realization.kernel(placement).sources:
            index = counts.get(source.root, 0)
            counts[source.root] = index + 1
            entries.append(
                (f"compute.{source.root}.{index}", str(roots[source.root] / source.path))
            )
    return tuple(entries)


def verify_manifest(entries: tuple[tuple[str, str], ...]) -> None:
    """Refuse a manifest naming source files absent from this checkout."""

    missing = tuple(path for _, path in entries if not Path(path).is_file())
    if missing:
        raise _fail(
            "mvau-decomposed-source-missing",
            "declared decomposed RTL sources do not exist; is FinnLib fetched?",
            (("paths", missing),),
        )


__all__ = [
    "FINNLIB_DEFAULT_SUBDIRECTORY",
    "FINNLIB_ROOT_VARIABLE",
    "finnlib_root",
    "resolved_manifest",
    "source_roots",
    "verify_manifest",
]
