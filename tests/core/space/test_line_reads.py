# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The read inference per scope, ported to an eager model's addresses
(``core.space._line_reads``), answers as the line's inference over a table holding every node
(``_reads.infer`` of ``_structure.expand``'s table) did, by key: each ``self`` method's
possible reads (or dynamic), its paths that name nothing, each guarded member's existence
guards, and each case guard; on every Space class of the space tests, the test support's,
every kernel's and stage 7's chain of eight kernels as one Space.

It checks the port before the table and its inference go: it runs at the commit that adds
the port, and goes with the table."""

from __future__ import annotations

import pkgutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from support.one_space import chain_space

import finn.custom_op
import finn.kernels
from core.space._line_reads import line_infer, line_scopes
from core.space._templates_support import defined, links
from finn.core.space import Space
from finn.core.space._reads import ModelReads, infer
from finn.core.space._structure import expand

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parents[1] / "support"


def _by_key(reads: ModelReads, key: Callable[[Any], str]) -> dict[str, dict[str, object]]:
    def keys(found: Iterable[Any] | None) -> list[str] | None:
        return None if found is None else sorted(map(key, found))

    return {
        "methods": {key(index): keys(found) for index, found in reads.methods.items()},
        "unresolved": {key(index): list(found) for index, found in reads.unresolved.items()},
        "existence": {
            key(index): [key(guard) for guard in found] for index, found in reads.existence.items()
        },
        "cases": {
            key(index): (key(selector), case) for index, (selector, case) in reads.cases.items()
        },
    }


COMPARED: dict[str, int] = {}


def _differences(space_type: type[Space]) -> list[str]:
    table = expand(space_type)
    line = _by_key(infer(table), lambda index: table.nodes[index].key)
    scopes = line_scopes(space_type)
    port = _by_key(line_infer(scopes), lambda index: index.key)
    found: list[str] = []
    for part, held in line.items():
        ported = port[part]
        COMPARED[part] = COMPARED.get(part, 0) + len(held)
        for key in sorted(held.keys() | ported.keys()):
            if held.get(key) != ported.get(key):
                found.append(
                    f"{space_type.__qualname__} {part} {key}:\n"
                    f"  table: {held.get(key)}\n  port: {ported.get(key)}"
                )
    return found[:3]


def _classes() -> list[type[Space]]:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))]
    modules += [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    modules += [
        info.name
        for package in (finn.kernels, finn.custom_op)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    ]
    classes = [space_type for space_type in dict.fromkeys(defined(modules)) if links(space_type)]
    return [*classes, chain_space(4)[0]]


def test_the_ported_inference_answers_as_the_tables_on_every_class() -> None:
    classes = _classes()
    assert len(classes) > 250
    failures = [found for space_type in classes for found in _differences(space_type)]
    assert not failures, "\n".join(failures)
    # Each part compared something: methods (some dynamic), unresolved paths, guards, cases.
    print(f"compared {len(classes)} classes: {COMPARED}")
    assert all(COMPARED[part] > 0 for part in ("methods", "unresolved", "existence", "cases"))
