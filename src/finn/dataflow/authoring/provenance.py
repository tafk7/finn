# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Problem-field ownership metadata for the dataflow adapter compiler."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from finn.dataflow.authoring.scope import AuthoringError
from finn.dataflow.design import QualifiedPath


class Provenance(Enum):
    """Who is entitled to supply one problem field."""

    GRAPH = "graph"
    GRAPH_ANALYSIS = "graph_analysis"
    TARGET = "target"
    BUILD = "build"
    UPSTREAM = "upstream"


GRAPH_OWNED: frozenset[Provenance] = frozenset({Provenance.GRAPH, Provenance.GRAPH_ANALYSIS})
BUILD_OWNED: frozenset[Provenance] = frozenset(
    {Provenance.TARGET, Provenance.BUILD, Provenance.UPSTREAM}
)


@dataclass(frozen=True, slots=True)
class ProblemProvenance:
    """The declared owner of every projected problem field."""

    kinds: Mapping[QualifiedPath, Provenance]

    def kind_of(self, path: QualifiedPath) -> Provenance | None:
        return self.kinds.get(path)

    def paths_for(self, *kinds: Provenance | Iterable[Provenance]) -> frozenset[QualifiedPath]:
        wanted: set[Provenance] = set()
        for item in kinds:
            if isinstance(item, Provenance):
                wanted.add(item)
            else:
                wanted.update(item)
        return frozenset(path for path, kind in self.kinds.items() if kind in wanted)

    def check(
        self,
        supplied: Mapping[QualifiedPath, object],
        *,
        allowed: Iterable[Provenance],
        owner: str,
    ) -> None:
        permitted = frozenset(allowed)
        unknown = sorted(str(path) for path in supplied if path not in self.kinds)
        misowned = sorted(
            f"{path} is {self.kinds[path].value}"
            for path in supplied
            if path in self.kinds and self.kinds[path] not in permitted
        )
        if unknown or misowned:
            raise AuthoringError(
                f"{owner} projection supplied fields it does not own; "
                f"undeclared {unknown}, wrong provenance {misowned}"
            )

    def check_graph_projection(self, supplied: Mapping[QualifiedPath, object]) -> None:
        self.check(supplied, allowed=GRAPH_OWNED, owner="graph")

    def check_build_projection(self, supplied: Mapping[QualifiedPath, object]) -> None:
        self.check(supplied, allowed=BUILD_OWNED, owner="build")


__all__ = ["BUILD_OWNED", "GRAPH_OWNED", "ProblemProvenance", "Provenance"]
