# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact whole-design physical realization records and validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from finn.dataflow.design import Answer, Decided, Finding, FindingKind, QualifiedPath, Unresolved
from finn.dataflow.hardware.kernel import HardwareKernel
from finn.dataflow.network import DataflowNetwork

DESIGN_REALIZATION_PATH = QualifiedPath("hardware.design_realization")


def _finding(code: str, message: str, **values: object) -> Finding:
    return Finding(
        FindingKind.LIMITATION,
        code,
        DESIGN_REALIZATION_PATH,
        message,
        tuple(values.items()),
    )


@dataclass(frozen=True)
class DesignRealization:
    """One resolved Network and the bindings that cover it exactly."""

    design_id: str
    network: DataflowNetwork
    kernels: Mapping[str, HardwareKernel]
    unabsorbed_edges: tuple[str, ...]
    boundaries: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernels", MappingProxyType(dict(self.kernels)))

    def kernel(self, placement: str) -> HardwareKernel:
        try:
            return self.kernels[placement]
        except KeyError:
            raise KeyError(f"no configured Kernel for placement {placement!r}") from None


def validate_realization(
    design_id: str,
    network: DataflowNetwork,
    placed: Mapping[str, HardwareKernel],
) -> Answer[DesignRealization]:
    """Validate exact node, edge, fan-out, and boundary coverage."""

    findings: list[Finding] = []
    node_counts = Counter(node for kernel in placed.values() for node in kernel.node_ids)
    edge_counts = Counter(edge for kernel in placed.values() for edge in kernel.edge_ids)
    network_nodes = {item.id for item in network.nodes}
    network_edges = {item.id: item for item in network.edges}

    for node in sorted(network_nodes):
        count = node_counts[node]
        if count != 1:
            findings.append(
                _finding(
                    "design-node-coverage-not-exact",
                    f"Network node {node!r} is covered {count} times instead of once",
                    node=node,
                    count=count,
                )
            )
    for node in sorted(set(node_counts) - network_nodes):
        findings.append(
            _finding(
                "design-foreign-node-coverage",
                f"configured Kernels cover foreign node {node!r}",
                node=node,
            )
        )
    for edge_id, count in sorted(edge_counts.items()):
        edge = network_edges.get(edge_id)
        if edge is None:
            findings.append(
                _finding(
                    "design-foreign-edge-absorption",
                    f"configured Kernels absorb foreign edge {edge_id!r}",
                    edge=edge_id,
                )
            )
            continue
        if count != 1:
            findings.append(
                _finding(
                    "design-edge-absorption-not-unique",
                    f"Network edge {edge_id!r} is absorbed {count} times",
                    edge=edge_id,
                    count=count,
                )
            )
        absorber = next(kernel for kernel in placed.values() if edge_id in kernel.edge_ids)
        required = {edge.source.node_id, *(sink.endpoint.node_id for sink in edge.sinks)}
        covered = set(absorber.node_ids)
        if not required <= covered:
            findings.append(
                _finding(
                    "design-absorbed-edge-incomplete-fanout",
                    f"Kernel {absorber.kernel_id!r} absorbs edge {edge_id!r} without "
                    "covering its source and every sink",
                    edge=edge_id,
                    missing=tuple(sorted(required - covered)),
                )
            )
    if findings:
        return Unresolved(tuple(findings))
    return Decided(
        DesignRealization(
            design_id,
            network,
            placed,
            tuple(sorted(set(network_edges) - set(edge_counts))),
            tuple(item.id for item in network.boundaries),
        )
    )


__all__ = ["DESIGN_REALIZATION_PATH", "DesignRealization", "validate_realization"]
