# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a kernel's spec states (``KernelSpec``), and the probes of its refused side."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from finn.core.space import Space
from finn.harness.points import Refused
from finn.kernels.base import Kernel
from support.conformance import KERNEL, configured, design_member, unchosen

Case = Callable[[], dict[str, Any]]
"""A conformance case: the arguments of ``support.conformance.conformance``, its
test-side ``reference`` among them."""

EMPTY: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True)
class Probe:
    """A point the kernel must refuse, offline, and the finding codes it is refused with.

    ``build`` binds the point. Without ``choices`` it is the kernel itself, refused by
    its own validity on its facts (``finn.harness.points.rejected``); with them it is any
    point, and committing ``choices`` (by decision key), or the configuration they make,
    is what it refuses (``finn.harness.points.refusal``)."""

    label: str
    build: Callable[[], Space]
    codes: frozenset[str]
    choices: Mapping[str, object] = EMPTY


@dataclass(frozen=True)
class SweepCases:
    """A numeric sweep's cases of the kernel: the module ``scripts/xsim-sweep.sh`` runs
    as its ``jobs`` (``scripts/emitted_text.py``'s ``SWEEPS``), and the module's case
    lists that place it."""

    module: str
    lists: tuple[str, ...]
    jobs: tuple[str, ...]


@dataclass(frozen=True)
class KernelSpec:
    """One kernel's evidence, in one place: what checks it at each level, and the lean
    points (decision KT10) its design space is checked at.

    - ``reference``: where its computational reference lives. A kernel a KernelOp
      reaches is checked against the op's ``execute_node`` at the op's boundary (the
      oracle, PRINCIPLES 8: ``tests/kernel_ops/test_parity.py``); every kernel's
      conformance cases state a test-side reference, the only one of a kernel no
      KernelOp reaches (decision KT12 A1);
    - ``cases``: its conformance cases (``tests/kernels/test_conformance.py``), by name;
    - ``space``: the case whose placement, nothing chosen, its covering points
      (``finn.harness.points.covering`` of its ``kernel.*`` Decisions) and its refused
      side are drawn from; ``refuses``, that side as the Space states it there;
    - ``probes``: points it must refuse beyond that side, each with its codes;
    - ``sweeps``: the numeric sweeps' cases that place it; ``unit``: its unit tests.
    """

    kernel: type[Kernel]
    reference: str
    cases: Mapping[str, Case]
    space: str
    refuses: frozenset[Refused] = frozenset()
    probes: tuple[Probe, ...] = ()
    sweeps: tuple[SweepCases, ...] = ()
    unit: tuple[str, ...] = field(default=())

    @property
    def name(self) -> str:
        return self.kernel.__name__


ORACLE = "the KernelOp {op}'s execute_node, at the op's boundary (tests/kernel_ops/test_parity.py)"
TEST_SIDE = "test-side: its conformance cases' references (no KernelOp reaches it)"


def placed(case: dict[str, Any], choices: Mapping[str, object] = EMPTY, **facts: object) -> Space:
    """A case's kernel, placed as ``unchosen`` places it, on its facts but ``facts``, with
    ``choices`` (by its own decision keys: ``pe``) committed."""
    point = unchosen(**{**case, "facts": {**case.get("facts", {}), **facts}})
    point = configured(point, {f"{KERNEL}.{key}": value for key, value in choices.items()})
    kernel: Space = design_member(point, KERNEL)
    return kernel


def refuses(*found: tuple[str, object, str | tuple[str, ...]]) -> frozenset[Refused]:
    """The refused side as a spec states it: (decision key, case, code or codes)."""
    return frozenset(
        Refused(key, case, frozenset((codes,) if isinstance(codes, str) else codes))
        for key, case, codes in found
    )


__all__ = [
    "Case",
    "EMPTY",
    "KernelSpec",
    "ORACLE",
    "Probe",
    "SweepCases",
    "TEST_SIDE",
    "placed",
    "refuses",
]
