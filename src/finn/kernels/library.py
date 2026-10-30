# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Plain library-module values and resolved source/ABI assembly.

A native kernel states ModuleSpec, its parameter/source/data Views and literal
serial physical helper calls. These functions take resolved values only;
no declaration lowering, callback tables, symbolic field naming or manual
read lists are installed on the kernel. library_module reads its actual Leaf
back from the accepted outward netlist.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from finn.core.space import DefinitionError, Rejected, reject
from finn.kernels.artifacts.abi import Clocking, Direction, Pin, Signal, physical_names
from finn.kernels.artifacts.contributions import (
    Contribution,
    CopiedSource,
    GeneratedData,
    HlsSource,
    request_stem,
)
from finn.kernels.artifacts.module import Abi, BuildError, Fragment, Held, Leaf

if TYPE_CHECKING:
    from finn.kernels.base import Kernel

_T = TypeVar("_T")


def _each(results: Iterable[_T | Rejected]) -> list[_T] | Rejected:
    """Each result, or, where any is refused, every refusal's findings together."""
    found = list(results)
    refused = [result for result in found if isinstance(result, Rejected)]
    if refused:
        return Rejected(
            tuple(dict.fromkeys(item for result in refused for item in result.findings))
        )
    return cast("list[_T]", found)


def _module_name(stem: str, contributions: tuple[Contribution, ...]) -> str:
    """``stem``, or the top of the HLS request among ``contributions``, named from it."""
    requests = [item for item in contributions if isinstance(item, HlsSource)]
    if not requests:
        return stem
    function = requests[0].function
    if request_stem(function) != stem:
        raise BuildError(f"{function} is not the top of the HLS module {stem}")
    return function


def _inputs(pin: Pin) -> tuple[str, ...]:
    """The input pins of ``pin``: a loose signal's, or each of a bus's members'."""
    if isinstance(pin, Signal):
        return (pin.name,) if pin.direction is Direction.IN else ()
    return tuple(name for name, direction in pin.member_directions() if direction is Direction.IN)


def _provided(sources: tuple[Contribution, ...]) -> tuple[str, ...]:
    """The symbols ``sources`` provide: a copied source's ``provides``, an HLS request's
    top as ``module:<its stem>``."""
    found: list[str] = []
    for item in sources:
        if isinstance(item, CopiedSource):
            found.extend(item.provides)
        elif isinstance(item, HlsSource):
            stem = request_stem(item.function)
            found.append(f"module:{stem if stem is not None else item.function}")
    return tuple(dict.fromkeys(found))


def _unprovided(stem: str, owner: str, sources: tuple[Contribution, ...]) -> str | None:
    """Why the resolved sources do not provide the module, preserving symbol order."""
    wanted = f"module:{stem}"
    provided = _provided(sources)
    if wanted in provided:
        return None
    return (
        f"{owner}: its sources provide {list(provided)}, not the library module {stem} ({wanted})"
    )


@dataclass(frozen=True)
class ModuleSpec:
    """Plain module identity and optional constant sources, without symbolic fields.

    None means the parent reads its dynamic sources explicitly. A constant tuple,
    including an empty tuple, is checked at definition time by validate().
    Parameters, data and physical products are resolved by ordinary parent source;
    this object cannot query a Space or manufacture its read declarations.
    """

    rtl_module: str
    constant_sources: tuple[Contribution, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.rtl_module) is not str:
            raise TypeError("ModuleSpec module name is a constant string")
        sources = self.constant_sources
        if sources is not None and (
            not isinstance(sources, tuple)
            or any(
                not isinstance(source, (CopiedSource, HlsSource, GeneratedData))
                for source in sources
            )
        ):
            raise TypeError("ModuleSpec sources are constant contribution values")

    def validate(self, owner: str, declaration: str) -> None:
        """Definition-time provision check for constant sources, with authored context."""
        if self.constant_sources is None:
            return
        why = _unprovided(self.rtl_module, owner, self.constant_sources)
        if why is not None:
            raise DefinitionError.broken(
                "library-module-unprovided", "ModuleSpec", why, declaration=declaration
            )


def module_unprovided(
    spec: ModuleSpec, owner: str, sources: tuple[Contribution, ...]
) -> Rejected | None:
    """Runtime provision check, called after successful pins and before data/held reads."""
    why = _unprovided(spec.rtl_module, owner, sources)
    return None if why is None else reject("library-module-unprovided", why)


def assemble_module(
    spec: ModuleSpec,
    kernel_id: str,
    owner: str,
    clocking: Clocking,
    parameters: tuple[tuple[str, int | str], ...],
    pin_parts: tuple[tuple[Pin, ...], ...],
    sources: tuple[Contribution, ...],
    data: GeneratedData | tuple[GeneratedData, ...],
    held_parts: tuple[Held, ...],
    presented_parts: tuple[bool, ...],
) -> Leaf | Rejected:
    """Assemble resolved products into a Leaf, without a receiver or evaluation.

    The parent sorts parameters at their original read step, evaluates pin and
    held contributors serially, and checks source provision before reading data
    and held products. Direct physical refusals are combined by _each there.
    Pin parts retain control-before-ports; held parts retain ports-before-control.
    Presented flags correspond to pin parts, exactly as input accounting requires.
    """
    pins = tuple(pin for part in pin_parts for pin in part)
    named = (*clocking.signals(), *pins)
    repeated = sorted(
        {
            name
            for names in ([pin.name for pin in named], physical_names(named))
            for name in names
            if names.count(name) > 1
        }
    )
    if repeated:
        return reject(
            "port-pins-collide",
            f"{owner}: its ports and clocks name the pins {repeated} more than once; "
            "a kernel states its own names for one",
        )
    contributions = (*sources, *((data,) if isinstance(data, GeneratedData) else data))
    parts = (Held(clocking.held()), *held_parts)
    holding = {pin for part in parts for pin, _ in part.inputs}
    loose = [
        pin
        for presented, part in zip(presented_parts, pin_parts, strict=True)
        if not presented
        for item in part
        for pin in _inputs(item)
        if pin not in holding
    ]
    if loose:
        return reject(
            "kernel-pins",
            f"{owner}: the inputs {loose} are neither held nor presented",
        )
    return Leaf(
        kernel_id,
        _module_name(spec.rtl_module, contributions),
        parameters,
        Abi(
            named,
            tuple((key, str(value)) for key, value in parameters),
            clocking.alignments(),
        ),
        tuple(item for item in contributions if isinstance(item, (CopiedSource, HlsSource))),
        tuple(item for item in contributions if isinstance(item, GeneratedData)),
        Held(
            tuple(pin for part in parts for pin in part.inputs),
            tuple(pin for part in parts for pin in part.unused),
        ),
    )


def leaf_at_root(netlist: Fragment, owner: str) -> Leaf:
    """Extract a resolved module at its empty label; no receiver enters this helper."""
    found = dict(netlist.instances).get("")
    if found is None:
        raise ValueError(f"{owner} binds no library module")
    return found


def library_module(point: Kernel) -> Leaf:
    """The module at the empty label of ``point``'s accepted netlist: the library module a
    kernel binds."""
    found = dict(point.netlist.instances).get("")
    if found is None:
        raise ValueError(f"{type(point).__qualname__} binds no library module")
    return found


__all__ = [
    "ModuleSpec",
    "assemble_module",
    "library_module",
    "leaf_at_root",
    "module_unprovided",
]
