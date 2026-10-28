# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The library module a kernel binds: a declaration of its kernel, and the module it builds.

A kernel binding one module of the library (FinnLib's RTL, or an HLS top FINN requests)
declares it in its class body, after the members it names::

    module = LibraryModule("fifo", parameters=parameters, sources=(CopiedSource(...),))

``rtl_module`` is the module's name, a constant. ``parameters`` is the kernel's member
stating the module's parameters (a view: nothing in a rule reads them). ``sources`` is the
tuple of sources that provide the module, constant (copied sources), or the kernel's
member where they depend on its point (an HLS kernel's one source, its build request).
``data`` is the kernel's member stating the data its module reads (a memory's image, a
``GeneratedData``, or a tuple of them), or nothing. Each member it names is resolved to
its name once, when the class is made (``KernelDeclaration.references``), and read by that
name: never a dynamic read.

It lowers to no member of the kernel. What it gives is its product, the module
(``LibraryModule.build``, a ``Leaf``), which the kernel's ``netlist`` places at the empty
label: its name (an HLS top's, for an HLS kernel), its parameters, its pins (the
kernel's clocking, then each declaration's pins), its sources and data, and what it holds
(the doubled clock while unused, then what each declaration holds idle), each
declaration's where its ``module_order`` puts it (``finn.kernels.declaration.ModuleOrder``: a
control bus's pins before the ports' and its hold after theirs). Each input of the module
is a port's, a presented control bus's (``KernelDeclaration.presents``), or held: any
other refuses the module (``kernel-pins``, naming the inputs).

Its sources must provide the module, ``module:<rtl_module>`` (an HLS request its top, of
that stem): a constant tuple is checked when the class is made, a member's in ``build``,
each refused as ``library-module-unprovided``, naming the module and what the sources
provide.

``library_module`` reads a kernel's module back from its accepted netlist, the module at
its empty label: every reader of a kernel's module takes it, inside a kernel's view (a
channel's FIFO stage, an adapter's stages) as outside.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from finn.core.space import DefinitionError, Rejected, reject
from finn.kernels.artifacts.abi import Clocking, Direction, Pin, Signal
from finn.kernels.artifacts.contributions import (
    Contribution,
    CopiedSource,
    GeneratedData,
    HlsSource,
    request_stem,
)
from finn.kernels.artifacts.module import Abi, BuildError, Fragment, Held, Leaf
from finn.kernels.declaration import KernelDeclaration, ModuleOrder, Product, is_member

if TYPE_CHECKING:
    from finn.kernels.base import Kernel, UntimedKernel

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
            (*clocking.signals(), *pins),
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


@dataclass(frozen=True, eq=False)
class LibraryModule(KernelDeclaration):
    """The one module of the library a kernel binds: its name, and the kernel's members
    stating its parameters, its sources and its data. The module docstring says what it
    builds."""

    rtl_module: str
    _: KW_ONLY
    parameters: Any
    sources: Any
    data: Any = ()

    KIND: ClassVar[str] = "library-module"

    def label(self) -> str:
        return self.rtl_module

    def unprovided(self, owner: str, sources: tuple[Contribution, ...]) -> str | None:
        """Why ``sources`` do not provide the module, for ``owner``; None when they do."""
        return _unprovided(self.rtl_module, owner, sources)

    def lower(self, owner: type[UntimedKernel], name: str) -> Mapping[str, object]:
        label = f"{owner.__qualname__}.{name}"
        self.references()  # each member it names, named
        if not is_member(self.sources):
            why = self.unprovided(owner.__qualname__, tuple(self.sources))
            if why is not None:
                raise DefinitionError.broken(
                    "library-module-unprovided", "LibraryModule", why, declaration=label
                )
        return {}

    def reads(self, name: str, product: Product) -> tuple[str, ...]:
        if product != "module":
            return ()
        return (*self.paths("parameters", "sources", "data"), "clocking")

    def build(self, kernel: UntimedKernel, name: str) -> Leaf | Rejected:
        """The module on ``kernel``, where it is declared as ``name``: its parameters, its
        clocking and each declaration's pins, its sources and data, and what it holds."""
        space_type = type(kernel)
        clocking = kernel.clocking
        parameters = tuple(sorted(self.value(kernel, "parameters").items()))
        items = tuple(space_type._declarations.items())
        enclosing = tuple(item for item in items if item[1].module_order is ModuleOrder.ENCLOSING)
        declared = tuple(item for item in items if item[1].module_order is ModuleOrder.DECLARED)
        ordered = (*enclosing, *declared)
        found = _each(declaration.pins(kernel, each) for each, declaration in ordered)
        if isinstance(found, Rejected):
            return found
        pins = tuple(pin for part in found for pin in part)
        sources: tuple[Contribution, ...] = self.value(kernel, "sources")
        if is_member(self.sources):
            why = self.unprovided(space_type.__qualname__, sources)
            if why is not None:
                return reject("library-module-unprovided", why)
        data = self.value(kernel, "data")
        contributions = (*sources, *((data,) if isinstance(data, GeneratedData) else data))
        held = _each(
            declaration.held(kernel, each) for each, declaration in (*declared, *enclosing)
        )
        if isinstance(held, Rejected):
            return held
        parts = (Held(clocking.held()), *held)
        holding = {pin for part in parts for pin, _ in part.inputs}
        loose = [
            pin
            for (each, declaration), part in zip(ordered, found, strict=True)
            if not declaration.presents(kernel, each)
            for item in part
            for pin in _inputs(item)
            if pin not in holding
        ]
        if loose:
            return reject(
                "kernel-pins",
                f"{space_type.__qualname__}: the inputs {loose} are neither held nor presented",
            )
        return Leaf(
            space_type.id,
            _module_name(self.rtl_module, contributions),
            parameters,
            Abi(
                (*clocking.signals(), *pins),
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
    "LibraryModule",
    "ModuleSpec",
    "assemble_module",
    "library_module",
    "leaf_at_root",
    "module_unprovided",
]
