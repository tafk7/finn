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
declaration's where its ``module_order`` puts it (``finn.kernels.base.ModuleOrder``: a
control bus's pins before the ports' and its hold after theirs).

Its sources must provide the module, ``module:<rtl_module>`` (an HLS request its top, of
that stem): a constant tuple is checked when the class is made, a member's in ``build``,
each refused as ``library-module-unprovided``, naming the module and what the sources
provide. ``library_module`` reads a kernel's module back from its accepted netlist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any, ClassVar

from finn.core.space import DefinitionError, Rejected, reject
from finn.kernels.artifacts.contributions import (
    Contribution,
    CopiedSource,
    GeneratedData,
    HlsSource,
    request_stem,
)
from finn.kernels.artifacts.module import Abi, Held, Leaf
from finn.kernels.base import (
    Kernel,
    KernelDeclaration,
    KernelSpace,
    ModuleOrder,
    Product,
    _each,
    _module_name,
    is_member,
)


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
        wanted = f"module:{self.rtl_module}"
        provided = _provided(sources)
        if wanted in provided:
            return None
        return (
            f"{owner}: its sources provide {list(provided)}, not the library module "
            f"{self.rtl_module} ({wanted})"
        )

    def lower(self, owner: type[Kernel], name: str) -> Mapping[str, object]:
        label = f"{owner.__qualname__}.{name}"
        if not issubclass(owner, KernelSpace):
            raise DefinitionError.broken(
                "library-module-owner",
                "LibraryModule",
                f"{label}: a library module is a KernelSpace's",
                declaration=label,
            )
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

    def build(self, kernel: KernelSpace, name: str) -> Leaf | Rejected:
        """The module on ``kernel``, where it is declared as ``name``: its parameters, its
        clocking and each declaration's pins, its sources and data, and what it holds."""
        space_type = type(kernel)
        clocking = kernel.clocking
        parameters = tuple(sorted(self.value(kernel, "parameters").items()))
        items = tuple(space_type._declarations.items())
        enclosing = tuple(item for item in items if item[1].module_order is ModuleOrder.ENCLOSING)
        declared = tuple(item for item in items if item[1].module_order is ModuleOrder.DECLARED)
        found = _each(
            declaration.pins(kernel, each) for each, declaration in (*enclosing, *declared)
        )
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
        return Leaf(
            space_type.id,
            str(space_type.version),
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


def library_module(point: KernelSpace) -> Leaf:
    """The module at the empty label of ``point``'s accepted netlist: the library module a
    kernel binds."""
    found = dict(point.netlist.instances).get("")
    if found is None:
        raise ValueError(f"{type(point).__qualname__} binds no library module")
    return found


__all__ = [
    "LibraryModule",
    "library_module",
]
