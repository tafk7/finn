# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The declarations of a kernel: what one states, how it is lowered, what its answers read.

A **declaration of a kernel** (``KernelDeclaration``) is a plain frozen value in its class
body, not a Space: a stream port (``finn.kernels.port``), a port a kernel presents for one
of its members (``finn.kernels.port.served``), the library module it binds
(``finn.kernels.library.LibraryModule``), or its control bus
(other physical declarations in still-unmigrated module fixtures). When the class is made, the kernel base
(``finn.kernels.base.UntimedKernel``) lowers each into ordinary members of the kernel, named
``<name>_<what>``: derived values, constraints and views (``KernelDeclaration.lower``),
each reading the kernel's members by the names its declaration gives, never dynamically.

Its arguments are its dataclass fields, each a constant or a member of its kernel
(``tensor=x_tensor``: ``is_member``), resolved to the member's name once
(``KernelDeclaration.references``) and read by that name (``value``, ``read``).

What a kernel's answers read of a declaration (``Product``: its module, its pins, what it
holds idle, its read of a tensor, the beats it carries a frame) it states with the member
paths each reads (``KernelDeclaration.reads``), and the answers that span them read those
by name, their reads declared per class when the class is made (``declared_reads``).
Where its pins and what it holds go in the module, beside the others', it states too
(``ModuleOrder``): a port's in the order it is declared, a control bus's pins before every
port's and its hold after theirs.

This module sits below the kernel base and every kind of declaration: the base, the ports,
the library module and the control bus each import it, and it imports none of them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeVar, cast

from finn.core.space import DefinitionError, Rejected, Space
from finn.core.space.declarations import Declaration, Required
from finn.dataflow.schedule import Access
from finn.kernels.artifacts.abi import Pin
from finn.kernels.artifacts.module import Held

if TYPE_CHECKING:
    from finn.kernels.base import UntimedKernel


Product = Literal["module", "pins", "held", "access", "beats"]
"""What a kernel's products read of a declaration: the module it builds (``module``, a
library module's: ``finn.kernels.library.LibraryModule.build``), its pins of a module
(``KernelDeclaration.pins``), what it holds idle (``held``), its read of a tensor
(``access``) and the beats it carries a frame (``beats``)."""


def is_member(value: object) -> bool:
    """Whether ``value``, an argument of a declaration, is a declaration of the engine (a
    member of the kernel, or one a subclass states: ``required()``), not a constant: a
    constant of an engine type (a domain) is a constant."""
    return isinstance(value, (Declaration, Required))


_F = TypeVar("_F", bound=Callable[..., object])


def declared_reads(*paths: str) -> Callable[[_F], _F]:
    """Declare what a method reads by names its source computes (``__reads__``: dotted
    paths from ``self``), which stand in for its source in the reads the engine infers; the
    engine holds the method to them when it runs."""

    def declare(function: _F) -> _F:
        function.__reads__ = paths  # type: ignore[attr-defined]
        return function

    return declare


class ModuleOrder(Enum):
    """Where a declaration's pins and what it holds go in its kernel's module, beside the
    other declarations'. The module's pins are its clocking's, then each ``ENCLOSING``
    declaration's, then each ``DECLARED`` one's; what it holds is its clocking's, then each
    ``DECLARED`` declaration's, then each ``ENCLOSING`` one's; each kind in the order the
    declarations are declared. Both orders reach the module's text and its identity."""

    DECLARED = "declared"
    """In the order the declarations are declared: a port's."""

    ENCLOSING = "enclosing"
    """Its pins before every ``DECLARED`` declaration's, its hold after theirs: a control
    bus's (other physical declarations in still-unmigrated module fixtures)."""


class KernelDeclaration:
    """A declaration of a kernel: a plain frozen value in its class body, lowered into
    ordinary members of the kernel when the class is made (``lower``). The module
    docstring says what one states.

    A kind of declaration states its lowering beside its declaration, and what a kernel's
    products read of it: by default nothing. Each product is a function of the kernel's
    design point by the names its lowering fixes, and ``reads`` names the member paths
    each reads, which the kernel declares as its products' reads.

    Its arguments are its dataclass fields, each a constant or a member of its kernel
    (``tensor=x_tensor``), which its members and products read by the name the lowering
    fixes (``references``, ``value``): never a dynamic read."""

    # The kind of declaration its definition errors name (``port-argument-unnamed``).
    KIND: ClassVar[str] = "declaration"
    # Where its pins and its hold go in its kernel's module (``ModuleOrder``).
    module_order: ClassVar[ModuleOrder] = ModuleOrder.DECLARED
    # Whether it is a stream port of its kernel's module (``finn.kernels.port.AxiStreamPort``):
    # a kernel binding a module states one, and a kernel stating one binds a module.
    module_port: ClassVar[bool] = False

    def label(self) -> str:
        """What its definition errors call it."""
        return type(self).__name__

    # -- its arguments, read from its kernel ------------------------------------------------

    def references(self) -> dict[str, str]:
        """Each argument given as a member of its kernel, by field: the member's name."""
        found: dict[str, str] = {}
        for item in fields(cast(Any, self)):
            value = getattr(self, item.name)
            if is_member(value):
                name = getattr(value, "name", None)
                if not isinstance(name, str):
                    raise DefinitionError.broken(
                        f"{self.KIND}-argument-unnamed",
                        type(self).__name__,
                        f"{self.label()}: its {item.name} is no member of its kernel; name it "
                        "in the class body and give that name",
                        declaration=self.label(),
                    )
                found[item.name] = name
        return found

    def paths(self, *names: str) -> tuple[str, ...]:
        """The members of its kernel the arguments ``names`` read."""
        references = self.references()
        return tuple(references[name] for name in names if name in references)

    def value(self, kernel: Space, name: str) -> Any:
        """Argument ``name``: its constant, or its member read from ``kernel``."""
        given = getattr(self, name)
        if is_member(given):
            return getattr(kernel, self.references()[name])
        return given

    def read(self, kernel: Space, *names: str) -> Self:
        """The declaration with its arguments ``names`` read from ``kernel``: what a view
        takes."""
        found: Self = replace(cast(Any, self), **{name: self.value(kernel, name) for name in names})
        return found

    # -- what it lowers to, and what a kernel's products read of it -------------------------

    def lower(self, owner: type[UntimedKernel], name: str) -> Mapping[str, object]:
        """The members it adds to ``owner``, where it is declared as ``name``, by name."""
        raise NotImplementedError

    def reads(self, name: str, product: Product) -> tuple[str, ...]:
        """The member paths ``product`` reads of it, declared as ``name``."""
        return ()

    def pins(self, kernel: UntimedKernel, name: str) -> tuple[Pin, ...] | Rejected:
        """Its pins of the module, in order."""
        return ()

    def held(self, kernel: UntimedKernel, name: str) -> Held | Rejected:
        """The pins of the module it holds."""
        return Held()

    def presents(self, kernel: UntimedKernel, name: str) -> bool:
        """Whether its pins are presented (a port's, a presented control bus's), read with
        what it holds (``held``): an input of the module that is neither presented nor held
        refuses the module (``kernel-pins``)."""
        return False

    def access(self, kernel: UntimedKernel, name: str) -> Access | None:
        """Its read of a tensor, from which the kernel binds its indices' extents; None for
        none."""
        return None

    def beats(self, kernel: UntimedKernel, name: str) -> int | None:
        """The beats it carries a frame; None for none."""
        return None


def declarations(
    cls: type[UntimedKernel], below: type[UntimedKernel]
) -> dict[str, KernelDeclaration]:
    """The declarations of ``cls`` and its bases down to, not including, ``below``, by name,
    in the order they are declared (a base's first)."""
    found: dict[str, KernelDeclaration] = {}
    for base in reversed(cls.__mro__[: cls.__mro__.index(below)]):
        for name, value in vars(base).items():
            if isinstance(value, KernelDeclaration):
                found[name] = value
            elif name in found:
                del found[name]  # a member of the same name in a subclass replaces it
    return found


__all__ = [
    "KernelDeclaration",
    "ModuleOrder",
    "Product",
    "declarations",
    "declared_reads",
    "is_member",
]
