# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Suppliers of a formal whose annotation does not resolve yet.

A formal annotated with a type its module defines further down leaves its
annotation unresolved while the class bodies above it run. A supplier of such a
formal is checked when linking, where every annotation resolves: a forward or a
member reference waits; a literal still needs its type at the call, and a node
is no value. An annotation that resolves to a Space class is refused there: a
formal takes a value.

Each toy is a module written to ``tmp_path`` and imported under its own name.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
from support.refusals import refused

from finn.core.space import Decision, DefinitionError, Space, design_space, inspection

TOY = """
from __future__ import annotations
from dataclasses import dataclass
from finn.core.space import Decision, Param, Select, Space, derived

class Port(Space):
    rate: {annotation} = Param()

    @derived
    def total(self) -> int:
        return self.rate.per_head * self.rate.heads

class Kernel(Space):
    rate: {annotation} = Param()
    port = Port(rate={supplier})
{extra}
@dataclass(frozen=True)
class Rate:
    per_head: int
    heads: int

class Chan(Space):
    width: int = Param()
"""


def toy(tmp_path: Path, name: str, **variant: str) -> ModuleType:
    """The toy with ``variant`` applied, imported as module ``name``."""
    values = {"annotation": "Rate", "supplier": "rate", "extra": "", **variant}
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(TOY.format(**values)))
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(name)  # kept in sys.modules: its annotations resolve there
    finally:
        sys.path.remove(str(tmp_path))


def test_a_forward_into_a_pending_formal_links(tmp_path: Path) -> None:
    module = toy(tmp_path, "toy_forward")
    point = design_space(module.Kernel(rate=module.Rate(10, 3)))
    assert point.port.total == 30


@pytest.mark.parametrize(
    ("variant", "code", "message"),
    [
        (
            {"extra": "    extra = Port(rate=3)"},
            "literal-annotation-unresolved",
            "the annotation of rate does not resolve yet, so a literal cannot be recognized "
            "here; bind a member, or assign the value where the Space class is defined",
        ),
        (
            {"extra": "    extra = Port(rate=Port(rate=rate))"},
            "node-not-value",
            "a node is not a value; bind one of its members",
        ),
        (
            {"extra": "    extra = Port(rate=Select(p=Port))"},
            "node-not-value",
            "a node is not a value; bind one of its members",
        ),
    ],
)
def test_a_pending_supplier_is_refused_at_the_call(
    tmp_path: Path, variant: dict[str, str], code: str, message: str
) -> None:
    with refused(code, error=DefinitionError) as caught:
        toy(tmp_path, f"toy_refused_{abs(hash(message)) % 10_000}", **variant)
    assert message in str(caught.error)


def test_a_pending_annotation_of_a_space_class_is_refused_when_it_resolves(
    tmp_path: Path,
) -> None:
    module = toy(tmp_path, "toy_space_class", annotation="Chan")
    with refused(
        "formal-takes-node",
        error=DefinitionError,
        message=r"a formal takes a value, not a Chan node",
    ):
        design_space(module.Kernel(rate=None))


def test_a_typo_in_a_pending_annotation_is_refused_when_linking(tmp_path: Path) -> None:
    module = toy(tmp_path, "toy_typo", annotation="Rat")
    with refused(
        "annotation-unresolved",
        error=DefinitionError,
        message="cannot resolve the annotation 'Rat'",
    ):
        inspection.decisions(module.Kernel)  # compiles the Space class


def test_only_an_unresolved_annotation_is_deferred() -> None:
    """Any other error of a formal or Decision is reported where its class is defined."""
    with refused(
        "annotation-missing", error=DefinitionError, message="Bare.choice .*annotate the decision"
    ):

        class Bare(Space):
            choice = Decision(values=(1, 2))
