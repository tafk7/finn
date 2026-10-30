# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Random composite Space classes, to compare the checks composed per template with the
line's over one table (``test_compose``).

``composite_space(seed)`` draws a Space of three levels: each places up to two children and
a selector of up to two cases (sometimes guarded by whether its optional formal ``b`` is
given, or by a choice), binds each child's formals to a literal, a formal, a choice, a
Decision of its own declared for that binding (sometimes with a ``when``), one Decision it
shares among its children (``shared``) or a sibling's member, or leaves ``b`` out (None), and
sometimes binds a child's formal back to one of its own members (a walk up re-enters it).
Each Space has a derived value ``x`` (explicit or a ``self`` method, sometimes reading a
choice first and returning early where it is false), a Decision ``p`` whose domain reads a
formal, ``x``, a child's ``x`` or a selection, and sometimes a presence test, a test of ``b``
and a view.
Most domains read facts; many read a choice, some through a slot, some through a guard."""

from __future__ import annotations

import random
from typing import Any

from finn.core.space import (
    Decision,
    Inapplicable,
    Param,
    Select,
    Space,
    composite,
    derived,
    divisors_of,
    view,
)
from finn.core.space.errors import DefinitionError


def ident(*, v: int) -> int:
    return v


def add(*, v: int, w: int) -> int:
    return v + w


def ident_if(*, flag: bool, v: int) -> int | Inapplicable:
    """``ident`` where ``flag`` holds, which it reads as an input."""
    return v if flag else Inapplicable()


def add_if(*, flag: bool, v: int, w: int) -> int | Inapplicable:
    """``add`` where ``flag`` holds, which it reads as an input."""
    return v + w if flag else Inapplicable()


def given(*, v: object) -> bool:
    return v is not None


def m_a(self) -> int:  # type: ignore[no-untyped-def]
    return int(self.a)


def m_c0x(self) -> int:  # type: ignore[no-untyped-def]
    return int(self.c0.x)


def m_c1p(self) -> int:  # type: ignore[no-untyped-def]
    return int(self.c1.p)


def m_chx(self) -> int:  # type: ignore[no-untyped-def]
    return int(self.ch.x)


def m_a_if(self) -> int | Inapplicable:  # type: ignore[no-untyped-def]
    if not self.flag:
        return Inapplicable()
    return int(self.a)


def m_c0x_if(self) -> int | Inapplicable:  # type: ignore[no-untyped-def]
    if not self.flag:
        return Inapplicable()
    return int(self.c0.x)


def m_c1p_if(self) -> int | Inapplicable:  # type: ignore[no-untyped-def]
    if not self.flag:
        return Inapplicable()
    return int(self.c1.p)


def m_chx_if(self) -> int | Inapplicable:  # type: ignore[no-untyped-def]
    if not self.flag:
        return Inapplicable()
    return int(self.ch.x)


# Each method of ``x``, and the same reading the choice ``flag`` first.
IF_FLAG = {m_a: m_a_if, m_c0x: m_c0x_if, m_chx: m_chx_if, m_c1p: m_c1p_if}


def m_present(self) -> bool:  # type: ignore[no-untyped-def]
    return bool(self.present(self.c0))


def m_given(self) -> bool:  # type: ignore[no-untyped-def]
    return self.b is not None


def m_flag(self) -> bool:  # type: ignore[no-untyped-def]
    return bool(self.flag)


class _Draw:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.count = 0
        # Whether the last child placed left its formal ``b`` out.
        self.unbound = False

    def maybe(self, chance: float) -> bool:
        return self.rng.random() < chance

    @staticmethod
    def ref(members: dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        current: Any = members[parts[0]]
        for part in parts[1:]:
            current = getattr(current, part)
        return current

    def space(self, level: int) -> type[Space]:
        rng = self.rng
        members: dict[str, Any] = {"a": Param(), "b": Param(default=None)}
        annotations: dict[str, object] = {"a": int, "b": int | None}
        if self.maybe(0.5):
            members["valued"] = derived(v=members["b"])(given)
        if self.maybe(0.6):
            members["flag"] = Decision(domain=(False, True))
            annotations["flag"] = bool
        if self.maybe(0.5):
            members["d"] = Decision(domain=(1, 2, 4))
            annotations["d"] = int
        children: list[str] = []
        if level > 0:
            for child in ("c0", "c1"):
                if self.maybe(0.8):
                    members[child] = self.place(level - 1, members, children, annotations)
                    children.append(child)
                    if self.unbound and self.maybe(0.5):
                        self.bind_back(members[child])
            if self.maybe(0.7):
                guard: Any = None
                if "valued" in members and self.maybe(0.6):
                    guard = members["valued"]
                elif "flag" in members and self.maybe(0.4):
                    guard = members["flag"]
                cases = {
                    case: self.place(level - 1, members, children, annotations, candidate=True)
                    for case in ("u", "v")
                    if self.maybe(0.85)
                }
                if cases:
                    extra: dict[str, Any] = {} if guard is None else {"when": guard}
                    none: dict[str, Any] = {"none": None} if self.maybe(0.3) else {}
                    members["ch"] = Select(**none, **cases, **extra)
        sources = ["a", "a", "b"] + (["d"] if "d" in members and self.maybe(0.3) else [])
        sources += [f"{child}.x" for child in children]
        if "ch" in members and self.maybe(0.5):
            sources.append("ch.x")
        kind = rng.random()
        # Sometimes ``x`` reads the choice ``flag`` first and returns early where it is false.
        flagged = "flag" in members and self.maybe(0.3)
        flag: dict[str, Any] = {"flag": members["flag"]} if flagged else {}
        if kind < 0.5:
            source = self.ref(members, rng.choice(sources))
            members["x"] = derived(**flag, v=source)(ident_if if flagged else ident)
        elif kind < 0.7:
            v, w = rng.sample(sources, 2)
            aliases = {**flag, "v": self.ref(members, v), "w": self.ref(members, w)}
            members["x"] = derived(**aliases)(add_if if flagged else add)
        else:
            method = rng.choice((m_a, m_c0x, m_chx, m_c1p))
            members["x"] = derived(IF_FLAG[method] if flagged else method)
        targets = ["a", "x", "b", *(f"{child}.x" for child in children)]
        if "ch" in members:
            targets.append("ch.x")
        when = members["flag"] if "flag" in members and self.maybe(0.3) else None
        members["p"] = Decision(
            domain=divisors_of(self.ref(members, rng.choice(targets))), when=when
        )
        annotations["p"] = int
        if self.maybe(0.4):
            members["y"] = derived(rng.choice((m_present, m_given, m_flag)))
        if self.maybe(0.3):
            path = rng.choice(["x", "a", *(f"{child}.x" for child in children)])
            members["v"] = view(self.ref(members, path))
        self.count += 1
        return composite(f"Drawn{level}_{self.count}", members, annotations=annotations)

    def place(
        self,
        level: int,
        members: dict[str, Any],
        children: list[str],
        annotations: dict[str, object],
        *,
        candidate: bool = False,
    ) -> Space:
        space_type = self.space(level)
        bindings: dict[str, Any] = {}
        for formal in ("a", "b"):
            form = self.rng.choice(
                ("lit", "lit", "formal", "formal", "formal", "choice", "fresh", "named")
                + ("sibling", "sibling", "none", "none")
            )
            if form == "formal":
                bindings[formal] = members[self.rng.choice(("a", "b"))]
            elif form == "choice" and "d" in members:
                bindings[formal] = members["d"]
            elif form == "fresh":
                fresh: dict[str, Any] = (
                    {"when": members["flag"]} if "flag" in members and self.maybe(0.5) else {}
                )
                name = f"given{len(members)}_{formal}"
                members[name], annotations[name] = Decision(domain=(2, 4), **fresh), int
                bindings[formal] = members[name]
            elif form == "named" and not candidate:
                if "shared" not in members:
                    members["shared"], annotations["shared"] = Decision(domain=(2, 4)), int
                bindings[formal] = members["shared"]
            elif form == "sibling" and children:
                bindings[formal] = getattr(members[self.rng.choice(children)], "x")
            elif not (form == "none" and formal == "b"):
                bindings[formal] = 4
        if not candidate and "flag" in members and self.maybe(0.3):
            bindings["when"] = members["flag"]
        node = space_type(**bindings)
        self.unbound = "b" not in bindings
        return node

    def bind_back(self, node: Any) -> None:
        """Bind ``node``'s formal ``b`` to one of its own members (its own, a case's or a
        child's): the placing body's walk re-enters it."""
        space_type = type(node)
        paths: list[tuple[str, ...]] = [("x",), ("p",), ("a",)]
        if "ch" in vars(space_type):
            paths += [("ch", case, member) for case in ("u", "v") for member in ("x", "p")]
        for child in ("c0", "c1"):
            if child in vars(space_type):
                paths += [(child, "x"), (child, "p")]
        path = self.rng.choice(paths)
        try:
            current: Any = node
            if len(path) == 3:
                current = getattr(getattr(current, path[0])[path[1]], path[2])
            else:
                for part in path:
                    current = getattr(current, part)
        except (AttributeError, KeyError, DefinitionError):
            return  # a case or a member not drawn
        node.b = current


def composite_space(seed: int, depth: int = 2) -> type[Space]:
    """A random composite Space class (see the module docstring)."""
    return _Draw(seed).space(depth)
