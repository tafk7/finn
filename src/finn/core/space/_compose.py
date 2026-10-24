# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What crosses a slot, checked per class and composed across the slots.

A Space class is compiled once (``_template``: its compiled form, ``Compiled``) and placed
as scopes (``_scopes``). Two checks read across the slots a class places, and each is
composed here from what one compiled form says of itself, never by copying a placement. A
walk over a compiled form reads its records (members, and the records it writes for the
Spaces it places); an edge into a placed Space (``ir.Placed``) is followed there, what it
reaches out of that Space lifted back through the slot (``lift``).

- **Cycles** (``order``). A class's dependency order runs over its records and the edges
  out of it, with an edge from each edge out to what its target reaches in the placed
  Space: the formals it reaches by structural edges (each followed to the placing class's
  binding: its record, or the guard the scope is placed under for a literal or a default)
  and the guard the child is placed under, through any depth (``exits``, on the child's
  compiled form). A cycle through a child is found once, in the placing class, and named
  by keys relative to it: its own records, the members its edges name, and the members of
  the Spaces it places on the way, as the one table of every placement named them (a
  cycle at two levels sharing members: the inner class's is found first, as that class is
  compiled first). A view's edge to its Space's validity is not in the order: views are
  read out (below), so it closes no cycle, and a read in is ``view-read-in``, not a cycle
  through the validity. Edges among views stay: views reading each other in a ring are a
  cycle.
- **Views read out** (``check_views``): only a view reads a view. A constraint, a derived
  value, a guard (a member's or a placed Space's ``when``), a domain (its inputs) and a
  binding of a placed Space's formal read facts, choices and derived values; one reading a
  view is ``view-read-in``, named with the view. What passes a value on (an alias, a
  selection, a ``Members``, a computation written for a member) reads what it passes, so a
  ``Members`` whose entries are views is read by views alone. A view reads anything, and
  so do the computations written for it and its own guard. Per class, its own records'
  reads (references, and a method's inferred reads) and each placed Space's, once; for a
  root, the first found, named relative to it. It is checked before the domain rule: a
  domain reading a view refuses as ``view-read-in`` whatever the view reads.
- **The domain rule** (``check_domains``): a domain's membership reads facts only. Per
  class, each domain's walk follows what its membership inputs may read (arguments,
  alternatives, a view's validity and a validity's pieces, outputs, selectors, guards and
  a method's inferred reads), skipping the guards that decide whether the Decision exists;
  it stops at a choice (``domain-reads-choice``), and at the class's own formals and the
  guard it is placed under, which it reports as reached. A child's are followed into the
  placing class's bindings, with the guards above the slot as existence guards of the
  child's Decision too, and so up to the root. A class records, for its own formals, which
  are reached by any domain in its subtree (``_Entry``), once.

Each walk is that of the design space the class is placed in: the composition is exact.
Where a walk back up re-enters the subtree holding the Decision it started from, its
existence guards inside that subtree are known only per Decision: the walk is done once
with none of them and, only where it re-entered, again per Decision (``_Approx``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from finn.core.reporting.findings import Finding, FindingKind

from ._ordering import cyclic, dependency_order
from ._reads import _Dynamic, _Missing, _Ref, _TemplateInference
from ._table import Binding, Table, member_key
from .errors import DefinitionError
from .ir import Placed, Placing, Selected, forwards

if TYPE_CHECKING:
    from ._template import Compiled
    from .ir import Edge, Member, Node, NodeRef

_OWNER = "space.compose"


# -- slots ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Slot:
    """A Space a class places: its class's compiled form, the bindings of its formals (a
    reference's ``node``: the record the placing class writes for it) and what it is placed
    under in the placing class."""

    compiled: Compiled
    bindings: Mapping[str, Binding]
    guard: int | Placing


def _cache(compiled: Compiled, name: str) -> dict[Any, Any]:
    return cast(dict[Any, Any], compiled.composed.setdefault(name, {}))


def slot(compiled: Compiled, index: int) -> _Slot:
    """Slot ``index`` of ``compiled``, as the walks read it."""
    cache = _cache(compiled, "slot")
    found = cache.get(index)
    if found is None:
        held = compiled.slots[index]
        found = cache[index] = _Slot(
            held.compiled, held.bindings, cast("int | Placing", compiled.ref(held.guard))
        )
    return cast(_Slot, found)


def _table_slot(table: Table, draft: int) -> _Slot:
    """A slot of a class being compiled (its bindings and guard references of the class)."""
    held = table.placing(draft)
    return _Slot(held.compiled, held.bindings, cast("int | Placing", table.drafts[draft].guard))


def _table_descent(table: Table, edge: Placed | Selected) -> tuple[_Slot, tuple[int, ...], int]:
    """An edge out of a class being compiled, as a walk reads it: its slot, the slots below
    it to the class holding its target, and the target's record there."""
    held = _table_slot(table, table.slot_drafts[edge.path[0]])
    steps = edge.path[1:]
    if isinstance(edge, Selected):
        reached = held.compiled.reached(steps)
        return held, steps, reached.own_selection[(edge.choice, edge.member)]
    return held, steps, edge.member


# -- a class's records -------------------------------------------------------------------------

# A reference out of a class: a formal's (``formal``, name), or the guard it is placed under
# (``guard``,).
Exit: TypeAlias = tuple[str, ...]
# What a walk reads: a record of the class walked, the guard it is placed under, a member of
# a Space it places (walked there), or a reference out of it.
Ref: TypeAlias = "int | Placing | Placed | Exit"


def pieces(compiled: Compiled) -> list[Ref]:
    """A validity's pieces: its constraints and Decisions, then each placed Space's
    validity. A formal it lists stands for nothing here (no binding is a Decision of its
    own)."""
    own: list[Ref] = []
    placed: list[Ref] = []
    for edge in compiled.records[compiled.validity].constraints:
        piece = compiled.ref(edge)
        if isinstance(piece, Placed):
            placed.append(piece)
        elif cast(int, piece) not in compiled.formal_names:
            own.append(piece)
    return [*own, *placed]


@dataclass(frozen=True, slots=True)
class _Fields:
    """A node's references, by role."""

    arguments: tuple[Ref, ...]
    domain_arguments: tuple[Ref, ...]
    guard: Ref | None
    output: Ref | None
    selector: Ref | None
    constraints: tuple[Ref, ...]
    alternatives: tuple[Ref, ...]

    def dependencies(self) -> list[Ref]:
        """In ``Node.dependencies``' order."""
        found: list[Ref] = [*self.arguments, *self.domain_arguments]
        found.extend(
            single for single in (self.guard, self.output, self.selector) if single is not None
        )
        found.extend(self.constraints)
        found.extend(self.alternatives)
        return list(dict.fromkeys(found))

    def following(self) -> list[Ref]:
        """In the domain rule's order: arguments, alternatives, constraints, output,
        selector, guard."""
        found: list[Ref] = [*self.arguments, *self.alternatives, *self.constraints]
        found.extend(
            single for single in (self.output, self.selector, self.guard) if single is not None
        )
        return found


def fields(compiled: Compiled, record: int) -> _Fields:
    """A record's references, as a walk reads them (``Compiled.ref``)."""
    spec = compiled.records[record]
    ref = compiled.ref
    alternatives = tuple(ref(target) for _, target in spec.alternatives)
    constraints: tuple[Ref, ...] = tuple(ref(piece) for piece in spec.constraints)
    if record == compiled.validity:
        constraints = tuple(pieces(compiled))
    return _Fields(
        tuple(ref(edge) for _, edge in spec.arguments),
        tuple(ref(edge) for _, edge in spec.domain_arguments),
        None if spec.guard is None else ref(spec.guard),
        None if spec.output is None else ref(spec.output),
        None if spec.selector is None else ref(spec.selector),
        constraints,
        alternatives,
    )


def lift(slot_: _Slot, exits: Iterable[Exit]) -> Iterator[int | Placing]:
    """What a placed Space's references out stand for in its placing class: a formal, its
    binding's record (or, for a literal or a default, the guard the scope is placed under: a
    constant's only edge); the guard it is placed under."""
    for exit in exits:
        if exit[0] == "formal":
            binding = slot_.bindings[exit[1]]
            yield slot_.guard if binding.node is None else binding.node
        else:  # guard
            yield slot_.guard


def _classify(compiled: Compiled, ref: Ref) -> Exit | None:
    """``ref``'s reference out of ``compiled``'s class, if it is one."""
    if isinstance(ref, tuple):
        return ref
    if isinstance(ref, Placing):
        return ("guard",)
    formal = compiled.formal_names.get(ref) if isinstance(ref, int) else None
    if formal is not None:
        return ("formal", formal)
    return None


# -- cycles ------------------------------------------------------------------------------------


def exits(compiled: Compiled, ref: int | Placed) -> frozenset[Exit]:
    """What ``ref`` reaches out of ``compiled``'s class by structural edges (its
    dependencies, through any depth of the Spaces it places). A class's graph is acyclic
    once linked, so each record's are found once."""
    memo = cast(dict[object, frozenset[Exit]], _cache(compiled, "exits"))
    if ref in memo:
        return memo[ref]
    stack: list[tuple[int | Placed, Iterator[Ref]]] = [(ref, iter(_successors(compiled, ref)))]
    open_: set[object] = {ref}
    found: dict[object, set[Exit]] = {ref: set()}
    while stack:
        current, successors = stack[-1]
        following = next(successors, None)
        if following is None:
            stack.pop()
            open_.discard(current)
            memo[current] = frozenset(found.pop(current))
            if stack:
                found[stack[-1][0]] |= memo[current]
            continue
        out = _classify(compiled, following)
        if out is not None:
            found[current].add(out)
            continue
        following = cast("int | Placed", following)
        if following in memo:
            found[current] |= memo[following]
        elif following not in open_:
            open_.add(following)
            found[following] = set()
            stack.append((following, iter(_successors(compiled, following))))
    return memo[ref]


def _successors(compiled: Compiled, ref: int | Placed) -> list[Ref]:
    """A record's dependencies in the static order (a view's without its Space's validity:
    ``_ordered``); a placed Space's member's, what it reaches there, lifted into this
    class."""
    if isinstance(ref, Placed):
        return list(_through(compiled, ref.path, ref.member))
    held = fields(compiled, ref)
    if compiled.records[ref].kind == "view":
        return [each for each in held.dependencies() if each not in held.constraints]
    return held.dependencies()


def _ordered(node: Node) -> tuple[NodeRef, ...]:
    """A node's dependencies in the static order. A view's edge to its Space's validity is
    left out: a view is read out, so nothing its validity reads may read it (``view-read-in``,
    checked per root) and the edge closes no cycle the order must find; a read in reaches
    that check rather than a cycle through the validity. Evaluation keeps the edge (a view
    is refused with its Space)."""
    if node.kind != "view":
        return node.dependencies
    return tuple(ref for ref in node.dependencies if ref not in node.constraints)


def _reaching(compiled: Compiled, ref: Ref) -> frozenset[Exit]:
    """What a reference of ``compiled``'s class reaches out of it: itself, if it is one."""
    out = _classify(compiled, ref)
    if out is not None:
        return frozenset((out,))
    return exits(compiled, cast("int | Placed", ref))


def _through(
    compiled: Compiled, steps: Sequence[int], target: int | Placing
) -> list[int | Placing]:
    """What ``target`` (a record of the class ``steps`` reach) reaches out of it, followed up
    to ``compiled``'s class."""
    return _up(slot(compiled, steps[0]), steps[1:], target)


def _up(held: _Slot, steps: Sequence[int], target: int | Placing) -> list[int | Placing]:
    """What ``target`` (a record of the class ``steps`` reach from ``held``'s) reaches out of
    it, followed up through ``held`` into its placing class."""
    child = held.compiled
    reached: set[Exit] = set()
    if steps:
        for ref in _through(child, steps, target):
            reached |= _reaching(child, ref)
    else:
        reached |= _reaching(child, target)
    return list(lift(held, sorted(reached)))


def order(table: Table) -> tuple[int, ...]:
    """The dependency order of a class being compiled: its records, and each edge out of it
    (``Table.edges``) with its edges to what its target reaches out of the placed Space
    (``exits``, on the placed classes' compiled forms), each in the order it was made; the
    records' order is returned. A cycle is ``cyclic-dependency``, named by keys relative to
    the class: its own records, the members its edges name, and the members of the Spaces
    it places on the way (``_named``)."""
    vertices = table.vertices
    position = {vertex: index for index, vertex in enumerate(vertices)}
    dependencies: list[tuple[int, ...]] = []
    names: list[str] = []
    for vertex in vertices:
        found: list[Any]
        if isinstance(vertex, int):
            node = table.nodes[vertex]
            found = list(_ordered(node))
            names.append(node.key)
        else:
            held, steps, target = _table_descent(table, vertex)
            found = list(_up(held, steps, target))
            names.append(table.edge_name(vertex))
        # The guard the class is placed under is no vertex: it reads nothing of the class.
        dependencies.append(
            tuple(dict.fromkeys(position[ref] for ref in found if not isinstance(ref, Placing)))
        )
    try:
        ordered = dependency_order(tuple(dependencies), tuple(names))
    except DefinitionError as error:
        index = {name: at for at, name in enumerate(names)}
        cycles: list[tuple[str, ...]] = []
        for finding in error.findings:
            members = cast(tuple[str, ...], dict(finding.details)["members"])
            at = {index[member] for member in members}
            named = set(members)
            for member in at:
                vertex = vertices[member]
                if not isinstance(vertex, int):
                    held, steps, target = _table_descent(table, vertex)
                    named |= _named(
                        held,
                        steps,
                        target,
                        table.drafts[table.slot_drafts[vertex.path[0]]].name,
                        lambda ref: isinstance(ref, int) and position.get(ref) in at,
                    )
            cycles.append(tuple(sorted(named)))
        raise cyclic(cycles) from None
    return tuple(cast(int, vertices[at]) for at in ordered if isinstance(vertices[at], int))


def _named(
    held: _Slot,
    steps: Sequence[int],
    target: int | Placing,
    name: str,
    wanted: Callable[[Ref], bool],
) -> set[str]:
    """The keys (under ``name``, the placed scope's) of the members on the ways from
    ``target`` (a record of the class ``steps`` reach from ``held``'s) back to a reference
    of the placing class that ``wanted`` accepts: what a cycle through an edge out of the
    class passes through inside the Spaces placed, as the one table of every placement named it."""
    child = held.compiled

    def reaches(ref: Ref) -> bool:
        return any(wanted(up) for exit in _reaching(child, ref) for up in lift(held, (exit,)))

    found: set[str] = set()
    starts: list[Ref] = []
    if steps:
        below = slot(child, steps[0])
        found |= _named(
            below, steps[1:], target, member_key(name, child.slots[steps[0]].name), reaches
        )
        starts = [ref for ref in _up(below, steps[1:], target) if reaches(ref)]
    elif reaches(target):
        starts = [target]
    formals: set[str] = set()
    seen: set[object] = set(starts)
    pending = list(starts)
    while pending:
        current = pending.pop()
        if _classify(child, current) is not None:
            continue
        current = cast("int | Placed", current)
        if isinstance(current, Placed):
            found |= _named(
                slot(child, current.path[0]),
                current.path[1:],
                current.member,
                member_key(name, child.slots[current.path[0]].name),
                reaches,
            )
        else:
            found.add(member_key(name, child.records[current].key))
        for ref in _successors(child, current):
            out = _classify(child, ref)
            if out is not None:
                if out[0] == "formal" and reaches(out):
                    formals.add(out[1])
                continue
            if ref not in seen and reaches(ref):
                seen.add(ref)
                pending.append(ref)
    # A formal bound to a literal or a default is a constant node of the placed scope.
    found.update(
        member_key(name, formal) for formal in formals if held.bindings[formal].node is None
    )
    return found


# -- the domain rule ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Refusal:
    """A domain that reads a choice (``choice``), or a method whose reads its source does
    not fix (``dynamic``): its Decision's key (``owner``) and the way (``route``)."""

    owner: str | None
    choice: str | None
    dynamic: str | None
    route: tuple[str, ...]

    def under(self, name: str) -> _Refusal:
        return _Refusal(
            None if self.owner is None else member_key(name, self.owner),
            None if self.choice is None else member_key(name, self.choice),
            None if self.dynamic is None else member_key(name, self.dynamic),
            tuple(member_key(name, step) for step in self.route),
        )

    def at(self, owner: str) -> _Refusal:
        """The refusal of the Decision ``owner``, whose walk found it."""
        return _Refusal(owner, self.choice, self.dynamic, self.route)

    def error(self) -> DefinitionError:
        owner = cast(str, self.owner)
        if self.dynamic is not None:
            detail = (
                f"{owner}: its domain's membership reads {self.dynamic}, whose reads its "
                "source does not fix: declare them with __reads__ (a domain reads facts only)"
            )
        else:
            steps = list(reversed(dict.fromkeys(reversed(self.route))))
            detail = (
                f"{owner}: its domain's membership reads the choice {self.choice} "
                f"({' -> '.join(steps)}); a domain reads facts only: state the relation as a "
                "constraint of the Space holding both"
            )
        return DefinitionError.broken("domain-reads-choice", owner, detail, declaration=owner)


# Where the Decision a walk started from is, from the class walked: the slots down to its
# class, then its guard there (``_Path``).
_Path: TypeAlias = "tuple[int | Placing, ...]"


@dataclass(frozen=True, slots=True)
class _Approx:
    """A walk up from below slot ``index``, the guards the Decision's scope is placed under in
    this class skipped; where it re-enters ``index`` it is walked with no existence guards (it
    reaches at least what the Decision's walk reaches) and marked."""

    index: int


Origin: TypeAlias = "_Path | _Approx | None"


@dataclass(slots=True)
class _Outcome:
    """A walk's result: a refusal, or what it reached out of its class (each with its way),
    and whether it re-entered an ``_Approx`` walk's slot (then the refusal or what it
    reached is a superset of the Decision's own)."""

    refusal: _Refusal | None = None
    exits: dict[Exit, tuple[str, ...]] = field(default_factory=dict)
    reentered: bool = False
    # ``_enter``'s: what the exits stand for in the class above, to read there.
    lifted: list[_Push] = field(default_factory=list)


@dataclass(slots=True)
class _Entry:
    """What the domains of a class's subtree reach out of it, for one set of references
    out: each with its way, the first Decision's key, and every Decision's place
    (``("own", guard, key)``; ``("via", slot, entry, name)``: every Decision of a child's
    entry below that slot, named ``name`` here; ``("path", path, key)``: one, by its
    ``_Path``)."""

    exits: dict[Exit, tuple[str, ...]]
    owner: str
    origins: list[tuple[Any, ...]]


def _places(origins: Iterable[tuple[Any, ...]], name: str = "") -> Iterator[tuple[_Path, str]]:
    """Each Decision of ``origins``: its place, and its key (under ``name``)."""
    for origin in origins:
        if origin[0] == "own":
            yield (origin[1],), member_key(name, origin[2])
        elif origin[0] == "path":
            yield cast(_Path, origin[1]), member_key(name, origin[2])
        else:
            index, entry, local = origin[1], cast(_Entry, origin[2]), origin[3]
            for place, key in _places(entry.origins, member_key(name, local)):
                yield (index, *place), key


@dataclass(slots=True)
class _Summary:
    """The domains of a class's subtree: what they reach out of it
    (``entries``), and the refusals found (one, unless every refusal is asked for: then
    every Decision refused, by its key)."""

    entries: list[_Entry] = field(default_factory=list)
    refusals: list[_Refusal] = field(default_factory=list)


_Push: TypeAlias = "tuple[Ref, tuple[str, ...]]"


def _way(name: str, exit: Exit, way: tuple[str, ...]) -> tuple[str, ...]:
    """A way out of a placed Space (at ``name``), under the placing class's keys. A way out
    through a formal keeps the formal as a step: the hop across the slot, from the child's
    formal to what the placing class binds it to, is what its author needs to see (a
    forwarding alias inside one class is no step)."""
    found = tuple(member_key(name, step) for step in way)
    if exit[0] == "formal":
        found = (*found, member_key(name, exit[1]))
    return found


class _Domains:
    """The composed domain rule (see the module docstring)."""

    def __init__(self, *, every: bool = False) -> None:
        self.inference = _TemplateInference()
        # Every refusal, not the first: a diagnostic (``domain_refusals``).
        self.every = every
        self.name = "domains, every" if every else "domains"

    # -- existence guards

    @staticmethod
    def _chain(compiled: Compiled, head: int | Placing) -> frozenset[object]:
        """``head`` and every guard it is evaluated under, in ``compiled``'s class."""
        cache = _cache(compiled, "chain")
        found = cache.get(head)
        if found is None:
            nodes: list[object] = []
            current: Ref | None = head
            while current is not None:
                nodes.append(current)
                guard = compiled.records[current].guard if isinstance(current, int) else None
                current = None if guard is None else compiled.ref(guard)
            found = cache[head] = frozenset(nodes)
        return cast(frozenset[object], found)

    def _skip(self, compiled: Compiled, origin: Origin) -> frozenset[object]:
        if origin is None:
            return frozenset()
        if isinstance(origin, _Approx):
            return self._chain(compiled, slot(compiled, origin.index).guard)
        if len(origin) == 1:
            return self._chain(compiled, origin[0])
        return self._chain(compiled, slot(compiled, cast(int, origin[0])).guard)

    @staticmethod
    def _into(origin: Origin, index: int) -> tuple[Origin, bool]:
        """The origin below slot ``index``, and whether an approximate walk re-entered
        there."""
        if isinstance(origin, _Approx):
            return None, origin.index == index
        if origin is not None and len(origin) > 1 and origin[0] == index:
            return origin[1:], False
        return None, False

    # -- the walk

    def walk(
        self,
        compiled: Compiled,
        origin: Origin,
        starts: Sequence[_Push],
        *,
        exempt: bool = False,
    ) -> _Outcome:
        """Everything ``starts`` may read in ``compiled``'s class, the existence guards of
        ``origin`` skipped (``exempt``: the starts themselves are a domain's inputs, read
        whatever guards them); see the module docstring."""
        if origin is None and len(starts) == 1 and not exempt:
            return self._single(compiled, starts[0])
        return self._walk(compiled, origin, starts, exempt)

    def _single(self, compiled: Compiled, start: _Push) -> _Outcome:
        ref, route = start
        memo = _cache(compiled, "walk")
        found = memo.get(ref)
        if found is None:
            found = memo[ref] = self._walk(compiled, None, [(ref, ())], False)
        outcome = cast(_Outcome, found)
        if not route:
            return outcome
        refusal = outcome.refusal
        return _Outcome(
            None
            if refusal is None
            else _Refusal(refusal.owner, refusal.choice, refusal.dynamic, (*route, *refusal.route)),
            {exit: (*route, *way) for exit, way in outcome.exits.items()},
            outcome.reentered,
        )

    def _walk(
        self,
        compiled: Compiled,
        origin: Origin,
        starts: Sequence[_Push],
        exempt: bool,
    ) -> _Outcome:
        outcome = _Outcome()
        skip = self._skip(compiled, origin)
        seen: set[int] = set()
        pending: list[tuple[int, tuple[str, ...]]] = []

        def push(ref: Ref, route: tuple[str, ...], checked: bool = True) -> None:
            if checked and not isinstance(ref, tuple) and ref in skip:
                return  # what decides whether the choice exists holds where it does
            out = _classify(compiled, ref)
            if out is not None:
                outcome.exits.setdefault(out, route)
                return
            node = cast("int | Placed", ref)
            if isinstance(node, Placed):
                down(node.path, node.member, route)
                return
            if node not in seen:
                seen.add(node)
                pending.append((node, route))

        def down(steps: Sequence[int], target: int | Placing, route: tuple[str, ...]) -> None:
            found = self._enter(compiled, origin, steps, [(target, ())])
            outcome.reentered |= found.reentered
            if found.refusal is not None:
                if outcome.refusal is None:
                    outcome.refusal = _Refusal(
                        None,
                        found.refusal.choice,
                        found.refusal.dynamic,
                        (*route, *found.refusal.route),
                    )
                return
            for ref, way in found.lifted:
                push(ref, (*route, *way))

        for ref, route in starts:
            push(ref, route, checked=not exempt)
        while pending and outcome.refusal is None:
            node, route = pending.pop()
            spec = compiled.records[node]
            step = route if forwards(spec) else (*route, spec.owner)
            if spec.kind == "decision":
                outcome.refusal = _Refusal(None, spec.key, None, step)
                break
            for ref in fields(compiled, node).following():
                push(ref, step)
            if spec.call_style == "self" and spec.function is not None:
                items = self.inference.reads(spec.function, compiled)
                if any(isinstance(item, _Dynamic) for item in items):
                    outcome.refusal = _Refusal(None, None, spec.owner, step)
                    break
                self._reads(items, step, push, down)
        if outcome.refusal is not None:
            outcome.exits.clear()
        return outcome

    @staticmethod
    def _reads(items: Iterable[object], route: tuple[str, ...], push: Any, down: Any) -> None:
        """A method's inferred reads (``_reads._TemplateInference``)."""
        for item in items:
            if isinstance(item, _Ref):
                if not item.steps:
                    push(item.node, route)
                else:
                    down(item.steps, item.node, route)
            else:
                assert isinstance(item, (_Missing, _Dynamic))

    def _enter(
        self,
        compiled: Compiled,
        origin: Origin,
        steps: Sequence[int],
        starts: Sequence[_Push],
    ) -> _Outcome:
        """Walk ``starts`` (records of the class ``steps`` reach) there, and what they reach
        out of it, up to ``compiled``'s class: what to read there, each with its way
        (``_Outcome.lifted``), or a refusal; and whether it re-entered an ``_Approx`` walk's
        slot."""
        held = slot(compiled, steps[0])
        name = compiled.slots[steps[0]].name
        below, reentered = self._into(origin, steps[0])
        child = held.compiled
        if len(steps) > 1:
            inner = self._enter(child, below, steps[1:], starts)
            reentered |= inner.reentered
            if inner.refusal is not None:
                return _Outcome(inner.refusal.under(name), reentered=reentered)
            outcome = self.walk(child, below, inner.lifted)
        else:
            outcome = self.walk(child, below, starts)
        reentered |= outcome.reentered
        if outcome.refusal is not None:
            return _Outcome(outcome.refusal.under(name), reentered=reentered)
        lifted: list[_Push] = [
            (ref, _way(name, exit, outcome.exits[exit]))
            for exit in outcome.exits
            for ref in lift(held, (exit,))
        ]
        return _Outcome(lifted=lifted, reentered=reentered)

    # -- per class

    def summary(self, compiled: Compiled) -> _Summary:
        """The domains of ``compiled``'s class's subtree (keys relative to it); found
        once."""
        cache = _cache(compiled, self.name)
        if None not in cache:
            # The classes below first, deepest first: a chain of placements may be deep.
            for below in self._below(compiled):
                held = _cache(below, self.name)
                if None not in held:
                    held[None] = self._summary(below)
        return cast(_Summary, cache[None])

    def _below(self, compiled: Compiled) -> list[Compiled]:
        """``compiled`` and every class it places, each after those it places (none whose
        domains are known already)."""
        found: list[Compiled] = []
        done: set[int] = set()
        stack: list[tuple[Compiled, Iterator[int]]] = [(compiled, iter(range(len(compiled.slots))))]
        while stack:
            current, slots = stack[-1]
            index = next(slots, None)
            if index is None:
                stack.pop()
                found.append(current)
                continue
            held = slot(current, index)
            if id(held.compiled) in done:
                continue
            done.add(id(held.compiled))
            if None not in _cache(held.compiled, self.name):
                stack.append((held.compiled, iter(range(len(held.compiled.slots)))))
        return found

    def _summary(self, compiled: Compiled) -> _Summary:
        summary = _Summary()
        merged: dict[frozenset[Exit], _Entry] = {}

        def add(exits: dict[Exit, tuple[str, ...]], owner: str, origin: tuple[Any, ...]) -> None:
            if not exits:
                return
            key = frozenset(exits)
            held = merged.get(key)
            if held is None:
                merged[key] = _Entry(dict(exits), owner, [origin])
            else:
                held.origins.append(origin)

        def refused(refusal: _Refusal) -> bool:
            """Record a refusal; whether to stop."""
            summary.refusals.append(refusal)
            return not self.every

        # Its own Decisions, in member order.
        for node, spec in enumerate(compiled.records):
            domain = spec.domain
            if domain is None or spec.kind != "decision":
                continue
            named = set(domain.membership_inputs)
            starts: list[_Push] = [
                (compiled.ref(edge), ())
                for _, edge in dict.fromkeys(
                    argument for argument in spec.domain_arguments if argument[0] in named
                )
            ]
            head = cast("int | Placing", compiled.ref(cast("Edge", spec.guard)))
            outcome = self.walk(compiled, (head,), starts, exempt=True)
            if outcome.refusal is not None:
                if refused(outcome.refusal.at(spec.key)):
                    return summary
                continue
            add(outcome.exits, spec.key, ("own", head, spec.key))
        # Each placed Space's, followed into its bindings here.
        for index in range(len(compiled.slots)):
            held = slot(compiled, index)
            name = compiled.slots[index].name
            below = self.summary(held.compiled)
            for refusal in below.refusals:
                if refused(refusal.under(name)):
                    return summary
            for entry in below.entries:
                pushes: list[_Push] = [
                    (ref, _way(name, exit, entry.exits[exit]))
                    for exit in entry.exits
                    for ref in lift(held, (exit,))
                ]
                outcome = self.walk(compiled, _Approx(index), pushes)
                if outcome.reentered:
                    # It re-entered the subtree holding the Decisions: per Decision, exactly.
                    for place, key in _places(entry.origins, name):
                        exact = self.walk(compiled, (index, *place), pushes)
                        if exact.refusal is not None:
                            if refused(exact.refusal.at(key)):
                                return summary
                            continue
                        add(exact.exits, key, ("path", (index, *place), key))
                    continue
                if outcome.refusal is not None:
                    if not self.every:
                        refused(outcome.refusal.at(member_key(name, entry.owner)))
                        return summary
                    for _, key in _places(entry.origins, name):
                        refused(outcome.refusal.at(key))
                    continue
                add(outcome.exits, member_key(name, entry.owner), ("via", index, entry, name))
        summary.entries = list(merged.values())
        return summary


def check_domains(compiled: Compiled) -> None:
    """The domain rule of ``compiled``'s class as a root (see the module docstring): the
    first refusal found, ``domain-reads-choice``, its Decision's key and its way relative
    to the root."""
    refusals = _Domains().summary(compiled).refusals
    if refusals:
        raise refusals[0].error()


def domain_refusals(compiled: Compiled) -> dict[str, DefinitionError]:
    """Every Decision the domain rule refuses in ``compiled``'s class as a root, by its key
    (a diagnostic: compiling stops at the first)."""
    refusals = _Domains(every=True).summary(compiled).refusals
    return {cast(str, refusal.owner): refusal.error() for refusal in refusals}


# -- views read out ----------------------------------------------------------------------------

# What passes a value on without computing from it: what reads one of these reads what it
# passes on.
_PASSING = frozenset(("alias", "members", "select"))


@dataclass(frozen=True, slots=True)
class _ReadIn:
    """A record reading a view: the key it is named by (``reader``), the view, and the
    members passing the view on to it (``through``), keys relative to the class walked."""

    reader: str
    view: str
    through: tuple[str, ...]

    def under(self, name: str) -> _ReadIn:
        return _ReadIn(
            member_key(name, self.reader),
            member_key(name, self.view),
            tuple(member_key(name, step) for step in self.through),
        )

    def error(self) -> DefinitionError:
        through = f" (through {' -> '.join(self.through)})" if self.through else ""
        detail = (
            f"{self.reader}: reads the view {self.view}{through}; a view is read out, never "
            "read in: a constraint, derived value, guard, domain or binding reads facts, "
            "choices and derived values (read what the view reads)"
        )
        details = (("declaration", self.reader), ("view", self.view))
        return DefinitionError(
            detail,
            findings=(
                Finding(FindingKind.AUTHORING, "view-read-in", self.reader, detail, details),
            ),
        )


# A record a reference names: the class it is a record of, its number there, and the slots'
# names from the class read to it.
_Target: TypeAlias = "tuple[Compiled, int, str]"


def _target(compiled: Compiled, ref: Ref) -> _Target | None:
    """The record ``ref`` names, a member of ``compiled``'s class or of a Space it places;
    None for a reference out of the class."""
    if isinstance(ref, int):
        return compiled, ref, ""
    if not isinstance(ref, Placed):
        return None
    names: list[str] = []
    below = compiled
    for step in ref.path:
        names.append(below.slots[step].name)
        below = below.slots[step].compiled
    return below, ref.member, ".".join(names)


def _ways(compiled: Compiled) -> dict[int, tuple[str, ...] | None]:
    """The view each record of ``compiled``'s class reads, by the way to it, as found."""
    return cast("dict[int, tuple[str, ...] | None]", _cache(compiled, "views read"))


@dataclass(slots=True)
class _Frame:
    compiled: Compiled
    record: int
    passed: Iterator[Ref]
    waiting: _Target | None = None


class _Views:
    """Views read out (see the module docstring): per class, the first record reading a
    view, its own records first, then each placed Space's (keys relative to the class)."""

    def __init__(self) -> None:
        self.inference = _TemplateInference()

    # -- what a reference reads

    def reads(self, compiled: Compiled, record: int) -> Iterator[Ref]:
        """What ``record`` reads: its references, then a ``self`` method's inferred reads (a
        dynamic method's are not known, and read as nothing here)."""
        yield from fields(compiled, record).dependencies()
        spec = compiled.records[record]
        if spec.call_style == "self" and spec.function is not None:
            for item in self.inference.reads(spec.function, compiled):
                if isinstance(item, _Ref) and not isinstance(item.node, Placing):
                    yield Placed(item.steps, item.node) if item.steps else item.node

    @staticmethod
    def _generated(spec: Member) -> bool:
        """A computation the engine writes for a member (an expression, a projection, a
        view's output), named by that member."""
        return spec.kind in ("derived", "const") and spec.source_owner is not None

    def passes(self, compiled: Compiled, record: int) -> Iterator[Ref]:
        """What ``record`` passes on, if it passes a value on: an alias's, a selection's or a
        ``Members``'s sources, a generated computation's inputs (a binding is a reader)."""
        spec = compiled.records[record]
        if spec.kind in _PASSING and record not in compiled.binds:
            held = fields(compiled, record)
            if held.output is not None:
                yield held.output
            yield from held.alternatives
        elif self._generated(spec):
            guard = fields(compiled, record).guard
            yield from (ref for ref in self.reads(compiled, record) if ref != guard)

    def view(self, compiled: Compiled, ref: Ref) -> tuple[str, ...] | None:
        """The view ``ref`` reads, if it reads one: a view, or what passes one on; the way
        (the passing members, then the view), keys relative to ``compiled``'s class. A
        reference out of the class reads none (a formal's binding is a reader above)."""
        target = _target(compiled, ref)
        if target is None:
            return None
        below, record, name = target
        way = self._way(below, record)
        return None if way is None else tuple(member_key(name, step) for step in way)

    def _way(self, compiled: Compiled, record: int) -> tuple[str, ...] | None:
        """``view`` of a record of ``compiled``'s class, found once per record. What passes a
        value on may be a long chain (an expression's), so it is walked with a stack of its
        own: each frame a record whose way is being found, what it passes on, and the
        record it waits on (its class, number, and the slots' names to it)."""
        memo = _ways(compiled)
        if record in memo:
            return memo[record]
        frames: list[_Frame] = []

        def opened(below: Compiled, at: int) -> bool:
            """Whether ``at`` waits on a frame of its own (else its way is known)."""
            spec = below.records[at]
            if spec.kind == "view":
                _ways(below)[at] = (spec.key,)
                return False
            _ways(below)[at] = None  # until found: the static order is acyclic
            frames.append(_Frame(below, at, iter(self.passes(below, at))))
            return True

        opened(compiled, record)
        while frames:
            frame = frames[-1]
            target = frame.waiting
            frame.waiting = None
            if target is None:
                following = next(frame.passed, None)
                if following is None:
                    frames.pop()  # passes no view on
                    continue
                target = _target(frame.compiled, following)
                if target is None:
                    continue
                if target[1] not in _ways(target[0]) and opened(target[0], target[1]):
                    frame.waiting = target
                    continue
            below, at, name = target
            found = _ways(below)[at]
            if found is None:
                continue
            way = tuple(member_key(name, step) for step in found)
            spec = frame.compiled.records[frame.record]
            _ways(frame.compiled)[frame.record] = way if self._generated(spec) else (spec.key, *way)
            frames.pop()
        return memo[record]

    # -- the readers

    def own(self, compiled: Compiled) -> _ReadIn | None:
        """The first of ``compiled``'s own records reading a view. A view reads anything:
        a view, the computations written for it and its own guard are not readers here; nor
        is what passes a value on (its reader is)."""
        views = {spec.key for spec in compiled.records if spec.kind == "view"}
        for record, spec in enumerate(compiled.records):
            if spec.kind in ("view", "validity", "param", "const"):
                continue
            if spec.kind in _PASSING and record not in compiled.binds:
                continue
            if self._generated(spec) or (spec.kind == "guard" and spec.owner in views):
                continue
            for ref in self.reads(compiled, record):
                found = self.view(compiled, ref)
                if found is not None:
                    return _ReadIn(spec.owner, found[-1], found[:-1])
        return None

    def summary(self, compiled: Compiled) -> _ReadIn | None:
        """The first record of ``compiled``'s class's subtree reading a view; found once per
        class, the classes it places first (deepest first)."""
        cache = _cache(compiled, "views read in")
        if None not in cache:
            for below in self._below(compiled):
                held = _cache(below, "views read in")
                if None not in held:
                    held[None] = self._summary(below)
        return cast("_ReadIn | None", cache[None])

    def _summary(self, compiled: Compiled) -> _ReadIn | None:
        found = self.own(compiled)
        if found is not None:
            return found
        for index in range(len(compiled.slots)):
            below = self.summary(slot(compiled, index).compiled)
            if below is not None:
                return below.under(compiled.slots[index].name)
        return None

    @staticmethod
    def _below(compiled: Compiled) -> list[Compiled]:
        """``compiled`` and every class it places, each after those it places (none whose
        readers are known already)."""
        found: list[Compiled] = []
        done: set[int] = set()
        stack: list[tuple[Compiled, Iterator[int]]] = [(compiled, iter(range(len(compiled.slots))))]
        while stack:
            current, slots = stack[-1]
            index = next(slots, None)
            if index is None:
                stack.pop()
                found.append(current)
                continue
            held = current.slots[index].compiled
            if id(held) in done:
                continue
            done.add(id(held))
            if None not in _cache(held, "views read in"):
                stack.append((held, iter(range(len(held.slots)))))
        return found


def check_views(compiled: Compiled) -> None:
    """Views read out, ``compiled``'s class as a root (see the module docstring): the first
    record found reading a view, ``view-read-in``, naming it and the view relative to the
    root."""
    found = _Views().summary(compiled)
    if found is not None:
        raise found.error()


__all__ = [
    "check_domains",
    "check_views",
    "domain_refusals",
    "order",
]
