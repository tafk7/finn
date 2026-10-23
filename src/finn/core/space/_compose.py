# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What crosses a slot, checked per class and composed per placement.

A Space class is compiled once (``_template``: its compiled form, ``Compiled``) and placed
as scopes (``_scopes``). Three checks read across the slots a class places, and each is
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
  compiled first).
- **Placements** (``check_placement``). Where a placing class leaves a formal unsupplied,
  a Decision over nodes guarded by ``supplied`` of it never applies: what exists only
  where it applies is not placed, and a record that still reads it is refused
  (``reference-not-placed``), as the placing class is linked.
- **The domain rule** (``check_domains``): a domain's membership reads facts only. Per
  class, each domain's walk follows what its membership inputs may read (arguments,
  alternatives, a view's validity and a validity's pieces, outputs, selectors, guards, a
  method's inferred reads and what ``present()`` reads), skipping the guards that decide
  whether the Decision exists; it stops at a choice (``domain-reads-choice``), and at the
  class's own formals and the guard it is placed under, which it reports as reached. A
  child's are followed into the placing class's bindings, with the guards above the slot
  as existence guards of the child's Decision too, and so up to the root, per placement
  chain. A class records, for its own formals, which are reached by any domain in its
  subtree (``_Entry``), once per placement signature.

A class is read **at a placement** (``Placement``): the Decisions over nodes of it that
never apply there. A placement changes a class only where such a Decision never applies:
its selector is a constant, what is read through it has no alternatives, and its
candidates, what exists only where it applies and the validity pieces of both are not
placed.

Each walk is that of the design space the class is placed in: the composition is exact.
Where a walk back up re-enters the subtree holding the Decision it started from, its
existence guards inside that subtree are known only per Decision: the walk is done once
with none of them and, only where it re-entered, again per Decision (``_Approx``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ._ordering import cyclic, dependency_order
from ._reads import _Dynamic, _Missing, _Ref, _TemplateInference
from ._table import Binding, Table, member_key
from .errors import DefinitionError
from .ir import Placed, Placing, Selected, forwards

if TYPE_CHECKING:
    from ._template import Compiled
    from .ir import Edge

_OWNER = "space.compose"


# -- placements ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Placement:
    """A class read at a placement: the Decisions over nodes of it that never apply there
    (guarded by ``supplied`` of a formal the placing class leaves unsupplied)."""

    never: frozenset[int] = frozenset()


ROOT = Placement()


def placement(compiled: Compiled, bindings: Mapping[str, Binding]) -> Placement:
    return ROOT


@dataclass(frozen=True, slots=True)
class _Slot:
    """A Space a class places: its class's compiled form and placement, the bindings of its
    formals (a reference's ``node``: the record the placing class writes for it), what it is
    placed under in the placing class, and the Decision over nodes whose candidate it is
    (where that may never apply)."""

    compiled: Compiled
    placement: Placement
    bindings: Mapping[str, Binding]
    guard: int | Placing
    requires: int | None


def _cache(compiled: Compiled, name: str) -> dict[Any, Any]:
    return cast(dict[Any, Any], compiled.composed.setdefault(name, {}))


def slot(compiled: Compiled, index: int) -> _Slot:
    """Slot ``index`` of ``compiled``, as the walks read it."""
    cache = _cache(compiled, "slot")
    found = cache.get(index)
    if found is None:
        held = compiled.slots[index]
        found = cache[index] = _Slot(
            held.compiled,
            placement(held.compiled, held.bindings),
            held.bindings,
            cast("int | Placing", compiled.ref(held.guard)),
            held.requires,
        )
    return cast(_Slot, found)


def _table_slot(table: Table, draft: int) -> _Slot:
    """A slot of a class being compiled (its bindings and guard references of the class)."""
    held = table.placing(draft)
    return _Slot(
        held.compiled,
        placement(held.compiled, held.bindings),
        held.bindings,
        cast("int | Placing", table.drafts[draft].guard),
        held.requires,
    )


def _table_descent(table: Table, edge: Placed | Selected) -> tuple[_Slot, tuple[int, ...], int]:
    """An edge out of a class being compiled, as a walk reads it: its slot, the slots below
    it to the class holding its target, and the target's record there."""
    held = _table_slot(table, table.slot_drafts[edge.path[0]])
    steps = edge.path[1:]
    if isinstance(edge, Selected):
        reached = held.compiled.reached(steps)
        return held, steps, reached.own_selection[(edge.choice, edge.member)]
    return held, steps, edge.member


# -- a class at a placement -----------------------------------------------------------------

# A reference out of a class: a formal's (``formal``, name, mode), or the guard it is placed
# under (``guard``, mode). A mode: ``r`` reached, ``s`` followed as ``present()`` follows a
# value's sources, ``S`` the first such step.
Exit: TypeAlias = tuple[str, ...]
Mode = Literal["r", "s", "S"]
# What a walk reads: a record of the class walked, the guard it is placed under, a member of
# a Space it places (walked there), or a reference out of it.
Ref: TypeAlias = "int | Placing | Placed | Exit"


def _dropped(compiled: Compiled, at: Placement, ref: Ref) -> bool:
    """A record written for a placed Space, or a member read in one, below a candidate of a
    Decision over nodes that never applies."""
    if isinstance(ref, Placed):
        return compiled.slots[ref.path[0]].requires in at.never
    written = compiled.written.get(ref) if isinstance(ref, int) else None
    return written is not None and compiled.slots[written].requires in at.never


def excluded(compiled: Compiled, at: Placement, ref: Ref) -> bool:
    """Whether ``at`` places nothing for ``ref``: it is below a candidate of a Decision over
    nodes that never applies there, or exists only where one applies. A selection is placed
    where something reads it."""
    if not at.never:
        return False
    if _dropped(compiled, at, ref):
        return True
    if isinstance(ref, int) and (ref in compiled.selections or _chosen(compiled, ref)):
        return False
    return compiled.requires.get(ref) in at.never


def _chosen(compiled: Compiled, record: int) -> bool:
    """Whether a record is a selection the class makes in a placed Space's choice."""
    cache = _cache(compiled, "chosen")
    found = cache.get(None)
    if found is None:
        found = cache[None] = frozenset(compiled.chosen.values())
    return record in cast(frozenset[int], found)


def is_decision(compiled: Compiled, at: Placement, ref: int) -> bool:
    """A Decision at ``at`` (a selector that never applies is a constant there)."""
    return compiled.records[ref].kind == "decision" and compiled.selectors.get(ref) not in at.never


def _never_selection(compiled: Compiled, at: Placement, record: int) -> bool:
    selection = compiled.selections.get(record)
    return selection is not None and selection[0] in at.never


def pieces(compiled: Compiled, at: Placement) -> list[Ref]:
    """A validity's pieces at ``at``: its constraints and Decisions, then each placed
    Space's validity. A formal it lists stands for nothing here (no binding is a Decision
    of its own)."""
    own: list[Ref] = []
    placed: list[Ref] = []
    for edge in compiled.records[compiled.validity].constraints:
        piece = compiled.ref(edge)
        if isinstance(piece, Placed):
            if not _dropped(compiled, at, piece):
                placed.append(piece)
            continue
        piece = cast(int, piece)
        if piece in compiled.formal_names or excluded(compiled, at, piece):
            continue
        if compiled.selectors.get(piece) not in at.never:
            own.append(piece)
    return [*own, *placed]


@dataclass(frozen=True, slots=True)
class _Fields:
    """A node's references at a placement, by role."""

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


def fields(compiled: Compiled, at: Placement, record: int) -> _Fields:
    """A record's references at ``at``, as a walk reads them (``Compiled.ref``)."""
    spec = compiled.records[record]
    ref = compiled.ref
    alternatives = tuple(ref(target) for _, target in spec.alternatives)
    constraints: tuple[Ref, ...] = tuple(ref(piece) for piece in spec.constraints)
    if record == compiled.validity:
        constraints = tuple(pieces(compiled, at))
    elif at.never and _never_selection(compiled, at, record):
        alternatives = ()  # read through a Decision over nodes that never applies here
    elif at.never and spec.kind == "members":
        alternatives = tuple(
            target for target in alternatives if not _dropped(compiled, at, target)
        )
    return _Fields(
        tuple(ref(edge) for _, edge in spec.arguments),
        tuple(ref(edge) for _, edge in spec.domain_arguments),
        None if spec.guard is None else ref(spec.guard),
        None if spec.output is None else ref(spec.output),
        None if spec.selector is None else ref(spec.selector),
        constraints,
        alternatives,
    )


def lift(slot_: _Slot, exits: Iterable[Exit]) -> Iterator[tuple[int | Placing, Mode]]:
    """What a placed Space's references out stand for in its placing class: a formal, its
    binding's record (or, for a literal or a default, the guard the scope is placed under: a
    constant's only edge); the guard it is placed under."""
    for exit in exits:
        kind = exit[0]
        if kind == "formal":
            binding = slot_.bindings[exit[1]]
            if binding.node is not None:
                yield binding.node, cast(Mode, exit[2])
            else:
                yield slot_.guard, "r"
        else:  # guard
            yield slot_.guard, cast(Mode, exit[1])


def _classify(compiled: Compiled, ref: Ref, mode: Mode) -> Exit | None:
    """``ref``'s reference out of ``compiled``'s class, if it is one."""
    if isinstance(ref, tuple):
        return ref
    if isinstance(ref, Placing):
        return ("guard", "s" if mode == "S" else mode)
    formal = compiled.formal_names.get(ref) if isinstance(ref, int) else None
    if formal is not None:
        return ("formal", formal, mode)
    return None


# -- cycles ------------------------------------------------------------------------------------


def exits(compiled: Compiled, at: Placement, ref: int | Placed) -> frozenset[Exit]:
    """What ``ref`` reaches out of ``compiled``'s class at ``at`` by structural edges (its
    dependencies, through any depth of the Spaces it places). A class's graph is acyclic
    once linked, so each record's are found once."""
    memo = cast(dict[object, frozenset[Exit]], _cache(compiled, "exits").setdefault(at, {}))
    if ref in memo:
        return memo[ref]
    stack: list[tuple[int | Placed, Iterator[Ref]]] = [(ref, iter(_successors(compiled, at, ref)))]
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
        out = _classify(compiled, following, "r")
        if out is not None:
            found[current].add(out)
            continue
        following = cast("int | Placed", following)
        if excluded(compiled, at, following):
            continue
        if following in memo:
            found[current] |= memo[following]
        elif following not in open_:
            open_.add(following)
            found[following] = set()
            stack.append((following, iter(_successors(compiled, at, following))))
    return memo[ref]


def _successors(compiled: Compiled, at: Placement, ref: int | Placed) -> list[Ref]:
    """A record's dependencies at ``at``; a placed Space's member's, what it reaches there,
    lifted into this class."""
    if isinstance(ref, Placed):
        return [found for found, _ in _through(compiled, ref.path, ref.member)]
    return fields(compiled, at, ref).dependencies()


def _reaching(compiled: Compiled, at: Placement, ref: Ref) -> frozenset[Exit]:
    """What a reference of ``compiled``'s class reaches out of it: itself, if it is one."""
    out = _classify(compiled, ref, "r")
    if out is not None:
        return frozenset((out,))
    return exits(compiled, at, cast("int | Placed", ref))


def _through(
    compiled: Compiled, steps: Sequence[int], target: int | Placing
) -> list[tuple[int | Placing, Mode]]:
    """What ``target`` (a record of the class ``steps`` reach) reaches out of it, followed up
    to ``compiled``'s class."""
    return _up(slot(compiled, steps[0]), steps[1:], target)


def _up(
    held: _Slot, steps: Sequence[int], target: int | Placing
) -> list[tuple[int | Placing, Mode]]:
    """What ``target`` (a record of the class ``steps`` reach from ``held``'s) reaches out of
    it, followed up through ``held`` into its placing class."""
    child, at = held.compiled, held.placement
    reached: set[Exit] = set()
    if steps:
        for ref, _ in _through(child, steps, target):
            reached |= _reaching(child, at, ref)
    else:
        reached |= _reaching(child, at, target)
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
            found = list(node.dependencies)
            names.append(node.key)
        else:
            held, steps, target = _table_descent(table, vertex)
            found = [ref for ref, _ in _up(held, steps, target)]
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
    child, at = held.compiled, held.placement

    def reaches(ref: Ref) -> bool:
        return any(
            wanted(up) for exit in _reaching(child, at, ref) for up, _ in lift(held, (exit,))
        )

    found: set[str] = set()
    starts: list[Ref] = []
    if steps:
        below = slot(child, steps[0])
        found |= _named(
            below, steps[1:], target, member_key(name, child.slots[steps[0]].name), reaches
        )
        starts = [ref for ref, _ in _up(below, steps[1:], target) if reaches(ref)]
    elif reaches(target):
        starts = [target]
    formals: set[str] = set()
    seen: set[object] = set(starts)
    pending = list(starts)
    while pending:
        current = pending.pop()
        if _classify(child, current, "r") is not None:
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
        for ref in _successors(child, at, current):
            out = _classify(child, ref, "r")
            if out is not None:
                if out[0] == "formal" and reaches(out):
                    formals.add(out[1])
                continue
            if ref not in seen and not excluded(child, at, ref) and reaches(ref):
                seen.add(ref)
                pending.append(ref)
    # A formal bound to a literal or a default is a constant node of the placed scope.
    found.update(
        member_key(name, formal) for formal in formals if held.bindings[formal].node is None
    )
    return found


# -- placements ------------------------------------------------------------------------------


def check_placements(table: Table) -> None:
    """Each Space a class being compiled places where a Decision over nodes of it never
    applies: no record it places there reads what is not placed (``check_placement``)."""
    for draft in table.slot_drafts:
        held = _table_slot(table, draft)
        if held.placement.never:
            check_placement(held.compiled, held.placement, table.drafts[draft].name)


def check_placement(compiled: Compiled, at: Placement, name: str) -> None:
    """No record of ``compiled``'s class placed at ``at`` (at scope ``name``) reads what it
    does not place: ``reference-not-placed`` (as the line's one table of every placement
    refused it, when lowering the reader there)."""
    for record in range(len(compiled.records)):
        if excluded(compiled, at, record):
            continue
        for ref in fields(compiled, at, record).dependencies():
            if isinstance(ref, (int, Placed)) and excluded(compiled, at, ref):
                raise _not_placed(compiled, at, name, record, ref)


def _not_placed(
    compiled: Compiled, at: Placement, name: str, reader: int, target: int | Placed
) -> DefinitionError:
    owner = member_key(name, compiled.records[reader].owner)
    if _dropped(compiled, at, target):
        # Below a candidate that is not placed: the candidate's node declaration.
        placed = target.path[0] if isinstance(target, Placed) else compiled.written[target]
        record = compiled.slots[placed].record
        describe = getattr(record, "describe", None)
        text = describe() if callable(describe) else repr(record)
        detail = f"{owner}: {text} is not placed in {name or '<root>'}"
    else:
        detail = f"{owner}: {member_key(name, compiled.key_of(target))} is not placed"
    return DefinitionError.broken("reference-not-placed", _OWNER, detail, declaration=owner)


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
    """The domains of a class's subtree at a placement: what they reach out of it
    (``entries``), and the refusals found (one, unless every refusal is asked for: then
    every Decision refused, by its key)."""

    entries: list[_Entry] = field(default_factory=list)
    refusals: list[_Refusal] = field(default_factory=list)


_Push: TypeAlias = "tuple[Ref, Mode, tuple[str, ...]]"


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
        at: Placement,
        origin: Origin,
        starts: Sequence[_Push],
        *,
        exempt: bool = False,
    ) -> _Outcome:
        """Everything ``starts`` may read in ``compiled``'s class at ``at``, the existence
        guards of ``origin`` skipped (``exempt``: the starts themselves are a domain's
        inputs, read whatever guards them); see the module docstring."""
        if origin is None and len(starts) == 1 and not exempt:
            return self._single(compiled, at, starts[0])
        return self._walk(compiled, at, origin, starts, exempt)

    def _single(self, compiled: Compiled, at: Placement, start: _Push) -> _Outcome:
        ref, mode, route = start
        memo = _cache(compiled, "walk").setdefault(at, {})
        key = (ref, mode)
        found = memo.get(key)
        if found is None:
            found = memo[key] = self._walk(compiled, at, None, [(ref, mode, ())], False)
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
        at: Placement,
        origin: Origin,
        starts: Sequence[_Push],
        exempt: bool,
    ) -> _Outcome:
        outcome = _Outcome()
        skip = self._skip(compiled, origin)
        seen: set[tuple[int, Mode]] = set()
        pending: list[tuple[int, Mode, tuple[str, ...]]] = []

        def push(ref: Ref, mode: Mode, route: tuple[str, ...], checked: bool = True) -> None:
            if checked and mode == "r" and not isinstance(ref, tuple) and ref in skip:
                return  # what decides whether the choice exists holds where it does
            out = _classify(compiled, ref, mode)
            if out is not None:
                outcome.exits.setdefault(out, route)
                return
            node = cast("int | Placed", ref)
            if excluded(compiled, at, node):
                return
            if isinstance(node, Placed):
                down(node.path, node.member, mode, route)
                return
            if (node, mode) not in seen:
                seen.add((node, mode))
                pending.append((node, mode, route))

        def down(
            steps: Sequence[int], target: int | Placing, mode: Mode, route: tuple[str, ...]
        ) -> None:
            found = self._enter(compiled, at, origin, steps, [(target, mode, ())])
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
            for ref, lifted_mode, way in found.lifted:
                push(ref, lifted_mode, (*route, *way))

        for ref, mode, route in starts:
            push(ref, mode, route, checked=not exempt)
        while pending and outcome.refusal is None:
            node, mode, route = pending.pop()
            spec = compiled.records[node]
            if mode == "r":
                step = route if forwards(spec) else (*route, spec.owner)
                if is_decision(compiled, at, node):
                    outcome.refusal = _Refusal(None, spec.key, None, step)
                    break
                for ref in fields(compiled, at, node).following():
                    push(ref, "r", step)
                if spec.call_style == "self" and spec.function is not None:
                    items = self.inference.reads(spec.function, compiled)
                    if self._dynamic(compiled, at, items):
                        outcome.refusal = _Refusal(None, None, spec.owner, step)
                        break
                    self._reads(compiled, at, items, step, push, down)
                continue
            # ``present()``'s sources: a guard is read, an alias, a guard or a view is
            # followed to its output, a selection to its selector and alternatives.
            if mode == "S" and node in compiled.selectors:
                push(node, "r", route)
            fields_ = fields(compiled, at, node)
            if fields_.guard is not None:
                push(fields_.guard, "r", route)
            kind = spec.kind
            if kind in ("alias", "guard", "view") and fields_.output is not None:
                push(fields_.output, "s", route)
            elif kind == "select" and fields_.selector is not None:
                push(fields_.selector, "r", route)
                for target in fields_.alternatives:
                    push(target, "s", route)
        if outcome.refusal is not None:
            outcome.exits.clear()
        return outcome

    def _dynamic(self, compiled: Compiled, at: Placement, items: Iterable[object]) -> bool:
        """Whether a method's reads are not fixed at ``at``: a dynamic read placed there."""
        return any(
            isinstance(item, _Dynamic) and self._placed(compiled, at, item.when) for item in items
        )

    def _reads(
        self,
        compiled: Compiled,
        at: Placement,
        items: Iterable[object],
        route: tuple[str, ...],
        push: Any,
        down: Any,
    ) -> None:
        """A method's inferred reads (``_reads._TemplateInference``) at ``at``."""
        for item in items:
            if isinstance(item, _Ref):
                mode: Mode = "S" if item.sources else "r"
                if not item.steps:
                    push(item.node, mode, route)
                elif self._placed(compiled, at, item.steps):
                    down(item.steps, item.node, mode, route)
            else:
                assert isinstance(item, (_Missing, _Dynamic))

    @staticmethod
    def _placed(compiled: Compiled, at: Placement, steps: Sequence[int]) -> bool:
        for index in steps:
            held = slot(compiled, index)
            if held.requires in at.never:
                return False
            compiled, at = held.compiled, held.placement
        return True

    def _enter(
        self,
        compiled: Compiled,
        at: Placement,
        origin: Origin,
        steps: Sequence[int],
        starts: Sequence[_Push],
    ) -> _Outcome:
        """Walk ``starts`` (records of the class ``steps`` reach) there, and what they reach
        out of it, up to ``compiled``'s class: what to read there, each with its way
        (``_Outcome.lifted``), or a refusal; and whether it re-entered an ``_Approx`` walk's
        slot."""
        held = slot(compiled, steps[0])
        if held.requires in at.never:
            return _Outcome()
        name = compiled.slots[steps[0]].name
        below, reentered = self._into(origin, steps[0])
        child, child_at = held.compiled, held.placement
        if len(steps) > 1:
            inner = self._enter(child, child_at, below, steps[1:], starts)
            reentered |= inner.reentered
            if inner.refusal is not None:
                return _Outcome(inner.refusal.under(name), reentered=reentered)
            outcome = self.walk(child, child_at, below, inner.lifted)
        else:
            outcome = self.walk(child, child_at, below, starts)
        reentered |= outcome.reentered
        if outcome.refusal is not None:
            return _Outcome(outcome.refusal.under(name), reentered=reentered)
        lifted: list[_Push] = [
            (ref, mode, _way(name, exit, outcome.exits[exit]))
            for exit in outcome.exits
            for ref, mode in lift(held, (exit,))
        ]
        return _Outcome(lifted=lifted, reentered=reentered)

    # -- per class

    def summary(self, compiled: Compiled, at: Placement) -> _Summary:
        """The domains of ``compiled``'s class's subtree at ``at`` (keys relative to it);
        found once."""
        cache = _cache(compiled, self.name)
        if at not in cache:
            # The classes below first, deepest first: a chain of placements may be deep.
            for below, below_at in self._below(compiled, at):
                held = _cache(below, self.name)
                if below_at not in held:
                    held[below_at] = self._summary(below, below_at)
        return cast(_Summary, cache[at])

    def _below(self, compiled: Compiled, at: Placement) -> list[tuple[Compiled, Placement]]:
        """``compiled`` at ``at`` and every class it places, each after those it places (none
        whose domains are known already)."""
        found: list[tuple[Compiled, Placement]] = []
        done: set[tuple[int, Placement]] = set()
        stack: list[tuple[Compiled, Placement, Iterator[int]]] = [
            (compiled, at, iter(range(len(compiled.slots))))
        ]
        while stack:
            current, current_at, slots = stack[-1]
            index = next(slots, None)
            if index is None:
                stack.pop()
                found.append((current, current_at))
                continue
            held = slot(current, index)
            key = (id(held.compiled), held.placement)
            if held.requires in current_at.never or key in done:
                continue
            done.add(key)
            if held.placement not in _cache(held.compiled, self.name):
                stack.append((held.compiled, held.placement, iter(range(len(held.compiled.slots)))))
        return found

    def _summary(self, compiled: Compiled, at: Placement) -> _Summary:
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
            if (
                domain is None
                or not is_decision(compiled, at, node)
                or excluded(compiled, at, node)
            ):
                continue
            named = set(domain.membership_inputs)
            starts: list[_Push] = [
                (compiled.ref(edge), "r", ())
                for _, edge in dict.fromkeys(
                    argument for argument in spec.domain_arguments if argument[0] in named
                )
            ]
            head = cast("int | Placing", compiled.ref(cast("Edge", spec.guard)))
            outcome = self.walk(compiled, at, (head,), starts, exempt=True)
            if outcome.refusal is not None:
                if refused(outcome.refusal.at(spec.key)):
                    return summary
                continue
            add(outcome.exits, spec.key, ("own", head, spec.key))
        # Each placed Space's, followed into its bindings here.
        for index in range(len(compiled.slots)):
            held = slot(compiled, index)
            if held.requires in at.never:
                continue
            name = compiled.slots[index].name
            below = self.summary(held.compiled, held.placement)
            for refusal in below.refusals:
                if refused(refusal.under(name)):
                    return summary
            for entry in below.entries:
                pushes: list[_Push] = [
                    (ref, mode, _way(name, exit, entry.exits[exit]))
                    for exit in entry.exits
                    for ref, mode in lift(held, (exit,))
                ]
                outcome = self.walk(compiled, at, _Approx(index), pushes)
                if outcome.reentered:
                    # It re-entered the subtree holding the Decisions: per Decision, exactly.
                    for place, key in _places(entry.origins, name):
                        exact = self.walk(compiled, at, (index, *place), pushes)
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
    refusals = _Domains().summary(compiled, ROOT).refusals
    if refusals:
        raise refusals[0].error()


def domain_refusals(compiled: Compiled) -> dict[str, DefinitionError]:
    """Every Decision the domain rule refuses in ``compiled``'s class as a root, by its key
    (a diagnostic: compiling stops at the first)."""
    refusals = _Domains(every=True).summary(compiled, ROOT).refusals
    return {cast(str, refusal.owner): refusal.error() for refusal in refusals}


__all__ = [
    "ROOT",
    "Placement",
    "check_domains",
    "check_placement",
    "check_placements",
    "domain_refusals",
    "order",
]
