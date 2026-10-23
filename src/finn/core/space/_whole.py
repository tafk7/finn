# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Questions about a whole design space, answered from compiled forms and scope records.

Some questions read every choice of a model, the cases not selected included: the
**flow order** (each choice after every choice its presence, its list or its Space's
facts read: ``PerClass.flow``, the order of ``inspection.decisions`` and ``states``,
and of a change's lists), and the **presence readers** (for each choice and fact, the
choices whose presence may read it: what a change may move). Each is answered here from
the classes' compiled forms and the scopes' records (``_scopes``): no member is evaluated.

**What each walks.** A choice's **upstream** is the first choices met on every way back
from its existence guards, its domain's inputs and the formals of its scope and of every
enclosing one: a way ends at a choice, a fact and a Space's validity (a view's Space is
not followed), and follows a node's dependencies and a method's inferred reads (a
dynamic method's: every node of its scope and of the scopes below it). A guard's
**reach** is every choice and fact on every way back from it, through choices too (a
dynamic method's: every choice and fact). A read through ``present()`` follows what
decides whether the value is supplied (its guards; an alias, a guard or a view to its
output; a selection to its selector and alternatives; a formal to its binding).

**Per class, summaries.** Each walk is done once per class (and placement:
``_compose.Placement``), from one start, over the records of its compiled form, and cached
with it: what it finds there (choices; for a reach, facts too), what it finds in the Spaces
it places (an edge into a placed Space continued in the child's compiled form by the
child's summary, and what that reaches out of the child followed into this class's
bindings), and what it reaches out of the class, which it reports as **exits**
(``_compose.Exit``: one of its formals, or the guard it is placed under). A dynamic
method's subtree is one summary per class too.

**Per scope, composition.** A summary's finds are addressed in the scope it is read at
(``ScopeRecord.reach`` and ``member_address``: a scope's record, no member read); each exit
is followed
into the placing scope's binding (``_compose.lift``: a literal, a default or an
unsupplied formal ends at the guard the scope is placed under; a reference continues
there), by that class's summaries, up to the root, where a formal is a fact. Composed per
scope and exit once.

That this is the line's answer, the line's algorithms over one table of every node,
is the tests' oracle (``core.space._whole_support``), on every Space class of the tests
and the kernels, and stage 7's chain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from ._compose import (
    ROOT,
    Exit,
    Mode,
    Placement,
    Ref,
    _classify,
    _Slot,
    excluded,
    fields,
    is_decision,
    lift,
    slot,
)
from ._reads import _Dynamic, _Found, _Ref, _TemplateInference
from .ir import Placed, Placing

if TYPE_CHECKING:
    from ._scopes import Address, ScopeRecord, Scopes
    from ._template import Compiled

# A choice in the flow order's algorithms: an address (a table's node id, in the tests).
N = TypeVar("N")

# The two walks: a choice's upstream, a guard's reach.
Walk = Literal["upstream", "reach"]
# A member a walk found: the slots from the class walked, and a record of the class they
# reach.
Step = tuple[tuple[int, ...], int]
# A walk's step to a reference, read as a mode.
_Push = Callable[[Ref, Mode], None]


@dataclass(frozen=True, slots=True)
class Summary:
    """What a walk found in a class and the Spaces it places (``found``), what it reached
    out of the class (``exits``), and whether it met a method whose reads are
    not fixed where the walk reads every choice and fact (``every``: a reach)."""

    found: frozenset[Step] = frozenset()
    exits: frozenset[Exit] = frozenset()
    every: bool = False


_NOTHING = Summary()


class _Union:
    """Summaries gathered into one."""

    __slots__ = ("found", "exits", "every")

    def __init__(self) -> None:
        self.found: set[Step] = set()
        self.exits: set[Exit] = set()
        self.every = False

    def add(self, summary: Summary, steps: tuple[int, ...] = ()) -> None:
        if steps:
            self.found.update(((*steps, *inner), node) for inner, node in summary.found)
        else:
            self.found |= summary.found
        self.exits |= summary.exits
        self.every |= summary.every

    def summary(self) -> Summary:
        if self.every:
            return Summary(every=True)
        return Summary(frozenset(self.found), frozenset(self.exits))


# -- per class ---------------------------------------------------------------------------------


class _Walks:
    """The summaries of one walk, per class and placement (cached on the compiled form:
    ``Compiled.composed``). ``inference`` answers each method's reads per class."""

    def __init__(self, walk: Walk, inference: _TemplateInference) -> None:
        self.walk = walk
        self.inference = inference
        self.name = f"whole.{walk}"

    def _memo(self, compiled: Compiled, at: Placement) -> dict[tuple[int, Mode], Summary]:
        held = cast(
            dict[Placement, dict[tuple[int, Mode], Summary]],
            compiled.composed.setdefault(self.name, {}),
        )
        return held.setdefault(at, {})

    def summary(self, compiled: Compiled, at: Placement, ref: Ref, mode: Mode) -> Summary:
        """The walk from ``ref`` (read as ``mode``) in ``compiled``'s class at ``at``; found
        once."""
        out = _classify(compiled, ref, mode)
        if out is not None:
            return Summary(exits=frozenset((out,)))
        node = cast(int, ref)
        memo = self._memo(compiled, at)
        found = memo.get((node, mode))
        if found is None:
            found = memo[(node, mode)] = self._dfs(compiled, at, ((node, mode),), memo)
        return found

    def gathered(
        self, compiled: Compiled, at: Placement, starts: Iterable[tuple[Ref, Mode]]
    ) -> Summary:
        """The walk from each of ``starts``, as one (a walk's finds are a union)."""
        union = _Union()
        for ref, mode in starts:
            union.add(self.summary(compiled, at, ref, mode))
        return union.summary()

    def subtree(self, compiled: Compiled, at: Placement) -> Summary:
        """The upstream walk from every record of the scope ``compiled``'s class is placed
        as at ``at`` and of the scopes below it (what a dynamic method of that scope may
        read): its own members and the records it writes for the Spaces it places (their
        formals' bindings), and each placed Space's subtree. Found once."""
        held = cast(dict[Placement, Summary], compiled.composed.setdefault("whole.subtree", {}))
        found = held.get(at)
        if found is None:
            union = _Union()
            starts: list[tuple[Ref, Mode]] = [
                (record, "r") for record in range(len(compiled.records))
            ]
            for index in range(len(compiled.slots)):
                child = slot(compiled, index)
                if child.requires in at.never:
                    continue
                below = self.subtree(child.compiled, child.placement)
                union.add(Summary(below.found), (index,))
                starts.extend(item for exit in below.exits for item in lift(child, (exit,)))
            union.add(self._dfs(compiled, at, starts, None, subtree=True))
            found = held[at] = union.summary()
        return found

    def _dfs(
        self,
        compiled: Compiled,
        at: Placement,
        starts: Sequence[tuple[Ref, Mode]],
        memo: Mapping[tuple[int, Mode], Summary] | None,
        *,
        subtree: bool = False,
    ) -> Summary:
        """The walk from ``starts`` over ``compiled``'s records at ``at``. A record whose
        walk is known (``memo``) contributes it whole. ``subtree``: the walk of the scope's
        subtree itself, to which a dynamic method of its own adds nothing."""
        upstream = self.walk == "upstream"
        union = _Union()
        seen: set[tuple[int, Mode]] = set()
        pending: list[tuple[int, Mode]] = []

        def push(ref: Ref, mode: Mode) -> None:
            out = _classify(compiled, ref, mode)
            if out is not None:
                union.exits.add(out)
                return
            node = cast("int | Placed", ref)
            if excluded(compiled, at, node):
                return
            if isinstance(node, Placed):
                found, lifted = self._down(compiled, at, node.path, node.member, mode)
                union.add(found)
                for item in lifted:
                    push(*item)
                return
            if (node, mode) in seen:
                return
            seen.add((node, mode))
            known = None if memo is None else memo.get((node, mode))
            if known is not None:
                union.add(known)
                return
            pending.append((node, mode))

        for ref, mode in starts:
            push(ref, mode)
        while pending:
            if union.every:
                return Summary(every=True)
            node, mode = pending.pop()
            spec = compiled.records[node]
            if mode != "r":
                # What ``present()`` reads of a value: its guard; an alias, a guard or a
                # view's output; a selection's selector and alternatives; a selector
                # itself where it is the value asked of.
                if mode == "S" and node in compiled.selectors:
                    push(node, "r")
                read = fields(compiled, at, node)
                if read.guard is not None:
                    push(read.guard, "r")
                if spec.kind in ("alias", "guard", "view") and read.output is not None:
                    push(read.output, "s")
                elif spec.kind == "select" and read.selector is not None:
                    push(read.selector, "r")
                    for target in read.alternatives:
                        push(target, "s")
                continue
            if is_decision(compiled, at, node):
                union.found.add(((), node))
                if upstream:
                    continue
            elif upstream and spec.kind == "validity":
                continue
            for ref in fields(compiled, at, node).dependencies():
                push(ref, "r")
            if spec.call_style == "self" and spec.function is not None:
                items = self.inference.reads(spec.function, compiled)
                if self._dynamic(compiled, at, items):
                    if not upstream:
                        return Summary(every=True)
                    if not subtree:
                        union.add(self.subtree(compiled, at))
                    continue
                self._reads(compiled, at, items, union, push)
        return union.summary()

    def _down(
        self,
        compiled: Compiled,
        at: Placement,
        steps: Sequence[int],
        target: int | Placing,
        mode: Mode,
    ) -> tuple[Summary, list[tuple[int | Placing, Mode]]]:
        """The walk from ``target``, a record of the class ``steps`` reach from
        ``compiled``'s (its slots), there: what it finds below (from ``compiled``'s class),
        and what it reaches out of the Space at ``steps[0]``, lifted into ``compiled``'s
        records to walk on there."""
        held = slot(compiled, steps[0])
        if held.requires in at.never:
            return _NOTHING, []
        child, child_at = held.compiled, held.placement
        union = _Union()
        if len(steps) > 1:
            found, lifted = self._down(child, child_at, steps[1:], target, mode)
            union.add(found)
            union.add(self.gathered(child, child_at, lifted))
        else:
            union.add(self.summary(child, child_at, target, mode))
        below = union.summary()
        if below.every:
            return Summary(every=True), []
        found = Summary(frozenset(((steps[0], *inner), node) for inner, node in below.found))
        return found, [item for exit in below.exits for item in lift(held, (exit,))]

    def _dynamic(self, compiled: Compiled, at: Placement, items: _Found) -> bool:
        """Whether a method's reads are not fixed at ``at`` (as ``_compose``'s domain rule
        reads them): a dynamic read placed there."""
        return any(
            isinstance(item, _Dynamic) and _placed(compiled, at, item.when) for item in items
        )

    def _reads(
        self,
        compiled: Compiled,
        at: Placement,
        items: _Found,
        union: _Union,
        push: _Push,
    ) -> None:
        """A method's inferred reads at ``at``, walked on (``_reads._Mapping``'s mapping, by
        class): a record, read as ``present()`` reads it where its source asks presence."""
        for item in items:
            if isinstance(item, _Ref):
                mode: Mode = "S" if item.sources else "r"
                if not item.steps:
                    push(item.node, mode)
                elif _placed(compiled, at, item.steps):
                    found, lifted = self._down(compiled, at, item.steps, item.node, mode)
                    union.add(found)
                    for lifted_item in lifted:
                        push(*lifted_item)


def _placed(compiled: Compiled, at: Placement, steps: Sequence[int]) -> bool:
    """Whether the slots ``steps`` are placed at ``at`` (none a candidate of a Decision over
    nodes that never applies)."""
    for index in steps:
        held = slot(compiled, index)
        if held.requires in at.never:
            return False
        compiled, at = held.compiled, held.placement
    return True


# -- the flow order --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Flow:
    """The model's choices in the flow order (``order``), and its loops (``loops``: each
    set of choices reading one another, in declaration order)."""

    order: tuple[Any, ...]
    loops: tuple[tuple[Any, ...], ...]


def flow_order(decisions: Sequence[N], reading: Mapping[N, frozenset[N]]) -> Flow:
    """The choices ``decisions`` (in declaration order) in the flow order, each after the
    choices it reads (``reading``), ties in declaration order. Kahn's algorithm orders them,
    the next ready choice the first declared. Choices that read one another (a loop) are
    taken as one, where the first of them declared would be, and in declaration order
    among themselves."""
    ranks = {decision: position for position, decision in enumerate(decisions)}
    loops = _loops(decisions, reading, ranks)
    unit = {member: loop[0] for loop in loops for member in loop}
    members = {loop[0]: loop for loop in loops}
    # The graph over units (a loop's first member stands for it): each unit's readers,
    # and how many units it waits on.
    readers: dict[N, set[N]] = {}
    waiting: dict[N, set[N]] = {}
    for decision in decisions:
        own = unit.get(decision, decision)
        waits = waiting.setdefault(own, set())
        for source in reading[decision]:
            source = unit.get(source, source)
            if source != own:
                waits.add(source)
                readers.setdefault(source, set()).add(own)
    count = {node: len(waits) for node, waits in waiting.items()}
    ready = [(ranks[node], node) for node, waits in count.items() if not waits]
    heapify(ready)
    order: list[N] = []
    while ready:
        _, node = heappop(ready)
        order.extend(members.get(node, (node,)))
        for reader in readers.get(node, ()):
            count[reader] -= 1
            if not count[reader]:
                heappush(ready, (ranks[reader], reader))
    return Flow(tuple(order), loops)


def _loops(
    decisions: Sequence[N],
    reading: Mapping[N, frozenset[N]],
    ranks: Mapping[N, int],
) -> tuple[tuple[N, ...], ...]:
    """The sets of choices that read one another (strongly connected, more than one), each
    in declaration order (``ranks``), the sets by their first member's (Tarjan's
    algorithm, without recursion; each choice's reads taken in declaration order)."""
    index: dict[N, int] = {}
    low: dict[N, int] = {}
    stack: list[N] = []
    on_stack: set[N] = set()
    found: list[tuple[N, ...]] = []

    def sources(node: N) -> Iterator[N]:
        return iter(sorted(reading[node], key=ranks.__getitem__))

    for root in decisions:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        frames = [(root, sources(root))]
        while frames:
            node, pending = frames[-1]
            source = next(pending, None)
            if source is not None:
                if source not in index:
                    index[source] = low[source] = len(index)
                    stack.append(source)
                    on_stack.add(source)
                    frames.append((source, sources(source)))
                elif source in on_stack:
                    low[node] = min(low[node], index[source])
                continue
            frames.pop()
            if frames:
                parent = frames[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[N] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(tuple(sorted(component, key=ranks.__getitem__)))
    return tuple(sorted(found, key=lambda loop: ranks[loop[0]]))


# -- per scope ---------------------------------------------------------------------------------


class Whole:
    """A model's answers to the questions about its whole design space, composed from its
    classes' summaries per scope (see the module docstring). Its scope records' lock is held
    while it is asked."""

    def __init__(self, scopes: Scopes) -> None:
        self.scopes = scopes
        inference = _TemplateInference()
        self.walks: dict[Walk, _Walks] = {
            "upstream": _Walks("upstream", inference),
            "reach": _Walks("reach", inference),
        }
        self._placements: dict[ScopeRecord, Placement] = {}
        self._up: dict[tuple[ScopeRecord, Exit, Walk], tuple[frozenset[Address], bool]] = {}
        self._formals: dict[ScopeRecord, frozenset[Address]] = {}
        self._flow: Flow | None = None
        self._positions: dict[Address, int] | None = None
        self._readers: tuple[dict[Address, tuple[Address, ...]], tuple[Address, ...]] | None = None

    def placement(self, record: ScopeRecord) -> Placement:
        """How ``record``'s class is placed there (``_compose.Placement``)."""
        found = self._placements.get(record)
        if found is None:
            found = ROOT if record.parent is None else self._slot(record).placement
            self._placements[record] = found
        return found

    @staticmethod
    def _slot(record: ScopeRecord) -> _Slot:
        """The slot ``record``'s scope is placed at, of its parent's compiled form."""
        parent = cast("ScopeRecord", record.parent)
        return slot(parent.compiled, cast(int, record.slot))

    # -- composition

    def reached(
        self, record: ScopeRecord, starts: Iterable[tuple[Ref, Mode]], walk: Walk
    ) -> tuple[frozenset[Address], bool]:
        """What the walk ``walk`` from ``starts`` (refs of ``record``'s compiled form) finds
        in the design space, by address; and whether it reads every choice and fact."""
        summary = self.walks[walk].gathered(record.compiled, self.placement(record), starts)
        return self._composed(record, summary, walk)

    def _composed(
        self, record: ScopeRecord, summary: Summary, walk: Walk
    ) -> tuple[frozenset[Address], bool]:
        if summary.every:
            return frozenset(), True
        found: set[Address] = set()
        for steps, node in summary.found:
            at = record.reach(steps)
            index = None if at is None else at.member_address(node)
            if index is not None:
                found.add(index)
        for exit in summary.exits:
            more, every = self._exit(record, exit, walk)
            if every:
                return frozenset(), True
            found |= more
        return frozenset(found), False

    def _exit(self, record: ScopeRecord, exit: Exit, walk: Walk) -> tuple[frozenset[Address], bool]:
        """What a walk reaching ``exit`` out of ``record``'s class finds above it: at the root
        a formal is a fact; else the exit followed into the placing class."""
        key = (record, exit, walk)
        held = self._up.get(key)
        if held is not None:
            return held
        parent = record.parent
        if parent is None:
            found: tuple[frozenset[Address], bool] = (frozenset(), False)
            if walk == "reach" and exit[0] == "formal":
                fact = record.canonical(record.compiled.formals[exit[1]])
                found = (frozenset(() if fact is None else (fact,)), False)
        else:
            found = self.reached(parent, lift(self._slot(record), (exit,)), walk)
        self._up[key] = found
        return found

    @staticmethod
    def _start(index: Address) -> tuple[ScopeRecord, Ref]:
        """Where a walk from ``index`` starts: its scope and member, or, for a formal its
        placing class binds, the exit it is there."""
        record, member = index.held
        name = record.compiled.formal_names.get(member)
        if name is not None and record.parent is not None:
            return record, ("formal", name, "r")
        return record, member

    # -- the flow order

    def formals(self, scope: ScopeRecord) -> frozenset[Address]:
        """The choices the formals of ``scope`` and of every enclosing scope read."""
        found = self._formals.get(scope)
        if found is None:
            own: set[Address] = set()
            for name in scope.compiled.formals:
                own |= self._exit(scope, ("formal", name, "r"), "upstream")[0]
            outer = frozenset() if scope.parent is None else self.formals(scope.parent)
            found = self._formals[scope] = frozenset(own) | outer
        return found

    def upstream(self, decision: Address) -> frozenset[Address]:
        """The choices ``decision``'s presence, its list or its Space's facts read: the
        first on every way back from its existence guards, its domain's inputs and its
        scopes' formals."""
        record, member = decision
        compiled = record.compiled
        spec = compiled.records[member]
        starts: list[tuple[Ref, Mode]] = [
            (compiled.ref(edge), "r") for _, edge in spec.domain_arguments
        ]
        if spec.guard is not None:
            starts.append((compiled.ref(spec.guard), "r"))
        found = self.reached(record, starts, "upstream")[0] | self.formals(record)
        return found - {decision}

    def flow(self) -> Flow:
        """The model's choices in the flow order (``flow_order``), with its loops."""
        if self._flow is None:
            with self.scopes.lock:
                decisions = self.scopes.decisions
                reading = {decision: self.upstream(decision) for decision in decisions}
                self._flow = flow_order(decisions, reading)
        return self._flow

    def position(self, decision: Address) -> int:
        """A choice's place in the flow order (``KeyError`` for anything else)."""
        if self._positions is None:
            order = self.flow().order
            self._positions = {index: position for position, index in enumerate(order)}
        return self._positions[decision]

    # -- presence readers

    def readers(self, source: Address) -> tuple[Address, ...]:
        """The choices whose presence may read the choice or fact ``source``: those whose
        guards (their own and every enclosing one's) may read it, through what each guard
        reaches; a guard reaching a dynamic method may read every choice and fact. In
        declaration order, those of every source last."""
        if self._readers is None:
            with self.scopes.lock:
                self._readers = self._all_readers()
        specific, everywhere = self._readers
        return (*specific.get(source, ()), *everywhere)

    def _all_readers(self) -> tuple[dict[Address, tuple[Address, ...]], tuple[Address, ...]]:
        reaches: dict[Address, tuple[frozenset[Address], bool]] = {}
        specific: dict[Address, list[Address]] = {}
        everywhere: list[Address] = []
        for decision in self.scopes.decisions:
            guard = decision.guard
            if guard is None:
                continue
            found = reaches.get(guard)
            if found is None:
                record, ref = self._start(guard)
                found = reaches[guard] = self.reached(record, ((ref, "r"),), "reach")
            sources, every = found
            if every:
                everywhere.append(decision)
                continue
            for source in sources:
                specific.setdefault(source, []).append(decision)
        return {source: tuple(found) for source, found in specific.items()}, tuple(everywhere)


__all__ = ["Flow", "Summary", "Whole", "flow_order"]
