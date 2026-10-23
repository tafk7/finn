# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What crosses a slot, checked per template and composed per placement.

A template is compiled once (``_template``) and placed as scopes (``_structure``). Three
checks read across the slots a template places, and each is composed here from what one
template says of itself, never by copying a placement:

- **Cycles** (``order``). A template's dependency order runs over its own nodes, with an
  edge from each import to what its target reaches in the child: the formals it reaches
  by structural edges (each followed to the placing template's binding: its node, or the
  guard the scope is placed under for a literal or a default) and the guard the child is
  placed under, through any depth (``exits``). A
  cycle through a child is found once, in the placing class's template, and named by
  that template's keys: its own nodes, its imports and bindings, and the nodes of the
  Spaces it places on the way, as the one table of every placement named them, relative
  to the template (a cycle at two levels sharing nodes: the inner template's is found
  first, as that template is linked first).
- **Placements** (``check_placement``). Where a placing template leaves a formal
  unsupplied, a Decision over nodes guarded by ``supplied`` of it never applies: what
  exists only where it applies is not placed, and a node that still reads it is
  refused (``reference-not-placed``), as the placing template is linked.
- **The domain rule** (``check_domains``): a domain's membership reads facts only. Per
  template, each domain's walk follows what its membership inputs may read (arguments,
  alternatives, a view's validity and a validity's pieces, outputs, selectors, guards,
  a method's inferred reads and what ``present()`` reads), skipping the guards that
  decide whether the Decision exists; it stops at a choice (``domain-reads-choice``),
  and at the template's own formals and the guard it is placed under, which it reports
  as reached. A child's are followed into the placing template's bindings, with the
  guards above the slot as existence guards of the child's Decision too, and so up to
  the root, per placement chain. A template records, for its own formals, which are reached by any
  domain in its subtree (``_Entry``), once per placement signature.

A template is read **at a placement** (``Placement``): the Decisions over nodes of it that
never apply there. A placement
changes a template only where such a Decision never applies: its selector is a
constant, what is read through it has no alternatives, and its candidates, what exists
only where it applies and the validity pieces of both are not placed.

Each walk is that of the design space the template is placed in: the composition is
exact. Where a walk back up re-enters the subtree holding the Decision it started from,
its existence guards inside that subtree are known only per Decision: the walk is done
once with none of them and, only where it re-entered, again per Decision (``_Approx``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ._collapse import forwards
from ._ordering import cyclic, dependency_order
from ._reads import _Dynamic, _Missing, _Ref, _TemplateInference
from ._table import Binding, Table, member_key
from .errors import DefinitionError

if TYPE_CHECKING:
    from ._template import Template

_OWNER = "space.compose"


# -- placements ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Placement:
    """A template read at a placement: the Decisions over nodes of it that never apply
    there (guarded by ``supplied`` of a formal the placing template leaves unsupplied)."""

    never: frozenset[int] = frozenset()


ROOT = Placement()


def placement(template: Template, bindings: Mapping[str, Binding]) -> Placement:
    never = frozenset(
        choice.index
        for choice in template.choices
        if choice.never_formal is not None and bindings[choice.never_formal].kind == "unsupplied"
    )
    return Placement(never)


@dataclass(frozen=True, slots=True)
class _Slot:
    """A Space a template places (a placed stub): its class's template and placement, the
    bindings of its formals, the node it is placed under, and the Decision over nodes whose
    candidate it is (where that may never apply)."""

    template: Template
    placement: Placement
    bindings: Mapping[str, Binding]
    guard: int
    requires: int | None


def _cache(template: Template, name: str) -> dict[Any, Any]:
    return cast(dict[Any, Any], template.composed.setdefault(name, {}))


def slot(template: Template, stub: int) -> _Slot:
    """A placed stub of ``template``, as a slot."""
    cache = _cache(template, "slot")
    found = cache.get(stub)
    if found is None:
        from ._template import link_template  # noqa: PLC0415 - a cycle

        scope = template.scopes[stub]
        child = link_template(scope.space_type)
        found = cache[stub] = _Slot(
            child,
            placement(child, scope.bindings),
            scope.bindings,
            cast(int, scope.guard),
            scope.requires,
        )
    return cast(_Slot, found)


def _table_slot(table: Table, stub: int) -> _Slot:
    """A placed stub of a table being linked, as a slot."""
    draft = table.drafts[stub]
    held = table.stub(stub)
    return _Slot(
        held.template,
        placement(held.template, held.bindings),
        held.bindings,
        cast(int, draft.guard),
        held.requires,
    )


def path(template: Template, stub: int) -> tuple[int, ...]:
    """A stub's placed stubs from the template down: a placed stub, then for a nested one
    each slot in its parent's template."""
    steps: list[int] = []
    while True:
        scope = template.scopes[stub]
        if scope.parent == 0:
            steps.append(stub)
            return tuple(reversed(steps))
        steps.append(cast(int, scope.slot))
        stub = cast(int, scope.parent)


def _table_path(table: Table, stub: int) -> tuple[int, ...]:
    steps: list[int] = []
    while True:
        draft = table.drafts[stub]
        if draft.parent == 0:
            steps.append(stub)
            return tuple(reversed(steps))
        steps.append(cast(int, table.stub(stub).slot))
        stub = cast(int, draft.parent)


# -- a template at a placement -----------------------------------------------------------------

# A reference out of a template: a formal's (``formal``, name, mode), or the guard it is
# placed under (``guard``, mode). A mode: ``r`` reached, ``s`` followed as ``present()``
# follows a value's sources, ``S`` the first such step.
Exit: TypeAlias = tuple[str, ...]
Mode = Literal["r", "s", "S"]
Ref: TypeAlias = "int | Exit"


def _dropped(template: Template, at: Placement, node: int) -> bool:
    """A node of a stub below a candidate of a Decision over nodes that never applies."""
    scope = template.nodes[node].scope
    return scope != 0 and template.scopes[scope].requires in at.never


def excluded(template: Template, at: Placement, node: int) -> bool:
    """Whether ``at`` places no node for ``node``: it is below a candidate of a Decision
    over nodes that never applies there, or exists only where one applies. A selection is
    placed where something reads it."""
    if not at.never:
        return False
    if _dropped(template, at, node):
        return True
    if node in template.index.selections or node in template.stub_selects:
        return False
    return template.requires.get(node) in at.never


def is_decision(template: Template, at: Placement, node: int) -> bool:
    """A Decision at ``at`` (a selector that never applies is a constant there)."""
    return template.nodes[node].kind == "decision" and template.selectors.get(node) not in at.never


def _never_selection(template: Template, at: Placement, node: int) -> bool:
    selection = template.index.selections.get(node)
    return selection is not None and selection[0] in at.never


def pieces(template: Template, at: Placement) -> list[Ref]:
    """A validity node's pieces at ``at``: its constraints and Decisions, then each placed
    Space's validity. A formal it lists stands for nothing here (no binding is a Decision
    of its own)."""
    own: list[Ref] = []
    placed: list[Ref] = []
    for piece in template.nodes[template.validity].constraints:
        if template.nodes[piece].scope != 0:
            if not _dropped(template, at, piece):
                placed.append(piece)
            continue
        if piece in template.formal_names or excluded(template, at, piece):
            continue
        if template.selectors.get(piece) not in at.never:
            own.append(piece)
    return [*own, *placed]


@dataclass(frozen=True, slots=True)
class _Fields:
    """A node's references at a placement, by role."""

    arguments: tuple[int, ...]
    domain_arguments: tuple[int, ...]
    guard: int | None
    output: int | None
    selector: int | None
    constraints: tuple[Ref, ...]
    alternatives: tuple[int, ...]

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


def fields(template: Template, at: Placement, node: int) -> _Fields:
    spec = template.nodes[node]
    alternatives = tuple(target for _, target in spec.alternatives)
    constraints: tuple[Ref, ...] = spec.constraints
    if node == template.validity:
        constraints = tuple(pieces(template, at))
    elif at.never and _never_selection(template, at, node):
        alternatives = ()  # read through a Decision over nodes that never applies here
    elif at.never and spec.kind == "members":
        alternatives = tuple(
            target for target in alternatives if not _dropped(template, at, target)
        )
    return _Fields(
        tuple(argument.node for argument in spec.arguments),
        tuple(argument.node for argument in spec.domain_arguments),
        spec.guard,
        spec.output,
        spec.selector,
        constraints,
        alternatives,
    )


def lift(slot_: _Slot, exits: Iterable[Exit]) -> Iterator[tuple[int, Mode]]:
    """What a placed Space's references out stand for in its placing template: a formal,
    its binding's node (or, for a literal or a default, the guard the scope is placed
    under: a constant's or an unsupplied formal's only edge); the guard it is placed
    under."""
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


def _classify(template: Template, ref: Ref, mode: Mode) -> Exit | None:
    """``ref``'s reference out of ``template``, if it is one."""
    if isinstance(ref, tuple):
        return ref
    if ref == template.placement_guard:
        return ("guard", "s" if mode == "S" else mode)
    formal = template.formal_names.get(ref)
    if formal is not None:
        return ("formal", formal, mode)
    return None


# -- cycles ------------------------------------------------------------------------------------


def exits(template: Template, at: Placement, node: int) -> frozenset[Exit]:
    """What ``node`` reaches out of ``template`` at ``at`` by structural edges (its
    dependencies, through any depth of the Spaces it places). A template's graph is
    acyclic once linked, so each node's are found once."""
    memo = cast(dict[int, frozenset[Exit]], _cache(template, "exits").setdefault(at, {}))
    if node in memo:
        return memo[node]
    stack: list[tuple[int, Iterator[Ref]]] = [(node, iter(_successors(template, at, node)))]
    open_ = {node}
    found: dict[int, set[Exit]] = {node: set()}
    while stack:
        current, successors = stack[-1]
        ref = next(successors, None)
        if ref is None:
            stack.pop()
            open_.discard(current)
            memo[current] = frozenset(found.pop(current))
            if stack:
                found[stack[-1][0]] |= memo[current]
            continue
        out = _classify(template, ref, "r")
        if out is not None:
            found[current].add(out)
            continue
        ref = cast(int, ref)
        if excluded(template, at, ref):
            continue
        if ref in memo:
            found[current] |= memo[ref]
        elif ref not in open_:
            open_.add(ref)
            found[ref] = set()
            stack.append((ref, iter(_successors(template, at, ref))))
    return memo[node]


def _successors(template: Template, at: Placement, node: int) -> list[Ref]:
    """A node's dependencies at ``at``; an import's, what its target reaches in the child,
    lifted into this template."""
    if template.nodes[node].kind == "import":
        return [ref for ref, _ in _descend(template, at, node)]
    return fields(template, at, node).dependencies()


def _descend(template: Template, at: Placement, node: int) -> list[tuple[int, Mode]]:
    stub = template.nodes[node].scope
    return _through(template, path(template, stub), template.imports[node])


def _reaching(template: Template, at: Placement, ref: Ref) -> frozenset[Exit]:
    """What a reference of ``template`` reaches out of it: itself, if it is one."""
    out = _classify(template, ref, "r")
    if out is not None:
        return frozenset((out,))
    return exits(template, at, cast(int, ref))


def _through(template: Template, steps: Sequence[int], target: int) -> list[tuple[int, Mode]]:
    """What ``target`` (a node of the template ``steps`` reach) reaches out of it, followed
    up to ``template``."""
    return _up(slot(template, steps[0]), steps[1:], target)


def _up(held: _Slot, steps: Sequence[int], target: int) -> list[tuple[int, Mode]]:
    """What ``target`` (a node of the template ``steps`` reach from ``held``'s) reaches out
    of it, followed up through ``held`` into its placing template."""
    child, at = held.template, held.placement
    reached: set[Exit] = set()
    if steps:
        for ref, _ in _through(child, steps, target):
            reached |= _reaching(child, at, ref)
    else:
        reached |= _reaching(child, at, target)
    return list(lift(held, sorted(reached)))


def order(table: Table) -> tuple[int, ...]:
    """The dependency order of a table being linked, with each import's edges to what its
    target reaches out of the child (``exits``). A cycle is ``cyclic-dependency``, named by
    the table's keys: its own nodes, its imports and bindings, and the nodes of the Spaces
    it places on the way (``_named``)."""
    dependencies: list[tuple[int, ...]] = []
    for node in table.nodes:
        found = list(node.dependencies)
        if node.kind == "import":
            steps = _table_path(table, node.scope)
            held = _table_slot(table, steps[0])
            found.extend(ref for ref, _ in _up(held, steps[1:], table.imports[node.index]))
        dependencies.append(tuple(dict.fromkeys(found)))
    names = tuple(node.key for node in table.nodes)
    try:
        return dependency_order(tuple(dependencies), names)
    except DefinitionError as error:
        index = {name: position for position, name in enumerate(names)}
        cycles: list[tuple[str, ...]] = []
        for finding in error.findings:
            members = cast(tuple[str, ...], dict(finding.details)["members"])
            nodes = {index[member] for member in members}
            named = set(members)
            for member in nodes:
                if table.nodes[member].kind == "import":
                    steps = _table_path(table, table.nodes[member].scope)
                    held = _table_slot(table, steps[0])
                    named |= _named(
                        held,
                        steps[1:],
                        table.imports[member],
                        table.drafts[steps[0]].name,
                        lambda ref: ref in nodes,
                    )
            cycles.append(tuple(sorted(named)))
        raise cyclic(cycles) from None


def _named(
    held: _Slot,
    steps: Sequence[int],
    target: int,
    name: str,
    wanted: Callable[[Ref], bool],
) -> set[str]:
    """The keys (under ``name``, the placed scope's) of the nodes on the ways from
    ``target`` (a node of the template ``steps`` reach from ``held``'s) back to a reference
    of the placing template that ``wanted`` accepts: what a cycle through an import passes
    through inside the Spaces placed, as the one table of every placement named it."""
    child, at = held.template, held.placement

    def reaches(ref: Ref) -> bool:
        return any(
            wanted(up) for exit in _reaching(child, at, ref) for up, _ in lift(held, (exit,))
        )

    found: set[str] = set()
    starts: list[int] = []
    if steps:
        below = slot(child, steps[0])
        found |= _named(
            below, steps[1:], target, member_key(name, child.scopes[steps[0]].name), reaches
        )
        starts = [ref for ref, _ in _up(below, steps[1:], target) if reaches(ref)]
    elif reaches(target):
        starts = [target]
    formals: set[str] = set()
    seen = set(starts)
    pending = list(starts)
    while pending:
        node = pending.pop()
        if _classify(child, node, "r") is not None:
            continue
        spec = child.nodes[node]
        if spec.kind == "import":
            inner = path(child, spec.scope)
            found |= _named(
                slot(child, inner[0]),
                inner[1:],
                child.imports[node],
                member_key(name, child.scopes[inner[0]].name),
                reaches,
            )
        else:
            found.add(member_key(name, spec.key))
        for ref in _successors(child, at, node):
            out = _classify(child, ref, "r")
            if out is not None:
                if out[0] == "formal" and reaches(out):
                    formals.add(out[1])
                continue
            if ref not in seen and not excluded(child, at, cast(int, ref)) and reaches(ref):
                seen.add(cast(int, ref))
                pending.append(cast(int, ref))
    # A formal bound to a literal or a default is a constant node of the placed scope.
    found.update(
        member_key(name, formal) for formal in formals if held.bindings[formal].node is None
    )
    return found


# -- placements ------------------------------------------------------------------------------


def check_placements(table: Table) -> None:
    """Each Space a table being linked places where a Decision over nodes of it never
    applies: no node it places there reads what is not placed (``check_placement``)."""
    for draft in table.drafts:
        if draft.parent != 0:
            continue
        held = _table_slot(table, draft.index)
        if held.placement.never:
            check_placement(held.template, held.placement, table.local_name(0, draft.index))


def check_placement(template: Template, at: Placement, name: str) -> None:
    """No node of ``template`` placed at ``at`` (at scope ``name``) reads a node it does not
    place: ``reference-not-placed`` (as the line's one table of every placement refused
    it, when lowering the reader there)."""
    for node in range(len(template.nodes)):
        if template.nodes[node].kind == "import" or excluded(template, at, node):
            continue
        for ref in fields(template, at, node).dependencies():
            if isinstance(ref, int) and excluded(template, at, ref):
                raise _not_placed(template, at, name, node, ref)


def _not_placed(
    template: Template, at: Placement, name: str, reader: int, target: int
) -> DefinitionError:
    owner = member_key(name, template.nodes[reader].owner)
    scope = template.nodes[target].scope
    if scope != 0 and _dropped(template, at, target):
        # Below a candidate that is not placed: the candidate's node declaration.
        while template.scopes[scope].parent != 0:
            scope = cast(int, template.scopes[scope].parent)
        record = template.scopes[scope].record
        describe = getattr(record, "describe", None)
        text = describe() if callable(describe) else repr(record)
        detail = f"{owner}: {text} is not placed in {name or '<root>'}"
    else:
        detail = f"{owner}: {member_key(name, template.nodes[target].key)} is not placed"
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


# Where the Decision a walk started from is, from the template walked: the placed stubs down
# to its template, then its guard there (``_Path``).
_Path: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Approx:
    """A walk up from below ``stub``, the guards the Decision's scope is placed under in
    this template skipped; where it re-enters ``stub`` it is walked with no existence
    guards (it reaches at least what the Decision's walk reaches) and marked."""

    stub: int


Origin: TypeAlias = "_Path | _Approx | None"


@dataclass(slots=True)
class _Outcome:
    """A walk's result: a refusal, or what it reached out of its template (each with its
    way), and whether it re-entered an ``_Approx`` walk's stub (then the refusal or what it
    reached is a superset of the Decision's own)."""

    refusal: _Refusal | None = None
    exits: dict[Exit, tuple[str, ...]] = field(default_factory=dict)
    reentered: bool = False
    # ``_enter``'s: what the exits stand for in the template above, to read there.
    lifted: list[_Push] = field(default_factory=list)


@dataclass(slots=True)
class _Entry:
    """What the domains of a template's subtree reach out of it, for one set of
    references out: each with its way, the first Decision's key, and every Decision's place
    (``("own", guard, key)``; ``("via", stub, entry, name)``: every Decision of a child's
    entry below that stub, named ``name`` here; ``("path", path, key)``: one, by its
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
            stub, entry, local = origin[1], cast(_Entry, origin[2]), origin[3]
            for place, key in _places(entry.origins, member_key(name, local)):
                yield (stub, *place), key


@dataclass(slots=True)
class _Summary:
    """The domains of a template's subtree at a placement: what they reach out of it
    (``entries``), and the refusals found (one, unless every refusal is asked for: then
    every Decision refused, by its key)."""

    entries: list[_Entry] = field(default_factory=list)
    refusals: list[_Refusal] = field(default_factory=list)


_Push: TypeAlias = "tuple[Ref, Mode, tuple[str, ...]]"


def _way(name: str, exit: Exit, way: tuple[str, ...]) -> tuple[str, ...]:
    """A way out of a placed Space (at ``name``), under the placing template's keys. A way
    out through a formal keeps the formal as a step: the hop across the slot, from the
    child's formal to what the placing template binds it to, is what its author needs to
    see (a forwarding alias inside one template is no step)."""
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
    def _chain(template: Template, head: int | None) -> frozenset[int]:
        """``head`` and every guard it is evaluated under, in ``template``."""
        cache = _cache(template, "chain")
        found = cache.get(head)
        if found is None:
            nodes: list[int] = []
            current = head
            while current is not None:
                nodes.append(current)
                current = template.nodes[current].guard
            found = cache[head] = frozenset(nodes)
        return cast(frozenset[int], found)

    def _skip(self, template: Template, origin: Origin) -> frozenset[int]:
        if origin is None:
            return frozenset()
        if isinstance(origin, _Approx):
            return self._chain(template, template.scopes[origin.stub].guard)
        if len(origin) == 1:
            return self._chain(template, origin[0])
        return self._chain(template, template.scopes[origin[0]].guard)

    @staticmethod
    def _into(origin: Origin, stub: int) -> tuple[Origin, bool]:
        """The origin below ``stub``, and whether an approximate walk re-entered there."""
        if isinstance(origin, _Approx):
            return None, origin.stub == stub
        if origin is not None and len(origin) > 1 and origin[0] == stub:
            return origin[1:], False
        return None, False

    # -- the walk

    def walk(
        self,
        template: Template,
        at: Placement,
        origin: Origin,
        starts: Sequence[_Push],
        *,
        exempt: bool = False,
    ) -> _Outcome:
        """Everything ``starts`` may read in ``template`` at ``at``, the existence guards of
        ``origin`` skipped (``exempt``: the starts themselves are a domain's inputs, read
        whatever guards them); see the module docstring."""
        if origin is None and len(starts) == 1 and not exempt:
            return self._single(template, at, starts[0])
        return self._walk(template, at, origin, starts, exempt)

    def _single(self, template: Template, at: Placement, start: _Push) -> _Outcome:
        ref, mode, route = start
        memo = _cache(template, "walk").setdefault(at, {})
        key = (ref, mode)
        found = memo.get(key)
        if found is None:
            found = memo[key] = self._walk(template, at, None, [(ref, mode, ())], False)
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
        template: Template,
        at: Placement,
        origin: Origin,
        starts: Sequence[_Push],
        exempt: bool,
    ) -> _Outcome:
        outcome = _Outcome()
        skip = self._skip(template, origin)
        seen: set[tuple[int, Mode]] = set()
        pending: list[tuple[int, Mode, tuple[str, ...]]] = []

        def push(ref: Ref, mode: Mode, route: tuple[str, ...], checked: bool = True) -> None:
            if checked and mode == "r" and isinstance(ref, int) and ref in skip:
                return  # what decides whether the choice exists holds where it does
            out = _classify(template, ref, mode)
            if out is not None:
                outcome.exits.setdefault(out, route)
                return
            node = cast(int, ref)
            if excluded(template, at, node):
                return
            if template.nodes[node].kind == "import":
                down(
                    path(template, template.nodes[node].scope), template.imports[node], mode, route
                )
                return
            if (node, mode) not in seen:
                seen.add((node, mode))
                pending.append((node, mode, route))

        def down(steps: Sequence[int], target: int, mode: Mode, route: tuple[str, ...]) -> None:
            found = self._enter(template, at, origin, steps, [(target, mode, ())])
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
            spec = template.nodes[node]
            if mode == "r":
                step = route if forwards(spec) else (*route, spec.owner)
                if is_decision(template, at, node):
                    outcome.refusal = _Refusal(None, spec.key, None, step)
                    break
                for ref in fields(template, at, node).following():
                    push(ref, "r", step)
                if spec.call_style == "self" and spec.function is not None:
                    items = self.inference.reads(spec.function, template)
                    if self._dynamic(template, at, items):
                        outcome.refusal = _Refusal(None, None, spec.owner, step)
                        break
                    self._reads(template, at, items, step, push, down)
                continue
            # ``present()``'s sources: a guard is read, an alias, a guard or a view is
            # followed to its output, a selection to its selector and alternatives.
            if mode == "S" and node in template.selectors:
                push(node, "r", route)
            fields_ = fields(template, at, node)
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

    def _dynamic(self, template: Template, at: Placement, items: Iterable[object]) -> bool:
        """Whether a method's reads are not fixed at ``at``: a dynamic read placed there."""
        return any(
            isinstance(item, _Dynamic) and self._placed(template, at, item.when) for item in items
        )

    def _reads(
        self,
        template: Template,
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
                elif self._placed(template, at, item.steps):
                    down(item.steps, item.node, mode, route)
            else:
                assert isinstance(item, (_Missing, _Dynamic))

    @staticmethod
    def _placed(template: Template, at: Placement, steps: Sequence[int]) -> bool:
        for stub in steps:
            held = slot(template, stub)
            if held.requires in at.never:
                return False
            template, at = held.template, held.placement
        return True

    def _enter(
        self,
        template: Template,
        at: Placement,
        origin: Origin,
        steps: Sequence[int],
        starts: Sequence[_Push],
    ) -> _Outcome:
        """Walk ``starts`` (nodes of the template ``steps`` reach) there, and what they reach
        out of it, up to ``template``: the nodes to read there, each with its way
        (``_Outcome.lifted``), or a refusal; and whether it re-entered an ``_Approx`` walk's
        stub."""
        held = slot(template, steps[0])
        if held.requires in at.never:
            return _Outcome()
        name = template.scopes[steps[0]].name
        below, reentered = self._into(origin, steps[0])
        child, child_at = held.template, held.placement
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

    # -- per template

    def summary(self, template: Template, at: Placement) -> _Summary:
        """The domains of ``template``'s subtree at ``at`` (keys relative to it); found
        once."""
        cache = _cache(template, self.name)
        if at not in cache:
            # The templates below first, deepest first: a chain of placements may be deep.
            for below, below_at in self._below(template, at):
                held = _cache(below, self.name)
                if below_at not in held:
                    held[below_at] = self._summary(below, below_at)
        return cast(_Summary, cache[at])

    def _below(self, template: Template, at: Placement) -> list[tuple[Template, Placement]]:
        """``template`` at ``at`` and every template it places, each after those it places
        (none whose domains are known already)."""
        found: list[tuple[Template, Placement]] = []
        done: set[tuple[int, Placement]] = set()
        stack: list[tuple[Template, Placement, Iterator[int]]] = [
            (template, at, iter(template.index.placed))
        ]
        while stack:
            current, current_at, stubs = stack[-1]
            stub = next(stubs, None)
            if stub is None:
                stack.pop()
                found.append((current, current_at))
                continue
            held = slot(current, stub)
            key = (id(held.template), held.placement)
            if held.requires in current_at.never or key in done:
                continue
            done.add(key)
            if held.placement not in _cache(held.template, self.name):
                stack.append((held.template, held.placement, iter(held.template.index.placed)))
        return found

    def _summary(self, template: Template, at: Placement) -> _Summary:
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

        # Its own Decisions, in node order.
        for node, spec in enumerate(template.nodes):
            domain = spec.domain
            if (
                domain is None
                or not is_decision(template, at, node)
                or excluded(template, at, node)
            ):
                continue
            named = set(domain.membership_inputs)
            starts: list[_Push] = [
                (argument.node, "r", ())
                for argument in dict.fromkeys(
                    argument for argument in spec.domain_arguments if argument.name in named
                )
            ]
            head = cast(int, spec.guard)
            outcome = self.walk(template, at, (head,), starts, exempt=True)
            if outcome.refusal is not None:
                if refused(outcome.refusal.at(spec.key)):
                    return summary
                continue
            add(outcome.exits, spec.key, ("own", head, spec.key))
        # Each placed Space's, followed into its bindings here.
        for stub in template.index.placed:
            held = slot(template, stub)
            if held.requires in at.never:
                continue
            name = template.scopes[stub].name
            below = self.summary(held.template, held.placement)
            for refusal in below.refusals:
                if refused(refusal.under(name)):
                    return summary
            for entry in below.entries:
                pushes: list[_Push] = [
                    (ref, mode, _way(name, exit, entry.exits[exit]))
                    for exit in entry.exits
                    for ref, mode in lift(held, (exit,))
                ]
                outcome = self.walk(template, at, _Approx(stub), pushes)
                if outcome.reentered:
                    # It re-entered the subtree holding the Decisions: per Decision, exactly.
                    for place, key in _places(entry.origins, name):
                        exact = self.walk(template, at, (stub, *place), pushes)
                        if exact.refusal is not None:
                            if refused(exact.refusal.at(key)):
                                return summary
                            continue
                        add(exact.exits, key, ("path", (stub, *place), key))
                    continue
                if outcome.refusal is not None:
                    if not self.every:
                        refused(outcome.refusal.at(member_key(name, entry.owner)))
                        return summary
                    for _, key in _places(entry.origins, name):
                        refused(outcome.refusal.at(key))
                    continue
                add(outcome.exits, member_key(name, entry.owner), ("via", stub, entry, name))
        summary.entries = list(merged.values())
        return summary


def check_domains(template: Template) -> None:
    """The domain rule of ``template`` compiled as a root (see the module docstring): the
    first refusal found, ``domain-reads-choice``, its Decision's key and its way relative
    to the root."""
    refusals = _Domains().summary(template, ROOT).refusals
    if refusals:
        raise refusals[0].error()


def domain_refusals(template: Template) -> dict[str, DefinitionError]:
    """Every Decision the domain rule refuses in ``template`` compiled as a root, by its key
    (a diagnostic: compiling stops at the first)."""
    refusals = _Domains(every=True).summary(template, ROOT).refusals
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
