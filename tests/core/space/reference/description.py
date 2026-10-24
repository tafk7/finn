# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A Space described as plain data and small pure functions, each with declared reads.

A description is a tree of scopes. Each scope holds:

- **choices**: a name and a domain. A domain is *listed* (a tuple, or a function of
  declared reads that returns one, possibly empty), optionally narrowed by a
  membership predicate, or *membership-only* (a predicate of the value and declared
  reads, never enumerated). Membership reads facts only: facts, and derived values
  whose reads are, through derived values, facts only.
  No domain reads a view. A domain carries no ranking and a choice no policy: those are a
  compiler's, never the semantics'.
- **selectors**: a choice whose cases are named child scopes; only the selected case
  is present. Its value is the case's name. A selector **picked by a derived value**
  (``Selector(..., by=d)``, the engine's ``Select(by=d)``) is no choice: it has no key
  and is never stored. Its path reads its value, ``d``'s where that names one of its
  cases, refused ``"select-no-case"`` (:data:`NO_CASE`) where it names none (``None``
  included: a picker never answers "nothing"; a case that places nothing does,
  :func:`nothing`). Its cases take no ``when`` of their own. The model's reading 17
  states its semantics.
- **derived** values: a function of declared reads, which returns a value or a
  :class:`Refuse` with its code (the derived value then reads refused, and every reader
  of it receives a refused input).
- **views**: products, read out; a body over declared reads (anything
  readable: other views too, its own Space's and its children's) that returns a value
  or a :class:`Refuse` with its code. Nothing else reads a view.
- **members** (:class:`Members`): a value over named entries, derived values
  in scopes strictly below the scope declaring it. Its value is the sorted (entry,
  value) pairs of the entries that answer a value; an entry answering inapplicable is
  skipped (the model's docstring, "Reads"). Its entries are never views: a restriction
  of the model, not of the semantics. A ``Members`` over views, read by views alone,
  would be view-side, views composing by reading views (read-out aggregation); the model does
  not express it.
- **constraints**: a predicate over declared reads, owned by the scope declaring it,
  with the code it refuses with.
- **children**: scopes always present with their parent, unless their ``when`` guard
  (a boolean of declared reads) says otherwise. A selector's case may carry a
  ``when`` guard too.

Facts are named at the root (:class:`Space`), each required or optional: a
configuration gives every required fact, and an optional fact left out reads ``None``.
A function over an optional fact handles ``None`` itself.

Paths name nodes: a root member's path is its name; a child scope's path is its
parent's path and its name; a selector's case scope is the selector's path and the
case's name (``compute.packed``), so a choice in it is ``compute.packed.pe``. Every
read names an absolute path: a fact, a choice, a selector, a derived value, a view or a
``Members``. Constraints and scopes are not readable.

Every function receives one mapping, from each declared read's path to its value, and
reads nothing else.

:meth:`Space.validate` rejects a malformed description (an unknown or unreadable
read, a duplicate name, a malformed domain or guard), a **view read in**
(``"view-read-in"``: a constraint, a derived value, a ``when`` guard, or a domain's
membership or computed list reading a view; and, since the model expresses no
view-side ``Members``, a ``Members`` naming one; checked before the rest), a domain
whose membership reads anything but facts, or an input whose existence a choice decides
other than through the choice's own guards (``"domain-reads-choice"``: a relation to
another choice is a constraint) and a **cycle**. The rule for cycles: the definition's
dependencies form a graph in which

- a value depends on its scope's presence; a choice's value also on its membership,
  which depends on the functions of the derived values it reads and on the ``when``
  guards over facts that decide whether they exist beyond the choice's own (§ 16.17:
  those guards read facts only, so they close no cycle and are not drawn); a derived
  value's and a view's on their reads; a ``Members``'s on its entries; a view's also on
  its Space's validity;
- a constraint depends on its reads and its scope's presence;
- a scope's presence depends on its parent's, its selector's value (a chosen one's, or
  the value of the derived selector it is a case of) and its guard's reads;
- a listed domain's piece (does it list a value?) depends on its choice's presence
  and membership; a guard's piece on its scope's presence; a derived selector's guard
  piece on its holding scope's presence and its value;
- a Space's validity depends on every constraint, listed domain's piece and guard's
  piece in its scope and below (a derived selector's among them).

Any cycle in that graph is an error naming its path: derived values reading each other
in a ring, a guard reading a value of the scope it guards, views reading each other in a
ring. A view read in closes none any more: only views read views, so a view,
which depends on its Space's validity, is never below a piece of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: what every function of a description receives: each declared read's value by path
Reads = Mapping[str, Any]


class DefinitionError(Exception):
    """A malformed description. ``kind`` is one of ``"malformed"``, ``"duplicate"``,
    ``"unknown-read"``, ``"unreadable"``, ``"view-read-in"`` (the reader, then the view),
    ``"domain-reads-choice"`` and ``"cycle"``; ``path`` names the nodes involved (for a
    domain, the choice, the derived values read through, and the read that is not a fact;
    for a cycle, the nodes around it, the first repeated at the end)."""

    def __init__(self, kind: str, path: tuple[str, ...], detail: str = "") -> None:
        super().__init__(f"{kind}: {' -> '.join(path)}{': ' + detail if detail else ''}")
        self.kind = kind
        self.path = path


@dataclass(frozen=True)
class Domain:
    """A choice's values. ``values`` lists them, or ``enumerate`` computes the list
    from ``reads``; ``member`` narrows a listed domain, or alone makes a
    membership-only domain. A listed domain lists all its values: a value outside the
    list is not a member, and every listed value is also checked by ``member``.
    ``reads`` are membership's, facts only."""

    values: tuple[Any, ...] | None = None
    enumerate: Callable[[Reads], Sequence[Any]] | None = None
    member: Callable[[Any, Reads], bool] | None = None
    reads: tuple[str, ...] = ()

    @property
    def listed(self) -> bool:
        return self.values is not None or self.enumerate is not None


def listed(
    *values: Any,
    member: Callable[[Any, Reads], bool] | None = None,
    reads: tuple[str, ...] = (),
) -> Domain:
    """A fixed list of values, optionally narrowed by a membership predicate over
    ``reads``."""

    return Domain(values=tuple(values), member=member, reads=reads)


def computed(
    enumerate: Callable[[Reads], Sequence[Any]],
    reads: tuple[str, ...],
    member: Callable[[Any, Reads], bool] | None = None,
) -> Domain:
    """A list computed from declared reads."""

    return Domain(enumerate=enumerate, member=member, reads=reads)


def membership(member: Callable[[Any, Reads], bool], reads: tuple[str, ...] = ()) -> Domain:
    """A domain known only by membership: checked, never enumerated."""

    return Domain(member=member, reads=reads)


@dataclass(frozen=True)
class Choice:
    name: str
    domain: Domain


@dataclass(frozen=True)
class When:
    """A scope's guard: present only when ``test`` holds over ``reads``."""

    reads: tuple[str, ...]
    test: Callable[[Reads], bool]


@dataclass(frozen=True)
class Constraint:
    name: str
    reads: tuple[str, ...]
    predicate: Callable[[Reads], bool]
    code: str


@dataclass(frozen=True)
class Derived:
    name: str
    reads: tuple[str, ...]
    compute: Callable[[Reads], Any]


@dataclass(frozen=True)
class Refuse:
    """What a derived value's function or a view's body returns to refuse, with its
    code."""

    code: str


@dataclass(frozen=True)
class View:
    name: str
    reads: tuple[str, ...]
    body: Callable[[Reads], Any]


@dataclass(frozen=True)
class Members:
    """The entries that answer a value, as sorted (entry, value) pairs: ``entries`` name
    derived values in scopes strictly below the declaring scope."""

    name: str
    entries: tuple[str, ...]


@dataclass(frozen=True)
class Scope:
    name: str
    choices: tuple[Choice, ...] = ()
    selectors: tuple[Selector, ...] = ()
    derived: tuple[Derived, ...] = ()
    views: tuple[View, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    children: tuple[Scope, ...] = ()
    when: When | None = None
    members: tuple[Members, ...] = ()


@dataclass(frozen=True)
class Selector:
    """A choice among child scopes, its cases; its value is a case's name. With ``by``
    (a readable path), it is picked by that value and is no choice: its path reads the
    picked case's name (``Index.derived`` holds it, reading ``by``), refused
    :data:`NO_CASE` where ``by`` names no case."""

    name: str
    cases: tuple[Scope, ...]
    by: str | None = None


#: the code a selector picked by a derived value refuses with where the value names no case
NO_CASE = "select-no-case"


def nothing(name: str) -> Scope:
    """A case that places nothing (the engine's ``Select(direct=None, ...)``): an empty
    scope."""

    return Scope(name)


def _picker(cases: tuple[str, ...], by: str) -> Callable[[Reads], Any]:
    """A derived selector's value: ``by``'s, where it names one of ``cases``."""

    def pick(read: Reads) -> Any:
        value = read[by]
        if type(value) is str and value in cases:
            return value
        return Refuse(NO_CASE)

    return pick


@dataclass(frozen=True)
class Space:
    """A description: the facts a configuration gives, the root scope (whose own name is
    not part of any path), and which facts are optional (left out, one reads
    ``None``); the others are required."""

    facts: tuple[str, ...]
    root: Scope
    optional: tuple[str, ...] = ()

    def validate(self) -> Index:
        """The description's index, or :class:`DefinitionError`."""

        return Index(self)


def join(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


# --- the index: the description flattened by path, with its dependency graph ---------


@dataclass(frozen=True)
class ScopeInfo:
    path: str
    parent: str | None
    selector: str | None
    case: str | None
    when: When | None
    scope: Scope
    #: a case of a selector picked by a derived value: that selector's path (its
    #: ``selector`` is then ``None``: no choice decides it)
    picker: str | None = None


@dataclass(frozen=True)
class ChoiceInfo:
    path: str
    scope: str
    domain: Domain
    #: the case names, for a selector
    cases: tuple[str, ...] | None


#: a node of the dependency graph: (kind, path), kind one of "value", "membership",
#: "function", "constraint", "presence", "domain", "guard", "validity"
Node = tuple[str, str]


@dataclass
class Index:
    """A validated description, flattened. Built once per description; it holds the
    definition only, never an answer."""

    space: Space
    scopes: dict[str, ScopeInfo] = field(default_factory=dict)
    choices: dict[str, ChoiceInfo] = field(default_factory=dict)
    derived: dict[str, tuple[Derived, str]] = field(default_factory=dict)
    views: dict[str, tuple[View, str]] = field(default_factory=dict)
    constraints: dict[str, tuple[Constraint, str]] = field(default_factory=dict)
    members: dict[str, tuple[Members, str]] = field(default_factory=dict)
    facts: frozenset[str] = frozenset()
    #: the optional facts; the other facts are required
    optional: frozenset[str] = frozenset()
    #: each selector picked by a derived value, by path: its holding scope, the path it
    #: reads (``by``) and its cases
    pickers: dict[str, tuple[str, str, tuple[str, ...]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names: set[str] = set()
        for fact in self.space.facts:
            self._name_ok(fact, "")
            if fact in names:
                raise DefinitionError("duplicate", (fact,))
            names.add(fact)
        self.facts = frozenset(self.space.facts)
        for fact in self.space.optional:
            if fact not in self.facts:
                raise DefinitionError("malformed", (fact,), "an optional fact is a fact")
        self.optional = frozenset(self.space.optional)
        if self.space.root.when is not None:
            raise DefinitionError("malformed", ("",), "the root scope has no guard")
        self._add_scope(self.space.root, "", None, None, None, names)
        self._check_reads()
        self._check_views_read_in()
        self._check_members()
        self._check_domains()
        self._check_cycles()

    # -- building

    @staticmethod
    def _name_ok(name: str, scope: str) -> None:
        if not name or "." in name:
            raise DefinitionError("malformed", (join(scope, name),), "a name is one word")

    def _claim(self, path: str, taken: set[str]) -> None:
        if path in taken:
            raise DefinitionError("duplicate", (path,))
        taken.add(path)

    def _add_scope(
        self,
        scope: Scope,
        path: str,
        parent: str | None,
        selector: str | None,
        case: str | None,
        taken: set[str],
        picker: str | None = None,
    ) -> None:
        self.scopes[path] = ScopeInfo(path, parent, selector, case, scope.when, scope, picker)
        for choice in scope.choices:
            self._name_ok(choice.name, path)
            node = join(path, choice.name)
            self._claim(node, taken)
            domain = choice.domain
            if domain.values is not None and domain.enumerate is not None:
                raise DefinitionError("malformed", (node,), "a domain lists or computes")
            if not domain.listed and domain.member is None:
                raise DefinitionError("malformed", (node,), "a domain needs values")
            self.choices[node] = ChoiceInfo(node, path, domain, None)
        for selector_ in scope.selectors:
            self._name_ok(selector_.name, path)
            node = join(path, selector_.name)
            self._claim(node, taken)
            cases = tuple(case_.name for case_ in selector_.cases)
            if len(set(cases)) != len(cases):
                raise DefinitionError("duplicate", (node,), "two cases share a name")
            if selector_.by is not None:
                if any(case_.when is not None for case_ in selector_.cases):
                    raise DefinitionError(
                        "malformed", (node,), "a case of a derived selector takes no guard"
                    )
                self.pickers[node] = (path, selector_.by, cases)
                pick = Derived(selector_.name, (selector_.by,), _picker(cases, selector_.by))
                self.derived[node] = (pick, path)
                continue
            self.choices[node] = ChoiceInfo(node, path, Domain(values=cases), cases)
        for derived in scope.derived:
            self._name_ok(derived.name, path)
            node = join(path, derived.name)
            self._claim(node, taken)
            self.derived[node] = (derived, path)
        for view in scope.views:
            self._name_ok(view.name, path)
            node = join(path, view.name)
            self._claim(node, taken)
            self.views[node] = (view, path)
        for members in scope.members:
            self._name_ok(members.name, path)
            node = join(path, members.name)
            self._claim(node, taken)
            self.members[node] = (members, path)
        for constraint in scope.constraints:
            self._name_ok(constraint.name, path)
            node = join(path, constraint.name)
            self._claim(node, taken)
            self.constraints[node] = (constraint, path)
        for child in scope.children:
            self._name_ok(child.name, path)
            node = join(path, child.name)
            self._claim(node, taken)
            self._add_scope(child, node, path, None, None, taken)
        for selector_ in scope.selectors:
            node = join(path, selector_.name)
            picked = selector_.by is not None
            for case_ in selector_.cases:
                self._name_ok(case_.name, node)
                case_path = join(node, case_.name)
                self._claim(case_path, taken)
                self._add_scope(
                    case_,
                    case_path,
                    path,
                    None if picked else node,
                    case_.name,
                    taken,
                    node if picked else None,
                )

    def _check_reads(self) -> None:
        for node, reads in self._declared_reads():
            for read in reads:
                if read in self.constraints or read in self.scopes:
                    raise DefinitionError("unreadable", (node, read))
                if self.kind(read) is None:
                    raise DefinitionError("unknown-read", (node, read))

    def _declared_reads(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        for path, info in self.choices.items():
            yield path, info.domain.reads
        for path, (derived, _) in self.derived.items():
            yield path, derived.reads
        for path, (view, _) in self.views.items():
            yield path, view.reads
        for path, (members, _) in self.members.items():
            yield path, members.entries
        for path, (constraint, _) in self.constraints.items():
            yield path, constraint.reads
        for path, scope in self.scopes.items():
            if scope.when is not None:
                yield path, scope.when.reads

    def _check_views_read_in(self) -> None:
        """Views are read out: only a view reads a view. A ``Members`` naming a
        view is refused here too, a restriction of the model: a ``Members`` over views,
        read by views alone, would be view-side, which the model does not express."""

        for node, reads in self._declared_reads():
            if node in self.views:
                continue
            for read in reads:
                if read in self.views:
                    raise DefinitionError(
                        "view-read-in",
                        (node, read),
                        "views are read out: only a view reads a view",
                    )

    def _check_members(self) -> None:
        """A ``Members`` entry is a derived value in a scope strictly below the declaring
        scope (the model expresses no ``Members`` over views, refused before as
        ``view-read-in``), named once."""

        for path, (members, scope) in sorted(self.members.items()):
            if len(set(members.entries)) != len(members.entries):
                raise DefinitionError("duplicate", (path,), "an entry is named twice")
            for entry in members.entries:
                if entry not in self.derived:
                    raise DefinitionError(
                        "malformed", (path, entry), "a Members entry is a derived value"
                    )
                below = self.derived[entry][1]
                if below == scope or not self.within(below, scope):
                    raise DefinitionError(
                        "malformed", (path, entry), "a Members entry is in a scope below it"
                    )

    def _check_domains(self) -> None:
        """Membership reads facts only: facts, and derived values reading, through
        derived values, facts only. Anything else reaches a choice (a view through its
        Space's validity). So does a derived value whose existence depends on a choice
        other than through the choice's own existence guards (§ 16.17): a selector of
        one of its scopes that does not hold the choice, or a ``when`` there reading
        anything but facts in this sense, or a derived selector there whose value does (a
        picker over facts alone is a ``when`` over facts). A scope holding the choice is
        never asked, so an input in the choice's own case is a definition whatever picks
        that case (§ 16.20: the case's guard is the choice's own existence guard)."""

        for path in sorted(self.choices):
            own = self.choices[path].scope
            seen: set[str] = set()
            # depth-first, keeping the derived values and guarded scopes read through
            stack: list[tuple[str, tuple[str, ...]]] = [
                (read, ()) for read in reversed(self.choices[path].domain.reads)
            ]
            while stack:
                read, through = stack.pop()
                kind = self.kind(read)
                if kind == "fact" or read in seen:
                    continue
                seen.add(read)
                if kind != "derived":
                    raise DefinitionError(
                        "domain-reads-choice",
                        (path, *through, read),
                        "a domain's membership reads facts only; "
                        "a relation to another choice is a constraint",
                    )
                derived, scope = self.derived[read]
                stack.extend((inner, (*through, read)) for inner in reversed(derived.reads))
                current: str | None = scope
                while current is not None and not self.within(own, current):
                    info = self.scopes[current]
                    if info.selector is not None:
                        raise DefinitionError(
                            "domain-reads-choice",
                            (path, *through, read, info.selector),
                            "a domain's membership input exists under the choice's own "
                            "guards or under facts; one under another selector reaches it",
                        )
                    if info.when is not None:
                        guarded = (*through, read, current)
                        stack.extend((inner, guarded) for inner in reversed(info.when.reads))
                    if info.picker is not None:
                        # a derived selector's case: a guard reading the selector's value
                        stack.append((info.picker, (*through, read, current)))
                    current = info.parent

    # -- queries

    def kind(self, path: str) -> str | None:
        """``"fact"``, ``"choice"``, ``"derived"``, ``"view"`` or ``"members"``: what a
        read can name."""

        if path in self.facts:
            return "fact"
        if path in self.choices:
            return "choice"
        if path in self.derived:
            return "derived"
        if path in self.views:
            return "view"
        if path in self.members:
            return "members"
        return None

    def scope_of(self, path: str) -> str:
        if path in self.choices:
            return self.choices[path].scope
        if path in self.derived:
            return self.derived[path][1]
        if path in self.views:
            return self.views[path][1]
        if path in self.members:
            return self.members[path][1]
        return self.constraints[path][1]

    def within(self, scope: str, ancestor: str) -> bool:
        """Whether ``scope`` is ``ancestor`` or below it."""

        current: str | None = scope
        while current is not None:
            if current == ancestor:
                return True
            current = self.scopes[current].parent
        return False

    def ordered_choices(self) -> list[str]:
        """Every choice and selector, sorted by path."""

        return sorted(self.choices)

    def reads_of(self, path: str) -> tuple[str, ...]:
        """A constraint's, a derived value's or a view's declared reads; a ``Members``'s
        entries; a choice's membership reads."""

        if path in self.constraints:
            return self.constraints[path][0].reads
        if path in self.derived:
            return self.derived[path][0].reads
        if path in self.views:
            return self.views[path][0].reads
        if path in self.members:
            return self.members[path][0].entries
        return self.choices[path].domain.reads

    # -- read closures: what decides a scope's existence

    def _followed(self, path: str) -> bool:
        """Whether a closure follows ``path``'s reads: a constraint's (a seed), a
        derived value's or a view's body, a ``Members``'s entries; never a choice's
        membership."""

        return (
            path in self.constraints
            or path in self.derived
            or path in self.views
            or path in self.members
        )

    def presence_seeds(self, scope: str) -> frozenset[str]:
        """The selector (chosen, or the derived selector a case is of) and the ``when``
        reads of ``scope`` and of every scope above it."""

        seeds: set[str] = set()
        current: str | None = scope
        while current is not None:
            info = self.scopes[current]
            if info.selector is not None:
                seeds.add(info.selector)
            if info.picker is not None:
                seeds.add(info.picker)
            if info.when is not None:
                seeds.update(info.when.reads)
            current = info.parent
        return frozenset(seeds)

    def dependencies(self, seeds: Iterable[str]) -> frozenset[str]:
        """``seeds`` followed through constraints', derived values' and views' bodies
        and ``Members``' entries (their declared reads; a view's Space's validity not
        followed), adding the presence seeds of the scope of every node met, to
        closure."""

        found: set[str] = set()
        stack = list(seeds)
        while stack:
            path = stack.pop()
            if path in found:
                continue
            found.add(path)
            if self._followed(path):
                stack.extend(self.reads_of(path))
            if path not in self.facts:
                stack.extend(self.presence_seeds(self.scope_of(path)))
        return frozenset(found)

    def deciders(self, scope: str) -> frozenset[str]:
        """The choices deciding whether ``scope`` exists: the selectors above it, its
        own included, and the choices its guards' reads depend on."""

        return frozenset(
            path for path in self.dependencies(self.presence_seeds(scope)) if path in self.choices
        )

    # -- dependencies

    def edges(self, node: Node) -> list[Node]:
        """What ``node`` depends on, in the definition's graph (module docstring)."""

        kind, path = node
        out: list[Node] = []
        if kind == "value":
            what = self.kind(path)
            if what == "fact":
                return out
            out.append(("presence", self.scope_of(path)))
            if what == "choice":
                out.append(("membership", path))
            else:
                out.extend(("value", read) for read in self.reads_of(path))
                if what == "view":
                    out.append(("validity", self.views[path][1]))
        elif kind == "membership":
            out.extend(("function", read) for read in self.reads_of(path) if read in self.derived)
        elif kind == "function":
            out.extend(("function", read) for read in self.reads_of(path) if read in self.derived)
        elif kind == "constraint":
            constraint, scope = self.constraints[path]
            out.extend(("value", read) for read in constraint.reads)
            out.append(("presence", scope))
        elif kind == "presence":
            info_ = self.scopes[path]
            if info_.parent is not None:
                out.append(("presence", info_.parent))
            if info_.selector is not None:
                out.append(("value", info_.selector))
            if info_.picker is not None:
                out.append(("value", info_.picker))
            if info_.when is not None:
                out.extend(("value", read) for read in info_.when.reads)
        elif kind == "domain":
            out.append(("presence", self.choices[path].scope))
            out.append(("membership", path))
        elif kind == "guard":
            if path in self.pickers:  # a derived selector's guard: its holding scope, its value
                out.append(("presence", self.pickers[path][0]))
                out.append(("value", path))
            else:
                out.append(("presence", path))
        else:  # validity
            out.extend(
                ("constraint", name)
                for name, (_, scope) in sorted(self.constraints.items())
                if self.within(scope, path)
            )
            out.extend(
                ("domain", name)
                for name, info in sorted(self.choices.items())
                if info.domain.listed and self.within(info.scope, path)
            )
            out.extend(
                ("guard", name)
                for name, info__ in sorted(self.scopes.items())
                if info__.when is not None and self.within(name, path)
            )
            out.extend(
                ("guard", name)
                for name, (owner, _, _) in sorted(self.pickers.items())
                if self.within(owner, path)
            )
        return out

    def _check_cycles(self) -> None:
        nodes: list[Node] = (
            [("value", path) for path in sorted(self.choices)]
            + [("membership", path) for path in sorted(self.choices)]
            + [("value", path) for path in sorted(self.derived)]
            + [("function", path) for path in sorted(self.derived)]
            + [("value", path) for path in sorted(self.views)]
            + [("value", path) for path in sorted(self.members)]
            + [("constraint", path) for path in sorted(self.constraints)]
            + [("presence", path) for path in sorted(self.scopes)]
            + [("validity", path) for path in sorted(self.scopes)]
        )
        done: set[Node] = set()
        for start in nodes:
            if start in done:
                continue
            # iterative depth-first search keeping the current path
            stack: list[tuple[Node, Iterator[Node]]] = [(start, iter(self.edges(start)))]
            on_path = {start: 0}
            while stack:
                current, children = stack[-1]
                child = next(children, None)
                if child is None:
                    stack.pop()
                    del on_path[current]
                    done.add(current)
                    continue
                if child in on_path:
                    cycle = [entry[0] for entry in stack[on_path[child] :]] + [child]
                    raise DefinitionError("cycle", tuple(f"{k}:{p}" for k, p in cycle))
                if child in done:
                    continue
                on_path[child] = len(stack)
                stack.append((child, iter(self.edges(child))))
