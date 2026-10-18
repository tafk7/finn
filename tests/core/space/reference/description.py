# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A Space described as plain data and small pure functions, each with declared reads.

A description is a tree of scopes. Each scope holds:

- **choices**: a name and a domain. A domain is *listed* (a tuple, or a function of
  declared reads that returns one, possibly empty), optionally narrowed by a
  membership predicate, or *membership-only* (a predicate of the value and declared
  reads, never enumerated). A required choice has no safe default; it stays open
  until something commits it.
- **selectors**: a choice whose cases are named child scopes; only the selected case
  is present. Its value is the case's name.
- **derived** values: a function of declared reads.
- **views**: products; a body over declared reads that returns a value or a
  :class:`Refuse` with its code.
- **constraints**: a predicate over declared reads, owned by the scope declaring it,
  with the code it refuses with.
- **children**: scopes always present with their parent, unless their ``when`` guard
  (a boolean of declared reads) says otherwise. A selector's case may carry a
  ``when`` guard too.

Facts are named at the root (:class:`Space`), and given, or left unknown, with each
configuration.

Paths name nodes: a root member's path is its name; a child scope's path is its
parent's path and its name; a selector's case scope is the selector's path and the
case's name (``compute.packed``), so a choice in it is ``compute.packed.pe``. Every
read names an absolute path: a fact, a choice, a selector, a derived value or a view.
Constraints and scopes are not readable.

Every function receives one mapping, from each declared read's path to its value, and
reads nothing else.

:meth:`Space.validate` rejects a malformed description (an unknown or unreadable
read, a duplicate name, a malformed domain or guard) and a **cycle**. The rule for
cycles: the definition's dependencies form a graph in which

- a value (a choice's readable value, a derived value, a view) depends on its reads
  and on its scope's presence; a choice's value also on its own check, which depends
  on its domain's reads; a view also on its Space's validity;
- a constraint depends on its reads and its scope's presence;
- a scope's presence depends on its parent's, its selector's value and its guard's
  reads;
- a Space's validity depends on every constraint and every choice's own check in its
  scope and below.

Any cycle in that graph is an error naming its path. So a constraint reading a view
of its own Space, or of any Space above it (whose validity counts that constraint),
directly or through derived values and other views, is a cycle; reading a child's
view is not. A choice's own check reading a view of its own Space is a cycle too.
Forcing adds no edge: a forced value follows from the constraints that read the
choice, by propagation over one configuration, not from a definition's dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: what every function of a description receives: each declared read's value by path
Reads = Mapping[str, Any]


class DefinitionError(Exception):
    """A malformed description. ``kind`` is one of ``"malformed"``, ``"duplicate"``,
    ``"unknown-read"``, ``"unreadable"`` and ``"cycle"``; ``path`` names the nodes
    involved (for a cycle, the nodes around it, the first repeated at the end)."""

    def __init__(self, kind: str, path: tuple[str, ...], detail: str = "") -> None:
        super().__init__(f"{kind}: {' -> '.join(path)}{': ' + detail if detail else ''}")
        self.kind = kind
        self.path = path


@dataclass(frozen=True)
class Domain:
    """A choice's values. ``values`` lists them, or ``enumerate`` computes the list
    from ``reads``; ``member`` narrows a listed domain, or alone makes a
    membership-only domain. A listed domain lists all its values: a value outside the
    list is not a member, and every listed value is also checked by ``member``."""

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
    """A domain known only by membership: checked, never enumerated or forced."""

    return Domain(member=member, reads=reads)


@dataclass(frozen=True)
class Choice:
    name: str
    domain: Domain
    required: bool = False


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
    """What a view's body returns to refuse, with its code."""

    code: str


@dataclass(frozen=True)
class View:
    name: str
    reads: tuple[str, ...]
    body: Callable[[Reads], Any]


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


@dataclass(frozen=True)
class Selector:
    """A choice among child scopes, its cases; its value is a case's name."""

    name: str
    cases: tuple[Scope, ...]
    required: bool = False


@dataclass(frozen=True)
class Space:
    """A description: the facts a configuration gives, and the root scope (whose own
    name is not part of any path).

    ``trials_count_retained_checks`` sets how far a value case's trial reaches. Off
    (the default), a value choice's case is tried against its own check and the
    constraints that read it (a selector's case also against the stored choices of
    that case), so a forced value may make a retained choice's own check false, which
    refuses the change. On, the trial also counts every other stored choice's own check
    that reads the choice, so forcing never picks such a value, and a retained choice's
    own check found false under forced values at forcing's final check starts a
    conflict (the change is never refused through a value nobody committed).
    """

    facts: tuple[str, ...]
    root: Scope
    trials_count_retained_checks: bool = False

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


@dataclass(frozen=True)
class ChoiceInfo:
    path: str
    scope: str
    domain: Domain
    required: bool
    #: the case names, for a selector
    cases: tuple[str, ...] | None


#: a node of the dependency graph: (kind, path), kind one of "value", "check",
#: "constraint", "presence", "validity"
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
    facts: frozenset[str] = frozenset()
    #: the choices each constraint's answer depends on (``reads_choices``)
    constraint_reads: dict[str, frozenset[str]] = field(default_factory=dict)
    #: the choices each choice's own check depends on, other than its own value
    check_reads: dict[str, frozenset[str]] = field(default_factory=dict)
    #: for each choice, the constraints whose answer depends on it, sorted
    readers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names: set[str] = set()
        for fact in self.space.facts:
            self._name_ok(fact, "")
            if fact in names:
                raise DefinitionError("duplicate", (fact,))
            names.add(fact)
        self.facts = frozenset(self.space.facts)
        if self.space.root.when is not None:
            raise DefinitionError("malformed", ("",), "the root scope has no guard")
        self._add_scope(self.space.root, "", None, None, None, names)
        self._check_reads()
        self._check_cycles()
        self.constraint_reads = {
            name: self.reads_choices(("constraint", name)) for name in sorted(self.constraints)
        }
        self.check_reads = {
            name: self.reads_choices(("check", name)) - {name} for name in sorted(self.choices)
        }
        self.readers = {
            choice: tuple(name for name, reads in self.constraint_reads.items() if choice in reads)
            for choice in sorted(self.choices)
        }

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
    ) -> None:
        self.scopes[path] = ScopeInfo(path, parent, selector, case, scope.when, scope)
        for choice in scope.choices:
            self._name_ok(choice.name, path)
            node = join(path, choice.name)
            self._claim(node, taken)
            domain = choice.domain
            if domain.values is not None and domain.enumerate is not None:
                raise DefinitionError("malformed", (node,), "a domain lists or computes")
            if not domain.listed and domain.member is None:
                raise DefinitionError("malformed", (node,), "a domain needs values")
            self.choices[node] = ChoiceInfo(node, path, domain, choice.required, None)
        for selector_ in scope.selectors:
            self._name_ok(selector_.name, path)
            node = join(path, selector_.name)
            self._claim(node, taken)
            cases = tuple(case_.name for case_ in selector_.cases)
            if len(set(cases)) != len(cases):
                raise DefinitionError("duplicate", (node,), "two cases share a name")
            self.choices[node] = ChoiceInfo(
                node, path, Domain(values=cases), selector_.required, cases
            )
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
            for case_ in selector_.cases:
                self._name_ok(case_.name, node)
                case_path = join(node, case_.name)
                self._claim(case_path, taken)
                self._add_scope(case_, case_path, path, node, case_.name, taken)

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
        for path, (constraint, _) in self.constraints.items():
            yield path, constraint.reads
        for path, scope in self.scopes.items():
            if scope.when is not None:
                yield path, scope.when.reads

    # -- queries

    def kind(self, path: str) -> str | None:
        """``"fact"``, ``"choice"``, ``"derived"`` or ``"view"``: what a read can name."""

        if path in self.facts:
            return "fact"
        if path in self.choices:
            return "choice"
        if path in self.derived:
            return "derived"
        if path in self.views:
            return "view"
        return None

    def scope_of(self, path: str) -> str:
        if path in self.choices:
            return self.choices[path].scope
        if path in self.derived:
            return self.derived[path][1]
        if path in self.views:
            return self.views[path][1]
        return self.constraints[path][1]

    def within(self, scope: str, ancestor: str) -> bool:
        """Whether ``scope`` is ``ancestor`` or below it."""

        current: str | None = scope
        while current is not None:
            if current == ancestor:
                return True
            current = self.scopes[current].parent
        return False

    def common_scope(self, paths: Sequence[str]) -> str:
        """The lowest scope holding every node of ``paths``."""

        chains = []
        for path in paths:
            chain = []
            current: str | None = self.scope_of(path)
            while current is not None:
                chain.append(current)
                current = self.scopes[current].parent
            chains.append(list(reversed(chain)))
        common = ""
        for level in zip(*chains, strict=False):
            if all(item == level[0] for item in level):
                common = level[0]
            else:
                break
        return common

    def ordered_choices(self) -> list[str]:
        """Every choice and selector, sorted by path."""

        return sorted(self.choices)

    # -- dependencies

    def edges(self, node: Node, definition: bool) -> list[Node]:
        """What ``node`` depends on. With ``definition`` a choice's value also depends
        on its own check (the graph cycles are found in); without it a choice's value
        is taken as given (what a trial on one frozen state reads)."""

        kind, path = node
        out: list[Node] = []
        if kind == "value":
            what = self.kind(path)
            if what == "fact":
                return out
            out.append(("presence", self.scope_of(path)))
            if what == "choice":
                if definition:
                    out.append(("check", path))
            elif what == "derived":
                out.extend(("value", read) for read in self.derived[path][0].reads)
            else:
                out.extend(("value", read) for read in self.views[path][0].reads)
                out.append(("validity", self.views[path][1]))
        elif kind == "check":
            info = self.choices[path]
            out.extend(("value", read) for read in info.domain.reads)
            out.append(("presence", info.scope))
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
            if info_.when is not None:
                out.extend(("value", read) for read in info_.when.reads)
        else:  # validity
            out.extend(
                ("constraint", name)
                for name, (_, scope) in sorted(self.constraints.items())
                if self.within(scope, path)
            )
            out.extend(
                ("check", name)
                for name, info in sorted(self.choices.items())
                if self.within(info.scope, path)
            )
        return out

    def reads_choices(self, node: Node) -> frozenset[str]:
        """The choices whose value or own check ``node``'s answer depends on, through
        derived values, views, validity and presence, on one frozen state."""

        seen: set[Node] = set()
        stack = [node]
        found: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            kind, path = current
            if kind == "check" or (kind == "value" and path in self.choices):
                found.add(path)
            stack.extend(self.edges(current, definition=False))
        return frozenset(found)

    def _check_cycles(self) -> None:
        nodes: list[Node] = (
            [("value", path) for path in sorted(self.choices)]
            + [("check", path) for path in sorted(self.choices)]
            + [("value", path) for path in sorted(self.derived)]
            + [("value", path) for path in sorted(self.views)]
            + [("constraint", path) for path in sorted(self.constraints)]
            + [("presence", path) for path in sorted(self.scopes)]
            + [("validity", path) for path in sorted(self.scopes)]
        )
        done: set[Node] = set()
        for start in nodes:
            if start in done:
                continue
            # iterative depth-first search keeping the current path
            stack: list[tuple[Node, Iterator[Node]]] = [
                (start, iter(self.edges(start, definition=True)))
            ]
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
                stack.append((child, iter(self.edges(child, definition=True))))
