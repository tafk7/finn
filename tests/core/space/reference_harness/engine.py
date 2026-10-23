# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine side of the harness: a model description built as an engine Space, and an
implementation of the engine's public questions asked the model's, each answer put in the
comparable form (``forms``).

This is the adapter to the engine. Everything that depends on how the engine is
**authored** or **asked** is in this one module, so a change of the engine's API is a port of
this file alone. The implementation answers each question itself, in the model's terms: no
legacy form, and no answer the adapter computes in its place.

- **The implementation.** The description is authored once, as Space classes (the public
  API: ``composite``, ``design_space``); every implementation reads the same classes.
  ``Engine.open`` opens the design space with an ``Implementation`` of the class-based
  oracle's interface (``support.space_oracle``; by default the harness's,
  ``implementation.default``: the public API, ``public.PUBLIC``, unless a session names
  another), ``Engine.make`` has it build one from nothing (``Implementation.make``, the
  walks' cold build), and ``Point`` asks every question of what it opens, an ``Answers``, by key,
  never of ``inspection`` or the configuration itself. So another implementation is
  compared with the model by naming it (``walks.walk(..., implementation=)``, the CLI's
  ``--implementation module:attribute``) or by making it the default (pytest's
  ``--space-implementation module:attribute``), and a planted fault (``public.Planted``)
  shows that the comparison is live.

- **Spelling.** A rule of the description is a constraint in the model, and in the engine
  every rule is a ``@constraint`` of the Space that owns it, which its Space's validity
  counts. The spellings (``Spelling.rules``, beside each case) name where a rule came from
  and how it reads: ``Admission`` (the default), ``Plain``, ``Requirement`` (a rule over a
  choice and a fact, for some of the choice's cases) and ``Obligation`` (a rule stated
  beside a view of its Space) are each a plain ``@constraint`` reading its inputs
  explicitly; ``ByName`` is a ``@constraint`` method reading its inputs by attribute name,
  ``getattr(self, name)``. A view carries no rule. A domain with a membership predicate is
  a ``domain(accepts=, candidates=)``: membership is the model's whole membership, the
  enumeration the listed values.
- **Questions**, each the implementation's own answer (``Answers``; the public API's through
  ``inspection`` and the point's public API):

  - a choice's state is ``Answers.states``' (set, waiting with what it waits on,
    dormant, open with what its presence waits on, absent);
  - a change's report: refused, its type checks refused (``ChangeOutcome``, status
    ``"refused"``) as ``assignment`` refusals with their codes, and each ``BrokenRule`` as
    a ``rule`` refusal (its origin, its result's codes, its standing choices); accepted,
    what it puts dormant, brings into play and opens, each stored value new facts dropped
    (``ChangeOutcome`` status ``"dropped"``: its old value the receiver's, its codes the
    failed check's, as an assignment refusal's), and its ``BrokenRule``s as the rules the
    published design space holds newly false (as a refusal's rules: origin, codes,
    standing). Drops and rules compare as sets;
  - a bind (``Answers.bind``; the public ``finn.core.space.bind``) of saved choices under a
    design space's facts, in the same form as an accepted change, read against the facts
    with nothing stored whatever the receiver stores: what it drops (its old value the
    saved one), the rules the kept choices break, and what it moves;
  - possible values are ``Answers.viable``'s: ``Possible`` with what it moves,
    ``Refused`` with its broken rules; ``None`` where the engine lists nothing;
  - validity is ``Answers.validity`` (``"valid"``, ``"refused"``, ``"open"``);
  - a fact's, a choice's (stored or not), a value's or a rule's answer is
    ``Answers.query``'s;
  - a definition is refused for a dependency cycle (when building or asked) and for a
    domain whose membership reads a choice (at compile).

The description becomes one Space class per scope, built with ``composite``. Every member
keeps the description's path as its key: a choice ``compute.packed.pe`` is the engine's
Decision ``compute.packed.pe``. A scope that reads something outside it takes it as a formal
(``in-<path>``), bound by its parent; a chosen selector is a ``Select`` of its cases, its
value read through a derived member ``key-<name>`` that binds the selector (a binding to a
selector reads its case's name); a selector picked by a derived value (the model's reading
17) is ``Select(by=...)``, a case whose scope holds nothing a ``None`` case, its value read
by binding the selector itself (a binding to a selector reads its case's name); a child's
guard is a derived member ``when-<name>``. Facts
are the root's formals (SZ100 (5)): a required fact is ``Param()``, an optional one
``Param(required=False)``, left unsupplied where the model reads ``None`` (the engine
reads it unresolved: T4c's row). A ``Members`` is the
engine's ``Members(KEY)``, a key of its own, each entry its child's value export of that key
(``exports = {KEY: derived}``); each entry is in a child or a case placed by the
``Members``'s scope, one per child. Its value, the engine's ``Located`` entries, is read as
the model's sorted (entry, value) pairs, ``Located.node`` mapped to the entry's path, in
its answer and wherever a function reads it.
"""

from __future__ import annotations

import inspect as pyinspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from support.space_oracle import Answers, Changed, Implementation

from core.space import reference as model
from core.space.reference.model import same
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    EvaluationError,
    Located,
    Members,
    Param,
    QueryResult,
    Rejected,
    Select,
    Space,
    Unresolved,
    ViewKey,
    composite,
    constraint,
    derived,
    design_space,
    domain,
    inspection,  # its answer types alone: every question goes to an Answers
    reject,
    view,
)
from finn.core.space.edits import BrokenRule, ChangeOutcome
from finn.core.space.occurrence import held

from .forms import (
    AnswerForm,
    ChangeForm,
    Code,
    DefinitionForm,
    DroppedForm,
    ListedForm,
    OutcomeForm,
    PossibleForm,
    RefusalForm,
    RuleForm,
    StateForm,
    ValidityForm,
)
from .implementation import chosen


class NotTranslatable(Exception):
    """A description, a spelling or a change this harness cannot write in the engine. Never
    a disagreement: a test that meets one fails."""


# --- spelling -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    """A ``@constraint`` of its Space (the default)."""


@dataclass(frozen=True)
class Plain:
    """A ``@constraint`` of its Space: every constraint counts, so the same as
    ``Admission``."""


@dataclass(frozen=True)
class Requirement:
    """A rule that ``choice``'s ``cases`` need ``fact``: a ``@constraint`` of its Space, like
    any rule. The model's rule must read ``choice`` and ``fact`` and hold exactly when the
    value is not one of ``cases`` or ``fact`` is truthy."""

    choice: str
    fact: str
    cases: tuple[object, ...]


@dataclass(frozen=True)
class ByName:
    """A ``@constraint`` method of its Space, reading each input by name (``getattr(self,
    name)`` over a list of names, through child nodes for a child's member): the engine
    discovers the reads when the body runs."""


@dataclass(frozen=True)
class Obligation:
    """A rule stated beside ``view``, a view of the rule's own Space: a ``@constraint`` of
    that Space, like any rule (a view carries none; it refuses when its Space is
    refused)."""

    view: str


Rule = Admission | Plain | Requirement | Obligation | ByName


@dataclass(frozen=True)
class Spelling:
    """How a case's rules are written in the engine (by rule path; ``Admission`` when
    absent), the value types the description does not state (by path; a listed domain's
    are read from its values, everything else is ``int``), and the guarded scopes written
    as their selector's own guard (``guarded_selectors``)."""

    rules: Mapping[str, Rule] = field(default_factory=dict)
    types: Mapping[str, type] = field(default_factory=dict)
    #: guarded scopes holding one selector and nothing else, each written as that
    #: selector's own guard: the scope a plain child, its selector a ``Select(...,
    #: when=)`` under the scope's ``when``, so the guard and the cases' guards are nodes of
    #: one Space. A guard piece of the selector names the scope, as the model's does
    guarded_selectors: frozenset[str] = frozenset()

    def rule(self, path: str) -> Rule:
        return self.rules.get(path, Admission())


# --- the translator ---------------------------------------------------------------------------


def _explicit(
    reads: tuple[str, ...], body: Callable[[dict[str, Any]], Any], returns: object
) -> Any:
    """A function of keyword arguments ``a0, a1, ...`` (one per read, in order) calling
    ``body`` with the mapping from read path to value: an explicit-input function the engine
    binds by its signature."""

    names = [f"a{index}" for index in range(len(reads))]

    def call(**arguments: Any) -> Any:
        return body({read: arguments[name] for read, name in zip(reads, names, strict=True)})

    call.__signature__ = pyinspect.Signature(  # type: ignore[attr-defined]
        [pyinspect.Parameter(name, pyinspect.Parameter.KEYWORD_ONLY) for name in names]
    )
    call.__annotations__ = {**dict.fromkeys(names, object), "return": returns}
    return call


def _value_type(values: Iterable[object], path: str) -> type:
    kinds = {type(value) for value in values}
    if len(kinds) == 1:
        (kind,) = kinds
        if kind in (bool, int, str):
            return kind
    if not kinds:
        return int
    raise NotTranslatable(f"{path}: state the value type of {sorted(map(str, kinds))}")


def _name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _empty(scope: model.Scope) -> bool:
    """Whether a case's scope holds nothing: the case places nothing."""
    return not (
        scope.choices
        or scope.selectors
        or scope.derived
        or scope.views
        or scope.constraints
        or scope.children
        or scope.members
    )


#: the member of a scope spelled as its selector's guard holding that guard
OWN_GUARD = "own-when"
#: the engine's code of a guard piece's own finding
GUARD_REFUSED = "guard-refused"


def _input(path: str) -> str:
    """A scope's formal taking ``path`` from outside it."""
    return "in-" + path.replace(".", "-")


class _Unchecked(model.Index):
    """The description's index without the model's definition checks (a view read in,
    cycles, a domain reading a choice): a description the model refuses is still built in
    the engine, to see what the engine answers."""

    def _check_views_read_in(self) -> None:
        return None

    def _check_domains(self) -> None:
        return None

    def _check_cycles(self) -> None:
        return None


class _Builder:
    """The engine Space classes of one description, one per scope, built on demand."""

    def __init__(self, space: model.Space, spelling: Spelling, unchecked: bool = False) -> None:
        self.index = _Unchecked(space) if unchecked else space.validate()
        self.spelling = spelling
        index = self.index
        #: the paths a derived selector is picked by (a case's name, a ``str``)
        self.picking = {by for _, by, _ in index.pickers.values()}
        for path, rule in spelling.rules.items():
            if path not in index.constraints:
                raise NotTranslatable(f"{path}: spelled, but not a rule of the description")
            if isinstance(rule, Requirement) and (
                rule.choice not in index.choices or index.kind(rule.fact) is None
            ):
                raise NotTranslatable(f"{path}: its requirement names unknown nodes")
            if isinstance(rule, Obligation) and (
                rule.view not in index.views
                or index.views[rule.view][1] != index.constraints[path][1]
            ):
                raise NotTranslatable(f"{path}: an obligation of a view of its own Space")
        #: a guarded selector's path -> the scope whose guard it carries
        self.guards: dict[str, str] = {}
        for path in sorted(spelling.guarded_selectors):
            info = index.scopes.get(path)
            if info is None or info.when is None or info.selector is not None:
                raise NotTranslatable(f"{path}: spelled as a selector's guard, but not a child")
            held = info.scope
            if len(held.selectors) != 1 or held != model.Scope(
                held.name, selectors=held.selectors, when=held.when
            ):
                raise NotTranslatable(f"{path}: a selector's guard holds one selector alone")
            self.guards[f"{path}.{held.selectors[0].name}"] = path
        self.externals = {scope: self._externals(scope) for scope in index.scopes}
        #: per ``Members``: its export key, and its entries by the child's local name
        self.keys: dict[str, ViewKey[Any]] = {}
        self.located: dict[str, dict[str, str]] = {}
        #: per scope: the export key of each entry it holds
        self.exported: dict[str, dict[str, str]] = {scope: {} for scope in index.scopes}
        for path, (members, scope) in sorted(index.members.items()):
            self.keys[path] = ViewKey(_name(path), int)
            self.located[path] = {}
            for entry in members.entries:
                below = index.derived[entry][1]
                if index.scopes[below].parent != scope:
                    raise NotTranslatable(f"{path}: {entry} is not in a child placed by its scope")
                if path in self.exported[below]:
                    raise NotTranslatable(f"{path}: two entries in {below}")
                self.exported[below][path] = entry
                local = below[len(scope) + 1 :] if scope else below
                self.located[path][local] = entry
        #: per scope: the declaration of each member it owns, by model path or generated name
        self.declared: dict[str, dict[str, object]] = {scope: {} for scope in index.scopes}
        #: per scope: the node declaration of each child scope and case placed in it
        self.nodes: dict[str, dict[str, Any]] = {scope: {} for scope in index.scopes}
        self.selectors: dict[str, Any] = {}
        self.classes: dict[str, type[Space]] = {}
        self.pending: set[tuple[str, str]] = set()
        #: an engine alias key (a scope's formal) -> the model path it takes
        self.aliases: dict[str, str] = {}

    # -- reads

    def owner(self, path: str) -> str:
        """The scope a readable node belongs to; a fact is the root's."""
        return "" if path in self.index.facts else self.index.scope_of(path)

    def _reads(self, scope: str) -> set[str]:
        """Every read of a node at or below ``scope`` (the guards of the scopes below it
        included, its own guard not, as its parent evaluates that, unless its selector
        carries it)."""
        index = self.index
        found: set[str] = set()
        for path, choice in index.choices.items():
            if index.within(choice.scope, scope):
                found.update(choice.domain.reads)
        for table in (index.derived, index.views, index.constraints):
            for path, (node, owner) in table.items():
                if index.within(owner, scope):
                    found.update(node.reads)
        for path, (members, owner) in index.members.items():
            if index.within(owner, scope):
                found.update(members.entries)
        for path, info in index.scopes.items():
            own = path == scope and path not in self.spelling.guarded_selectors
            if not own and info.when is not None and index.within(path, scope):
                found.update(info.when.reads)
        return found

    def _externals(self, scope: str) -> tuple[str, ...]:
        if scope == "":
            return ()
        return tuple(
            sorted(
                read
                for read in self._reads(scope)
                if not self.index.within(self.owner(read), scope)
            )
        )

    def value_type(self, path: str) -> object:
        if path in self.spelling.types:
            return self.spelling.types[path]
        if path in self.index.pickers or (path in self.picking and path in self.index.derived):
            return str  # a case's name: a derived selector's value, or what picks one
        info = self.index.choices.get(path)
        if info is not None:
            if info.cases is not None:
                return str
            if info.domain.values is not None:
                return _value_type(info.domain.values, path)
        return int

    def reference(self, scope: str, path: str) -> Any:
        """``path`` as an engine reference written in ``scope``'s class body."""
        if path in self.externals[scope]:
            return self.member(scope, _input(path))
        owner = self.owner(path)
        if owner == scope:
            return self.member(scope, path)
        if not self.index.within(owner, scope):
            raise NotTranslatable(f"{path} is not visible from {scope or 'the root'}")
        chain: list[str] = []
        current: str | None = owner
        while current != scope:
            assert current is not None
            chain.append(current)
            current = self.index.scopes[current].parent
        chain.reverse()
        node = self.node(scope, chain[0])
        for below in chain[1:]:
            info = self.index.scopes[below]
            if info.picker is not None:
                node = getattr(node, _name(info.picker))[info.case]
            elif info.selector is None:
                node = getattr(node, _name(below))
            else:
                node = getattr(node, _name(info.selector))[info.case]
        member = (
            "key-" + _name(path)
            if path in self.index.choices and self.index.choices[path].cases is not None
            else _name(path)
        )
        return getattr(node, member)

    # -- members

    def member(self, scope: str, name: str) -> Any:
        """The declaration of ``name`` (a model path the scope owns, or a generated name)
        in ``scope``, created on first use."""
        declared = self.declared[scope]
        if name in declared:
            return declared[name]
        if (scope, name) in self.pending:
            raise NotTranslatable(f"{name}: its declaration reads itself")
        self.pending.add((scope, name))
        declared[name] = self._declare(scope, name)
        self.pending.discard((scope, name))
        return declared[name]

    def _declare(self, scope: str, name: str) -> object:
        index = self.index
        if name.startswith("in-"):
            return Param()
        if name.startswith("when-"):
            child = name.removeprefix("when-")
            return self.guard(scope, f"{scope}.{child}" if scope else child)
        if name == OWN_GUARD:
            return self.guard(scope, scope)
        if name in index.pickers:
            return self.picked(scope, name)
        if name in index.facts:
            return Param(required=False) if name in index.optional else Param()
        if name in index.choices and index.choices[name].cases is not None:
            # a selector's value, read as its selected key
            return self._selector_key(scope, name)
        if name in index.choices:
            return self._choice(scope, name)
        if name in index.derived:
            return self._derived(scope, name)
        if name in index.views:
            return self._view(scope, name)
        if name in index.members:
            return Members(self.keys[name])
        if name in index.constraints:
            return self._constraint(scope, name)
        raise NotTranslatable(f"{name}: nothing to declare in {scope or 'the root'}")

    def model_reads(self, read: Mapping[str, Any]) -> dict[str, Any]:
        """What a function of the description receives: each ``Members``'s ``Located``
        entries as the model's sorted (entry, value) pairs."""
        return {
            path: pairs(self.located[path], value) if path in self.located else value
            for path, value in read.items()
        }

    def _aliases(self, scope: str, reads: tuple[str, ...]) -> dict[str, Any]:
        return {f"a{position}": self.reference(scope, read) for position, read in enumerate(reads)}

    def _selector_key(self, scope: str, path: str) -> object:
        decision = self.selector(scope, path)
        return derived(a0=decision)(_explicit(("key",), lambda read: read["key"], str))

    def selector(self, scope: str, path: str) -> Any:
        if path in self.selectors:
            return self.selectors[path]
        info = self.index.choices[path]
        assert info.cases is not None
        entries: dict[str, type[Space] | Space] = {}
        for case in info.cases:
            entries[case] = self.node(scope, f"{path}.{case}")
        guard = {"when": self.member(scope, OWN_GUARD)} if path in self.guards else {}
        self.selectors[path] = Select(**entries, **guard)
        return self.selectors[path]

    def picked(self, scope: str, path: str) -> Any:
        """A selector picked by a derived value, ``Select(by=...)``: each case a node, or
        ``None`` where its scope holds nothing (a case that places nothing)."""
        if path in self.selectors:
            return self.selectors[path]
        _, by, cases = self.index.pickers[path]
        entries: dict[str, object] = {}
        for case in cases:
            below = f"{path}.{case}"
            entries[case] = (
                None if _empty(self.index.scopes[below].scope) else self.node(scope, below)
            )
        if all(entry is None for entry in entries.values()):
            raise NotTranslatable(f"{path}: Select needs a case that places something")
        self.selectors[path] = Select(by=self.reference(scope, by), **entries)
        return self.selectors[path]

    def _choice(self, scope: str, path: str) -> object:
        info = self.index.choices[path]
        domain_ = info.domain
        simple = domain_.values is not None and domain_.member is None and not domain_.reads
        if simple and domain_.values:
            return Decision(values=domain_.values)
        reads = domain_.reads
        names = [f"a{position}" for position in range(len(reads))]

        def readings(arguments: Mapping[str, object]) -> dict[str, object]:
            return {read: arguments[name] for read, name in zip(reads, names, strict=True)}

        def listing(arguments: Mapping[str, object]) -> tuple[object, ...] | None:
            if domain_.values is not None:
                return domain_.values
            if domain_.enumerate is not None:
                return tuple(domain_.enumerate(readings(arguments)))
            return None

        def accepts(*, candidate: object, **arguments: object) -> bool:
            values = listing(arguments)
            if values is not None and not any(same(item, candidate) for item in values):
                return False
            return domain_.member is None or bool(domain_.member(candidate, readings(arguments)))

        def candidates(**arguments: object) -> tuple[object, ...]:
            values = listing(arguments)
            assert values is not None
            return values

        built = domain(
            accepts=accepts,
            candidates=candidates if domain_.listed else None,
            **dict(zip(names, (self.reference(scope, read) for read in reads), strict=True)),
        )
        return Decision(domain=built)

    def _derived(self, scope: str, path: str) -> object:
        """A derived value; one whose function refuses rejects with the refusal's code."""
        node = self.index.derived[path][0]

        def body(read: dict[str, Any]) -> object:
            output = node.compute(self.model_reads(read))
            if isinstance(output, model.Refuse):
                return reject(output.code, "the derived value refuses")
            return output

        returns = self.value_type(path) | Rejected
        return derived(**self._aliases(scope, node.reads))(_explicit(node.reads, body, returns))

    def _view(self, scope: str, path: str) -> object:
        node = self.index.views[path][0]

        def body(read: dict[str, Any]) -> object:
            output = node.body(self.model_reads(read))
            if isinstance(output, model.Refuse):
                return reject(output.code, "the view refuses")
            return output

        returns = self.value_type(path) | Rejected
        return view(**self._aliases(scope, node.reads))(_explicit(node.reads, body, returns))

    def _constraint(self, scope: str, path: str) -> object:
        rule = self.index.constraints[path][0]

        def body(read: dict[str, Any]) -> bool | Rejected:
            if rule.predicate(self.model_reads(read)):
                return True
            return reject(rule.code, "the rule refuses")

        if isinstance(self.spelling.rule(path), ByName):
            return constraint(self._by_name(scope, path, rule.reads, body))
        return constraint(**self._aliases(scope, rule.reads))(
            _explicit(rule.reads, body, bool | Rejected)
        )

    def _by_name(
        self,
        scope: str,
        path: str,
        reads: tuple[str, ...],
        body: Callable[[dict[str, Any]], bool | Rejected],
    ) -> Any:
        """A method reading each of ``reads`` by attribute name from ``self``."""
        chains: list[tuple[str, tuple[str, ...]]] = []
        for read in reads:
            owner = self.owner(read)
            if read in self.index.facts or not self.index.within(owner, scope):
                raise NotTranslatable(f"{path}: a rule read by name reads inside its Space")
            chain = [_name(read)]
            current = owner
            while current != scope:
                info = self.index.scopes[current]
                if info.selector is not None or info.when is not None:
                    raise NotTranslatable(f"{path}: read by name through a plain child only")
                chain.append(_name(current))
                assert info.parent is not None
                current = info.parent
            chains.append((read, tuple(reversed(chain))))

        def method(self: Any) -> bool | Rejected:
            read: dict[str, Any] = {}
            for target, chain in chains:
                node = self
                for name in chain:
                    node = getattr(node, name)
                read[target] = node
            return body(read)

        method.__annotations__ = {"return": bool | Rejected}
        return method

    # -- scopes

    def node(self, scope: str, below: str) -> Any:
        """The node declaration placing the child scope or case ``below`` in ``scope``."""
        nodes = self.nodes[scope]
        if below in nodes:
            return nodes[below]
        info = self.index.scopes[below]
        space_type = self.space_class(below)
        if info.selector is not None:
            if info.when is not None:
                raise NotTranslatable(f"{below}: a selector's case takes no guard here")
            nodes[below] = space_type()
        elif info.when is None or below in self.spelling.guarded_selectors:
            nodes[below] = space_type()
        else:
            guard = self.member(scope, "when-" + _name(below))
            nodes[below] = space_type(when=guard)
        return nodes[below]

    def guard(self, scope: str, below: str) -> object:
        when = self.index.scopes[below].when
        assert when is not None
        if any(self.index.within(self.owner(read), below) for read in when.reads):
            raise NotTranslatable(f"{below}: its guard reads inside it")
        return derived(**self._aliases(scope, when.reads))(
            _explicit(when.reads, lambda read: bool(when.test(self.model_reads(read))), bool)
        )

    def space_class(self, scope: str) -> type[Space]:
        if scope in self.classes:
            return self.classes[scope]
        index = self.index
        members: dict[str, object] = {}
        annotations: dict[str, object] = {}
        mine = [
            path
            for table in (index.choices, index.derived, index.views, index.members)
            for path in table
            if self.owner(path) == scope
        ] + [path for path in index.constraints if self.owner(path) == scope]
        if scope == "":
            for fact in sorted(index.facts):
                members[fact] = self.member(scope, fact)
                annotations[fact] = self.value_type(fact)
        for path in self.externals[scope]:
            members[_input(path)] = self.member(scope, _input(path))
            annotations[_input(path)] = self.value_type(path)
        if scope in self.spelling.guarded_selectors:
            members[OWN_GUARD] = self.member(scope, OWN_GUARD)
        for path in mine:
            if path in index.pickers:
                members[_name(path)] = self.member(scope, path)
                annotations[_name(path)] = object | None  # a case may place nothing
                continue
            if path in index.choices and index.choices[path].cases is not None:
                members[_name(path)] = self.selector(scope, path)
                members["key-" + _name(path)] = self.member(scope, path)
                annotations[_name(path)] = object
                continue
            members[_name(path)] = self.member(scope, path)
            if path in index.choices:
                annotations[_name(path)] = self.value_type(path)
        for path in sorted(index.scopes):
            child = index.scopes[path]
            if child.parent == scope and child.selector is None and child.picker is None:
                members[_name(path)] = self.node(scope, path)
                if child.when is not None and path not in self.spelling.guarded_selectors:
                    members["when-" + _name(path)] = self.member(scope, "when-" + _name(path))
        # bind every placed node's formals (assigned after the node, as a class body may)
        for below, placed in self.nodes[scope].items():
            for path in self.externals[below]:
                setattr(placed, _input(path), self.reference(scope, path))
                self.aliases[f"{below}.{_input(path)}"] = path
        owned = {_name(path) for path in mine} | {
            _name(path) for path, child in index.scopes.items() if child.parent == scope
        }
        clashes = sorted(name for name in owned if "-" in name)
        if clashes:
            raise NotTranslatable(f"{scope}: names the harness generates: {clashes}")
        name = "Ref_" + (scope.replace(".", "_") or "root")
        exports = {
            self.keys[path]: self.member(scope, entry)
            for path, entry in sorted(self.exported[scope].items())
        }
        self.classes[scope] = composite(
            name, members, annotations=annotations, exports=exports or None
        )
        return self.classes[scope]


# --- the engine's answers --------------------------------------------------------------------


@dataclass(frozen=True)
class Engine:
    """A description built as an engine Space class, with what maps the engine's owners
    back to the description's paths."""

    description: model.Space
    spelling: Spelling
    root: type[Space]
    index: model.Index
    aliases: Mapping[str, str]
    #: per ``Members``: its entries by the child's local name (``Located.node``)
    located: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: a selector spelled with its scope's guard -> that scope, which its guard piece names
    guards: Mapping[str, str] = field(default_factory=dict)

    def open(
        self, facts: Mapping[str, object], implementation: Implementation | None = None
    ) -> Point:
        """The configuration with ``facts`` and nothing committed, as ``implementation``
        answers it (the harness's default by default, ``implementation.default``): an
        optional fact the model reads ``None`` left unsupplied. The design space is built by
        the public API, the common input every implementation opens."""
        supplied = {name: value for name, value in facts.items() if not self._left_out(name, value)}
        configuration = design_space(self.root(**cast(dict[str, Any], supplied)))
        implementation = chosen(implementation)
        return Point(
            self, implementation.open(configuration), implementation.name, _reached(implementation)
        )

    def one_space(self, left: Mapping[str, object], right: Mapping[str, object]) -> AnswerForm:
        """Whether the facts ``left`` and ``right``, each given to the root as written (an
        optional fact given ``None`` passed as ``None``, never left out), make one design
        space: the same interned facts layer and equal design space values. Refused with
        the definition error's code where the engine refuses either."""
        try:
            first = held(design_space(self.root(**cast(dict[str, Any], left)))).ds
            second = held(design_space(self.root(**cast(dict[str, Any], right)))).ds
        except DefinitionError:
            return AnswerForm("refused", codes=frozenset({("definition-error", "")}))
        return AnswerForm("value", first.layer is second.layer and first == second)

    def make(
        self,
        facts: Mapping[str, object],
        stored: Mapping[str, object],
        implementation: Implementation | None = None,
    ) -> Point:
        """The configuration with ``facts`` (as ``open`` takes them) and exactly the
        choices ``stored`` (by path, dormant ones included), built from nothing by
        ``implementation`` (``Implementation.make``; the harness's default by default): no
        rule checked, nothing dropped. A walked configuration answers as the one made from its
        facts and stored choices."""
        supplied = {name: value for name, value in facts.items() if not self._left_out(name, value)}
        implementation = chosen(implementation)
        made = implementation.make(self.root, supplied, dict(stored))
        return Point(self, made, implementation.name, _reached(implementation))

    def _left_out(self, name: str, value: object) -> bool:
        return value is None and name in self.index.optional

    def path(self, owner: str) -> str:
        """The model path an engine finding's owner names."""
        return self.aliases.get(owner, owner)

    def codes(self, findings: Iterable[Any], only: str | None = None) -> frozenset[Code]:
        """Findings as codes; ``only``: the findings owned by that node alone. A guard
        piece's own finding (``guard-refused``) of a selector spelled with its scope's guard
        names the scope (``Spelling.guarded_selectors``)."""
        return frozenset(
            (
                finding.code,
                self.guards.get(finding.owner, finding.owner)
                if finding.code == GUARD_REFUSED
                else self.path(finding.owner),
            )
            for finding in findings
            if only is None or finding.owner == only
        )

    def blockers(self, findings: Iterable[Any], asked: str | None = None) -> frozenset[str]:
        """What an answer waits on: its findings' owners, but for ``asked``, the node whose
        answer it is (the engine names a choice whose own check waits as the owner of what
        it waits on; the model names the inputs, and blockers compare as a subset)."""
        return frozenset(self.path(finding.owner) for finding in findings) - {asked}


def build(
    description: model.Space, spelling: Spelling | None = None, unchecked: bool = False
) -> Engine:
    """The engine Space class of ``description``, spelled as ``spelling`` says; with
    ``unchecked``, built even where the model refuses the description as no definition."""
    builder = _Builder(description, spelling or Spelling(), unchecked)
    root = builder.space_class("")
    return Engine(
        description,
        builder.spelling,
        root,
        builder.index,
        dict(builder.aliases),
        builder.located,
        dict(builder.guards),
    )


def definition(
    description: model.Space,
    spelling: Spelling | None = None,
    facts: Mapping[str, object] | None = None,
    implementation: Implementation | None = None,
) -> DefinitionForm:
    """Whether the engine takes ``description`` as a definition: a dependency cycle reported
    when building the Space, or when ``implementation`` is asked (the harness's default by
    default; its validity, every rule and every view of the configuration with ``facts``),
    is ``"cycle"``; a domain whose membership reads a choice, refused at compile, is
    ``"domain-reads-choice"``; otherwise ``None``. (The engine has no ``"view-read-in"``:
    it builds a Space reading a view in, T4b's row.)"""
    try:
        engine = build(description, spelling, unchecked=True)
        point = engine.open(dict(facts or {}), implementation)
        point.validity()
        for path in sorted(engine.index.constraints):
            point.rule(path)
        for path in sorted(engine.index.views):
            point.answer(path)
    except DefinitionError as error:
        codes = {finding.code for finding in error.findings}
        if "cyclic-dependency" in codes:
            return DefinitionForm("cycle")
        if "domain-reads-choice" in codes:
            return DefinitionForm("domain-reads-choice")
        raise
    except EvaluationError as error:
        if error.role == "dependency cycle":
            return DefinitionForm("cycle")
        raise
    return DefinitionForm(None)


def _reached(implementation: Implementation) -> frozenset[str]:
    """The later milestones ``implementation`` declares it has reached (``matrix.reached``,
    unchecked here: the matrix imports this module)."""
    return frozenset(getattr(implementation, "milestones", frozenset()))


@dataclass(frozen=True)
class Point:
    """One configuration, as an implementation answers it (``answers``), asked the model's
    questions: every one through the ``Answers`` interface, by key (a description's path is
    its engine key)."""

    engine: Engine
    answers: Answers
    #: the implementation answering (``Implementation.name``)
    implementation: str
    #: the later milestones it declares it has reached (its ``milestones``; ``matrix.reached``)
    reached: frozenset[str] = frozenset()

    def stored(self) -> dict[str, object]:
        """The committed choices, dormant ones included, by path."""
        return dict(self.answers.stored())

    def _query(self, path: str) -> QueryResult[Any]:
        index = self.engine.index
        if path in index.choices and index.choices[path].cases is not None:
            # a selector's value, read as its selected key
            scope = index.choices[path].scope
            return self.answers.query(_join(scope, f"key-{_name(path)}"))
        return self.answers.query(path)

    def _paths(self, keys: Iterable[str]) -> frozenset[str]:
        return frozenset(self.engine.path(key) for key in keys)

    def _successor(self, changed: Changed) -> Point | None:
        if changed.after is None:
            return None
        return Point(self.engine, changed.after, self.implementation, self.reached)

    # -- the questions

    def listed(self) -> ListedForm:
        """The choices the implementation lists (``Answers.states``' keys)."""
        return ListedForm(frozenset(path for path, _ in self.answers.states()))

    def state(self, path: str) -> StateForm:
        """A choice's state (``Answers.states``); a choice the implementation does not list
        (its facts alone rule it out, SZ113) is absent, as the model answers it by name."""
        found = dict(self.answers.states()).get(path)
        if found is None:
            return StateForm("absent")
        if isinstance(found, inspection.Set):
            return StateForm("set", found.value)
        if isinstance(found, inspection.Waiting):
            return StateForm("waiting", found.value, self._paths(found.waits) - {path})
        if isinstance(found, inspection.Dormant):
            return StateForm("dormant", found.value)
        if isinstance(found, inspection.Open):
            return StateForm("open", blockers=self._paths(found.waits))
        return StateForm("absent")

    def answer(self, path: str) -> AnswerForm:
        """A fact's, a choice's, a derived value's or a view's answer."""
        result = self._query(path)
        engine = self.engine
        if isinstance(result, Available):
            value = result.value
            if path in engine.located:
                value = pairs(engine.located[path], result.value)
            return AnswerForm("value", value)
        if isinstance(result, Unresolved):
            return AnswerForm("unresolved", blockers=engine.blockers(result.findings, path))
        if isinstance(result, Rejected):
            return AnswerForm("refused", codes=engine.codes(result.findings))
        return AnswerForm("inapplicable")

    def rule(self, path: str) -> RuleForm:
        """A rule's answer."""
        return _rule(self.engine, self.answers.query(path))

    def validity(self) -> ValidityForm:
        """The implementation's validity of the configuration (``Answers.validity``):
        refused with its false pieces' codes, open with what its unknown pieces wait on
        (each piece's own key aside), or valid."""
        engine = self.engine
        answer = self.answers.validity()
        if answer.status == "refused":
            false = frozenset().union(
                *(engine.codes(piece.result.findings) for piece in answer.pieces)
            )
            return ValidityForm("refused", false)
        if answer.status == "open":
            blockers = frozenset().union(
                *(
                    engine.blockers(piece.result.findings, engine.path(piece.owner))
                    for piece in answer.pieces
                )
            )
            return ValidityForm("open", blockers=blockers)
        return ValidityForm("valid")

    def possible(self, path: str) -> PossibleForm:
        """A choice's possible values (``Answers.viable``): each listed value possible,
        with what it moves, or impossible, with each rule it breaks; ``None`` where the
        engine lists nothing."""
        (found,) = self.answers.viable([path])
        if found.values is None:
            return PossibleForm(None)
        outcomes: list[tuple[object, OutcomeForm]] = []
        for item in found.values:
            if isinstance(item, inspection.Possible):
                moved = (self._paths(item.dormant), self._paths(item.in_play))
                outcome = OutcomeForm("possible", *moved, self._paths(item.opened))
            else:
                outcome = OutcomeForm("impossible", refusals=_broken(self.engine, item.broken))
            outcomes.append((item.value, outcome))
        return PossibleForm(tuple(outcomes))

    def change(self, edit: model.Change) -> tuple[ChangeForm, Point | None]:
        """``edit`` applied: its comparable result, and the configuration it publishes."""
        if edit.unset or any(
            self.engine._left_out(name, value) for name, value in edit.facts.items()
        ):
            raise NotTranslatable("the engine cannot set an optional fact back to unsupplied")
        before = self.stored()
        changed = self.answers.change(dict(edit.commit), tuple(edit.clear), dict(edit.facts))
        return self._form(changed, before)

    def bind(self, stored: Mapping[str, object]) -> tuple[ChangeForm, Point]:
        """``stored`` bound under this design space's facts (``Answers.bind``): the design
        space of the facts with exactly those choices, never refused. Its report, and the
        design space it publishes. Its report reads against the facts with nothing stored,
        whatever this design space stores, as the model's ``bind`` does."""
        form, point = self._form(self.answers.bind(dict(stored)), stored)
        if point is None:
            raise AssertionError(f"the engine refused a bind: {form}")
        return form, point

    def bind_outcomes(self, stored: Mapping[str, object]) -> tuple[tuple[str, str, bool], ...]:
        """Each outcome of ``stored``'s bind (``Answers.bind``), in the saved order: its
        choice's path, its status and whether it was requested. The model's bind has no
        statuses; a hand case reads them against what it keeps and drops."""
        report = self.answers.bind(dict(stored)).report
        return tuple(
            (self.engine.path(outcome.owner), outcome.status, outcome.requested)
            for outcome in report.outcomes
        )

    def _form(
        self, changed: Changed, before: Mapping[str, object]
    ) -> tuple[ChangeForm, Point | None]:
        """A change's or a bind's report in the comparable form, and what it publishes.
        A dropped value's old value is read from ``before``: the receiver's stored choices,
        or a bind's saved ones."""
        engine, report = self.engine, changed.report
        if not report.accepted:
            refusals = _assignments(engine, report.outcomes) | _broken(engine, report.broken)
            return ChangeForm(False, refusals=refusals), None
        moved = (self._paths(report.dormant), self._paths(report.in_play))
        form = ChangeForm(
            True,
            *moved,
            self._paths(report.opened),
            dropped=_dropped(engine, report.outcomes, before),
            broken=_broken(engine, report.broken),
        )
        successor = self._successor(changed)
        if successor is None:
            raise AssertionError(f"{self.implementation} accepted a change with no successor")
        return form, successor


def _join(scope: str, name: str) -> str:
    return f"{scope}.{name}" if scope else name


def pairs(entries: Mapping[str, str], located: object) -> tuple[tuple[str, object], ...]:
    """A ``Members``'s ``Located`` entries as the model's sorted (entry, value) pairs, each
    ``Located.node`` mapped to its entry's path."""
    found: list[tuple[str, object]] = []
    for item in cast(tuple[Located[object], ...], located):
        if item.node is None or item.node not in entries:
            raise AssertionError(f"a Members entry from an unknown child {item.node!r}")
        found.append((entries[item.node], item.value))
    return tuple(sorted(found, key=lambda pair: pair[0]))


def _rule(engine: Engine, result: QueryResult[Any]) -> RuleForm:
    if isinstance(result, Available):
        return RuleForm("true" if result.value else "false")
    if isinstance(result, Rejected):
        return RuleForm("false", codes=engine.codes(result.findings))
    if isinstance(result, Unresolved):
        return RuleForm("unknown", blockers=engine.blockers(result.findings))
    return RuleForm("inapplicable")


def _assignments(engine: Engine, outcomes: Iterable[ChangeOutcome]) -> frozenset[RefusalForm]:
    """A refused change's type-check refusals: each outcome refused (a choice named, or a
    stored one a facts change checked again), with every code of its type check's answer
    (a domain input's origin's codes where it refuses through that input)."""
    found: set[RefusalForm] = set()
    for outcome in outcomes:
        if outcome.status != "refused":
            continue
        result = outcome.result
        codes: frozenset[Code] = (
            engine.codes(result.findings) if isinstance(result, Rejected) else frozenset()
        )
        found.add(RefusalForm("assignment", engine.path(outcome.owner), codes))
    return frozenset(found)


def _dropped(
    engine: Engine, outcomes: Iterable[ChangeOutcome], before: Mapping[str, object]
) -> frozenset[DroppedForm]:
    """An accepted report's dropped values: each outcome ``"dropped"``, its old value read
    from ``before`` (the engine reports the check, not the value), its codes those of the
    check it failed, as an assignment refusal's. A set: the engine reports them in flow
    order, the model by path."""
    found: set[DroppedForm] = set()
    for outcome in outcomes:
        if outcome.status != "dropped":
            continue
        path = engine.path(outcome.owner)
        result = outcome.result
        codes: frozenset[Code] = (
            engine.codes(result.findings) if isinstance(result, Rejected) else frozenset()
        )
        found.add(DroppedForm(path, before[path], codes))
    return frozenset(found)


def _broken(engine: Engine, broken: Iterable[BrokenRule]) -> frozenset[RefusalForm]:
    """Each rule a change breaks, at its origin: its own codes and the stored choices
    standing in its way."""
    return frozenset(
        RefusalForm(
            "rule",
            engine.path(item.rule),
            engine.codes(item.result.findings),
            frozenset(engine.path(key) for key in item.standing),
        )
        for item in broken
    )


__all__ = [
    "Admission",
    "ByName",
    "Engine",
    "NotTranslatable",
    "Obligation",
    "Plain",
    "Point",
    "Requirement",
    "Rule",
    "Spelling",
    "build",
    "definition",
]
