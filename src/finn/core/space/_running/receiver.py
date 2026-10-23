# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The method reader: what a ``self`` method's receiver reads through.

A method's ``self`` (``receiver``) is a Space of its scope's class whose reads go to its
``MethodReader``: the read boundary (only from the method's own call, never in a value
transformation), the read check (a read its source does not name is the method's
failure), resolution of what it reads, ``present()`` (whether a node is present, or a value
input supplied: a source applies, its value never read), and the receivers below it
(``child``, ``candidate``, ``selected_candidate``, ``root``).

The reader reads the compiled model alone: its scope records, their choices and the
addresses' member records (``model.scopes``), its resolution (``model.resolve``), and what
each method may read, the
model's per-class record's inferred reads (``per_class.PerClass.reads``). How a read is
answered is the evaluator's: a reader of an evaluation says how a member's value and
answer are read (``_read_value``, ``_value``, ``_answer``) and whether a source applies
(``_source_applies``). What a method read only to know whether something exists
(``existence``: a selector, for its candidate; a guard or a source's presence, for
``present``) is told apart from what it read as a value (``valued``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._configuration import Space
from .._failure import (
    Call,
    check_transformation,
    current,
    fault,
    remember_exception,
)
from .._nodes import NodeDecl, unwrap
from ..declarations import Param, declared_path
from ..errors import EvaluationError, RequestError
from ..references import descend
from ..results import Available, Inapplicable, QueryResult, Unresolved
from .depth import Deeper

if TYPE_CHECKING:
    from .._scopes import Address, ChoiceRecord, ScopeRecord
    from ..compiler import Model


def scope_choice(scope: ScopeRecord, decision: object) -> ChoiceRecord:
    """A Decision over nodes of scope ``scope``, there."""
    try:
        return scope.choices[unwrap(decision)]
    except (KeyError, TypeError) as error:
        raise RequestError("the selector is not part of this compiled scope") from error


class MethodReader:
    """The reads of one running ``self`` method (see the module docstring). ``model``: the
    compiled model it reads; ``node``: the method's address; ``call``: its call. ``trace``:
    what it read, in order, with the answers it got (the evaluator's reads record them)."""

    __slots__ = ("model", "node", "call", "trace", "existence", "valued")

    def __init__(self, model: Model[Space], node: Address, call: Call) -> None:
        self.model = model
        self.node = node
        self.call = call
        self.trace: list[tuple[Address, QueryResult[object]]] = []
        self.existence: set[Address] = set()
        self.valued: set[Address] = set()

    # -- the read boundary -------------------------------------------------------------

    def _enter(self) -> None:
        """Any read through this reader: only from its own call, never in a value
        transformation; an abandoned call reads nothing more."""
        check_transformation()
        active = current()
        if active is not self.call:
            if active is None:
                raise EvaluationError(
                    self.node.owner, "field read", "a method's receiver is read after it returned"
                )
            error = EvaluationError(active.owner, "field read", "cross-configuration access")
            fault(active, error)
            raise error
        if self.call.abandoned is not None:
            raise self.call.abandoned

    def _check(self, index: Address) -> None:
        """A read the method's source does not name is its failure."""
        allowed = self.model.per_class.reads.methods.get(self.node)
        if allowed is None or index in allowed:
            return
        member = index.key
        error = EvaluationError(
            self.node.owner,
            "read-undeclared",
            f"read {member}, which its source does not name: read it through a path from "
            "its receiver, or declare it in __reads__",
        )
        fault(self.call, error)
        raise error

    # -- how a read is answered: the evaluator's ------------------------------------------

    def _read_value(self, point: Space, reference: object) -> object:
        """A member read as a value (``self.x``): resolved, checked, recorded, detached; a
        non-value halts the method."""
        raise NotImplementedError

    def _answer(self, index: Address) -> QueryResult[object]:
        """``index``'s answer, checked and recorded, read to know whether something exists;
        a non-value does not halt."""
        raise NotImplementedError

    def _value(self, index: Address) -> object:
        """``index``'s value, recorded and detached; a non-value halts the method (in
        cleanup, after a failure, it is a cleanup read)."""
        raise NotImplementedError

    def _source_applies(self, index: Address) -> bool | Address:
        """Whether a source applies at ``index``, or the presence node still undecided."""
        raise NotImplementedError

    # -- what a receiver reads ---------------------------------------------------------

    def read_value(self, point: Space, reference: object) -> object:
        """A member read as a value (``self.x``). Whatever it raises is the call's."""
        try:
            return self._read_value(point, reference)
        except BaseException as cause:
            if current() is self.call and not isinstance(cause, Deeper):
                remember_exception(self.call, cause)
            raise

    def present(self, point: Space, node: object) -> bool:
        """Whether a node is present, or a value input supplied (``occurrence.present``)."""
        self._enter()
        target = self._node_scope(point, node)
        if target is None:
            if isinstance(node, Param):
                return self._supplied(point, node)
            raise RequestError(
                "present() takes a node (a child or a selector's case) or a value input"
            )
        presence = target.guard
        if presence is None:
            return True
        answer = self._answer(presence)
        if isinstance(answer, Available):
            return answer.value is True
        if isinstance(answer, Inapplicable) or (
            isinstance(answer, Unresolved) and _unsupplied(answer)
        ):
            return False
        return bool(self._value(presence))  # undecided: halts like any value read

    def _node_scope(self, point: Space, reference: object) -> ScopeRecord | None:
        """For a node reference, the node's scope; ``None`` for anything else."""
        if not isinstance(reference, Space):
            return None
        path = declared_path(reference)
        if path is None:
            return None
        *outer, last = path
        if not isinstance(last, NodeDecl):
            return None
        holder = descend(point._scope, outer)
        if last is holder.record:
            return holder
        target = holder.children.get(last)
        if target is None:
            raise RequestError("node is not part of this compiled scope")
        return target

    def _supplied(self, point: Space, formal: object) -> bool:
        """Whether a value input is supplied: a source applies (never its value)."""
        index = self.model.resolve(point._scope, formal)
        applies = self._source_applies(index)
        if isinstance(applies, bool):
            return applies
        self._value(applies)  # undecided: halts like any value read
        raise AssertionError("an undecided presence read halts")

    def _holds(self, guard: Address) -> bool | Address:
        """Whether a guard holds (inapplicable: it does not), or the guard while undecided."""
        answer = self._answer(guard)
        if isinstance(answer, Available):
            return answer.value is True
        return False if isinstance(answer, Inapplicable) else guard

    def child(self, point: Space, record: object) -> Space:
        """A child node's receiver."""
        self._enter()
        child_scope = point._scope.children.get(record)
        if child_scope is None:
            raise RequestError("child node is not part of this compiled scope")
        return receiver(self, child_scope)

    def _choice(self, scope: ScopeRecord, decision: object) -> ChoiceRecord:
        return scope_choice(scope, decision)

    def selected_candidate(self, point: Space, decision: object) -> Space | None:
        """The receiver of the candidate a Decision over nodes selects, or ``None``."""
        self._enter()
        choice = self._choice(point._scope, decision)
        self._check(choice.selector)
        self.existence.add(choice.selector)
        case = self._value(choice.selector)
        for key, candidate in choice.cases:
            if key == case:
                return None if candidate is None else receiver(self, candidate)
        raise EvaluationError(choice.key, "selection", "selector is not a declared case")

    def candidate(self, point: Space, decision: object, case: str) -> Space | None:
        """A candidate's receiver, selected or not (``None`` for a case placing nothing)."""
        self._enter()
        if type(case) is not str:
            raise RequestError("a case name must be a string")
        choice = self._choice(point._scope, decision)
        for key, index in choice.cases:
            if key == case:
                return None if index is None else receiver(self, index)
        raise RequestError(f"{choice.key}: unknown case {case!r}")

    def root(self, point: Space) -> Space:
        """The root's receiver."""
        self._enter()
        return point if point._scope.parent is None else receiver(self, self.model.scopes.root)


def _unsupplied(answer: Unresolved) -> bool:
    findings = answer.findings
    return bool(findings) and all(finding.code == "input-unsupplied" for finding in findings)


def receiver(reader: MethodReader, scope: ScopeRecord) -> Space:
    """A Space of ``scope``'s class reading through ``reader``: a method's ``self``."""
    space_type = scope.space_type
    made = object.__new__(space_type)
    object.__setattr__(made, "_held", reader)
    object.__setattr__(made, "_scope", scope)
    return made


__all__ = ["MethodReader", "receiver", "scope_choice"]
