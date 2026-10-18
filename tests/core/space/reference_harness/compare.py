# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model's answers in the comparable form, and the comparison.

A case (``Case``) is a description, built in both the model and the engine; a ``Pair`` is
one configuration of it on both sides, reached by the same changes. Each question a test
asks compares one answer:

- **kinds, values, codes and verdicts exactly**: a refused set, a forced value, a
  verdict, each refusal's codes;
- **blockers as a subset**: the engine runs opaque method bodies that stop at their first
  unavailable read, so it may name fewer of what an answer waits on than the model, never
  one the model does not name;
- **an open choice's possible cases** only where both can list them: the engine lists no
  case of a choice whose presence waits (``inspection.viable``), so where the model's
  answer waits, the cases are not compared.

A disagreement raises :class:`Disagreement`, an ``AssertionError`` naming the case, the
question and both answers. A test may state the answer a ruling on the semantics gives in
place of the model's (``ruled=``) where the model has not yet been brought to it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from core.space import reference
from core.space.reference.model import same
from finn.core.space import DefinitionError, EvaluationError

from .engine import Engine, Point, Spelling, build
from .forms import (
    AnswerForm,
    ChangeForm,
    Code,
    DefinitionForm,
    Form,
    Refusal,
    RuleForm,
    StateForm,
    ValidityForm,
)

#: the model's reason kinds as the engine's finding codes. A rule's, a view body's and a
#: derived value's reasons carry their own code (the engine's rejection by that code); a
#: refused input and a refused Space are expanded to the reasons that refuse them (the
#: engine carries those findings, not one of its own).
CODES: Mapping[reference.ReasonKind, str] = {
    reference.ReasonKind.OUTSIDE_DOMAIN: "domain-membership",
    reference.ReasonKind.NO_POSSIBLE_VALUE: "decision-no-viable-case",
    reference.ReasonKind.INACTIVE: "inactive",
    reference.ReasonKind.CONFLICT: "decision-conflict",
}


#: the reason kinds that carry their own code, the code of the engine's rejection
_OWN_CODE = (
    reference.ReasonKind.CONSTRAINT,
    reference.ReasonKind.VIEW_BODY,
    reference.ReasonKind.DERIVED_BODY,
)


class Disagreement(AssertionError):
    """The model and the engine answer one question differently."""

    def __init__(self, case: str, question: str, expected: Form, engine: Form) -> None:
        self.case, self.question, self.expected, self.engine = case, question, expected, engine
        super().__init__(self.describe())

    def describe(self) -> str:
        lines = [f"{self.case}: {self.question}"]
        for item in fields(self.expected):
            left = getattr(self.expected, item.name)
            right = getattr(self.engine, item.name)
            mark = "  " if _same(left, right) else "! "
            lines.append(f"  {mark}{item.name}: model {_show(left)} | engine {_show(right)}")
        return "\n".join(lines)


class SetupError(Exception):
    """A step that brings a configuration to its question failed on one side: the test is
    written wrongly, never an expected disagreement."""


def _show(value: object) -> str:
    if isinstance(value, frozenset):
        return "{" + ", ".join(sorted(map(_show, value))) + "}"
    if isinstance(value, Refusal):
        return f"{value.choice} ({value.kind}: {_show(value.codes)})"
    return repr(value)


def _same(left: object, right: object) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    if isinstance(left, (frozenset, Refusal)) or left is None:
        return bool(left == right)
    return same(left, right)


# --- the model's answers in the comparable form ------------------------------------------------


def _codes(solution: reference.Solution, reasons: Iterable[reference.Reason]) -> frozenset[Code]:
    found: set[Code] = set()
    for reason in reasons:
        if reason.kind in _OWN_CODE:
            found.add((reason.code, reason.subject))
        elif reason.kind is reference.ReasonKind.INPUT_REFUSED:
            for path in reason.related:
                found |= _refusal(solution, path)
        elif reason.kind is reference.ReasonKind.SPACE_REFUSED:
            found |= validity_form(solution, reason.owner).false
        else:
            found.add((CODES[reason.kind], reason.subject))
    return frozenset(found)


def _refusal(solution: reference.Solution, path: str) -> frozenset[Code]:
    """Why a refused input is refused."""
    if path in solution.index.choices:
        state = solution.state(path)
        if isinstance(state, reference.RefusedChoice):
            return _codes(solution, state.reasons)
        return frozenset()
    answer = solution.read(path)
    return (
        _codes(solution, answer.reasons) if isinstance(answer, reference.Refused) else frozenset()
    )


def state_form(solution: reference.Solution, path: str) -> StateForm:
    state = solution.state(path)
    if isinstance(state, reference.Readable):
        return StateForm("readable", state.value)
    if isinstance(state, reference.GuardRefused):
        # admitted (its own check holds), but published it reads refused through its
        # guard: the engine's refused state, through the selector's no-possible-case
        return StateForm("refused", codes=_codes(solution, state.reasons))
    if isinstance(state, reference.InputRefused):
        # kept, never read as a value; published it reads refused through the domain's
        # refused input: the engine's refused state, with the input's findings
        return StateForm("refused", codes=_codes(solution, state.reasons))
    if isinstance(state, reference.Unchecked):
        return StateForm("unchecked", state.value, blockers=frozenset(state.waits))
    if isinstance(state, reference.Forced):
        return StateForm("forced", state.value)
    if isinstance(state, reference.Open):
        possible = None if state.waits else state.possible
        return StateForm("open", possible=possible, blockers=frozenset(state.waits) - {path})
    if isinstance(state, reference.RefusedChoice):
        return StateForm("refused", codes=_codes(solution, state.reasons))
    return StateForm("inactive", state.value)


def answer_form(solution: reference.Solution, path: str) -> AnswerForm:
    answer = solution.read(path)
    if isinstance(answer, reference.Known):
        return AnswerForm("value", answer.value)
    if isinstance(answer, reference.Unresolved):
        return AnswerForm("unresolved", blockers=frozenset(answer.waits))
    if isinstance(answer, reference.Refused):
        return AnswerForm("refused", codes=_codes(solution, answer.reasons))
    return AnswerForm("inapplicable")


def rule_form(solution: reference.Solution, path: str) -> RuleForm:
    piece = solution.piece(path)
    if not piece.applies:
        return RuleForm("inapplicable")
    if piece.verdict is reference.Verdict.TRUE:
        return RuleForm("true")
    if piece.verdict is reference.Verdict.FALSE:
        return RuleForm("false", codes=_codes(solution, piece.reasons))
    return RuleForm("unknown", blockers=frozenset(piece.waits))


def validity_form(solution: reference.Solution, scope: str = "") -> ValidityForm:
    validity = solution.validity(scope)
    false = frozenset().union(*(_codes(solution, piece.reasons) for piece in validity.false))
    verdict = {
        reference.Verdict.TRUE: "valid",
        reference.Verdict.FALSE: "refused",
        reference.Verdict.UNKNOWN: "undecided",
    }[validity.verdict]
    blockers = frozenset(validity.waits) if verdict == "undecided" else frozenset[str]()
    return ValidityForm(verdict, false, blockers)  # type: ignore[arg-type]


def change_form(
    solution: reference.Solution, result: reference.Accepted | reference.Rejected
) -> ChangeForm:
    """``solution`` is the configuration the change would publish."""
    if isinstance(result, reference.Accepted):
        return ChangeForm(True, unchecked=frozenset(result.unchecked))
    refusals = frozenset(
        Refusal(
            item.choice,
            "named" if item.kind == "named" else "reached",
            _codes(solution, item.reasons),
        )
        for item in result.refusals
    )
    return ChangeForm(False, refusals=refusals)


# --- the comparison ------------------------------------------------------------------------------


def agrees(expected: Form, engine: Form) -> bool:
    """Whether the engine's answer is the model's: exactly, but for blockers (a subset) and
    the possible cases of a choice whose answer waits (not compared)."""
    if type(expected) is not type(engine):
        return False
    for item in fields(expected):
        left, right = getattr(expected, item.name), getattr(engine, item.name)
        if item.name == "blockers":
            if not right <= left:
                return False
        elif item.name == "possible" and isinstance(expected, StateForm) and expected.blockers:
            continue
        elif not _same(left, right):
            return False
    return True


def _check(case: str, question: str, expected: Form, engine: Form) -> None:
    __tracebackhide__ = True  # a disagreement is shown at the test's question
    if not agrees(expected, engine):
        raise Disagreement(case, question, expected, engine)


class Case:
    """A description built in the model and in the engine (spelled as ``spelling`` says)."""

    def __init__(self, name: str, description: reference.Space, spelling: Spelling | None = None):
        if description.trials_count_retained_checks:
            # The matrix is built on the model's default trial reach; a case on the other
            # reading would compare an answer no row accounts for.
            raise ValueError(f"{name}: a case compares the model's default trial reach only")
        self.name = name
        self.description = description
        self.engine: Engine = build(description, spelling)

    def start(
        self, facts: Mapping[str, object] | None = None, stored: Mapping[str, object] | None = None
    ) -> Pair:
        """The configuration with ``facts``, then ``stored`` committed as one change (a step:
        it must be accepted on both sides)."""
        facts = dict(facts or {})
        pair = Pair(self, reference.Configuration(self.description, facts), self.engine.open(facts))
        return pair.step(reference.Change(commit=stored)) if stored else pair


@dataclass(frozen=True)
class Pair:
    """One configuration of a case, on both sides."""

    case: Case
    model: reference.Configuration
    engine: Point

    def solution(self) -> reference.Solution:
        return self.model.solve()

    def step(self, edit: reference.Change) -> Pair:
        """A change that brings the configuration to a question: accepted on both sides, or
        the test is written wrongly (``SetupError``); nothing else is compared."""
        result = reference.change(self.model, edit)
        form, successor = self.engine.change(edit)
        if not isinstance(result, reference.Accepted) or successor is None:
            raise SetupError(
                f"{self.case.name}: a step must be accepted on both sides: {edit}: model "
                f"{type(result).__name__}, engine {'accepted' if form.accepted else form}"
            )
        return Pair(self.case, result.configuration, successor)

    # -- the questions

    def check_state(self, path: str, *, ruled: StateForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or state_form(self.solution(), path)
        _check(self.case.name, f"the state of {path}", expected, self.engine.state(path))

    def check_answer(self, path: str, *, ruled: AnswerForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or answer_form(self.solution(), path)
        _check(self.case.name, f"the answer of {path}", expected, self.engine.answer(path))

    def check_rule(self, path: str, *, ruled: RuleForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or rule_form(self.solution(), path)
        _check(self.case.name, f"the rule {path}", expected, self.engine.rule(path))

    def check_validity(self, *, ruled: ValidityForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or validity_form(self.solution())
        _check(self.case.name, "validity", expected, self.engine.validity())

    def check_possible(self, path: str, *, ruled: tuple[Any, ...] | None = None) -> None:
        """An open choice's possible cases (its forced value, or none when refused)."""
        __tracebackhide__ = True
        solution = self.solution()
        expected = state_form(solution, path)
        if ruled is not None:
            expected = replace(expected, possible=ruled)
        _check(self.case.name, f"the possible cases of {path}", expected, self.engine.state(path))

    def check_change(
        self, edit: reference.Change, *, ruled: ChangeForm | None = None
    ) -> Pair | None:
        """``edit`` compared; the successor where both accept it."""
        __tracebackhide__ = True
        result = reference.change(self.model, edit)
        if ruled is None:
            successor = _published(self.model, edit)
            expected = change_form(successor.solve(), result)
        else:
            expected = ruled
        form, point = self.engine.change(edit)
        _check(self.case.name, f"the change {_edit(edit)}", expected, form)
        if isinstance(result, reference.Accepted) and point is not None:
            return Pair(self.case, result.configuration, point)
        return None


def check_definition(
    name: str,
    description: reference.Space,
    spelling: Spelling | None = None,
    facts: Mapping[str, object] | None = None,
) -> None:
    """Whether ``description`` is a definition: the model's definition error (a cycle) or
    none, against the engine's. The engine reports a cycle when building the Space, or
    when asked (its validity, every rule and every view of the configuration with
    ``facts``), as a dependency cycle; it answers otherwise."""
    __tracebackhide__ = True
    try:
        description.validate()
        expected = DefinitionForm(None)
    except reference.DefinitionError as error:
        expected = DefinitionForm(error.kind)
    _check(name, "the definition", expected, _engine_definition(description, spelling, facts))


def _engine_definition(
    description: reference.Space,
    spelling: Spelling | None,
    facts: Mapping[str, object] | None,
) -> DefinitionForm:
    try:
        engine = build(description, spelling, cycles=True)
        point = engine.open(dict(facts or {}))
        point.validity()
        for path in sorted(engine.index.constraints):
            point.rule(path)
        for path in sorted(engine.index.views):
            point.answer(path)
    except DefinitionError as error:
        if any(finding.code == "cyclic-dependency" for finding in error.findings):
            return DefinitionForm("cycle")
        raise
    except EvaluationError as error:
        if error.role == "dependency cycle":
            return DefinitionForm("cycle")
        raise
    return DefinitionForm(None)


def _published(
    configuration: reference.Configuration, edit: reference.Change
) -> reference.Configuration:
    """The configuration ``edit`` would publish, refused or not (what its reasons read)."""
    stored = {k: v for k, v in configuration.stored.items() if k not in edit.clear}
    stored.update(edit.commit)
    facts = {k: v for k, v in configuration.facts.items() if k not in edit.unset}
    facts.update(edit.facts)
    return reference.Configuration(configuration.space, facts, stored)


def _edit(edit: reference.Change) -> str:
    parts = [f"commit {dict(edit.commit)}"] if edit.commit else []
    parts += [f"clear {list(edit.clear)}"] if edit.clear else []
    parts += [f"facts {dict(edit.facts)}"] if edit.facts else []
    parts += [f"unset {list(edit.unset)}"] if edit.unset else []
    return ", ".join(parts) or "(nothing)"


__all__ = [
    "CODES",
    "Case",
    "Disagreement",
    "Pair",
    "SetupError",
    "agrees",
    "answer_form",
    "change_form",
    "check_definition",
    "rule_form",
    "state_form",
    "validity_form",
]
