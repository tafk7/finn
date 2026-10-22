# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluations of the running type (``finn.core.space._running.evaluate``) for the engine's
own tests.

An evaluation is opened from a root declaration: the design space of its facts, as
``design_space`` binds them, with stored choices by key, recognized and frozen as a
change's are and checked by nothing (a design space is its facts and stored choices),
built as the public API with no cache builds one (``core.space._running_type.Running``),
evaluated in a session of its own with no cache. ``source_line`` finds an interruption
point in the engine's source.
"""

from __future__ import annotations

import inspect as _inspect
from collections.abc import Callable, Mapping
from typing import cast

from finn.core.space import Space, design_space
from finn.core.space._running.evaluate import Ans, Evaluation, evaluation
from finn.core.space.ir import NodeId
from finn.core.space.occurrence import held
from finn.core.space.results import QueryResult

from core.space._running_type import Running, RunningAnswers


def open_evaluation(declaration: Space, choices: Mapping[str, object] | None = None) -> Evaluation:
    """The evaluation of ``declaration``'s facts with ``choices`` (by key) stored, unchecked,
    in a session of its own with no cache."""
    ds = held(design_space(declaration)).ds
    header = ds.model.linked.header
    facts = {header(index).key: value for index, value in ds.facts.items()}
    made = cast(RunningAnswers, Running().make(type(declaration), facts, dict(choices or {})))
    return evaluation(made.ds, made.session)


def node(ev: Evaluation, key: str) -> NodeId:
    return ev.model.linked.keys[key]


def answer(ev: Evaluation, key: str) -> QueryResult[object]:
    return ev.result(node(ev, key))


def entry(ev: Evaluation, key: str) -> Ans:
    """``key``'s answer with its trace, assessment and refusal."""
    return ev.get(node(ev, key))


def validity(ev: Evaluation, scope: int = 0) -> Ans:
    """``scope``'s validity node's answer."""
    return ev.get(cast(NodeId, ev.linked.scopes[scope].validity))


def source_line(function: Callable[..., object], marker: str) -> int:
    """The line of ``function``'s source holding ``marker``, once (an interruption point)."""
    source, start = _inspect.getsourcelines(function)
    matches = [start + offset for offset, line in enumerate(source) if marker in line]
    assert len(matches) == 1, f"expected one interruption boundary for {marker!r}: {matches}"
    return matches[0]
