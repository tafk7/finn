# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The revised vocabulary and the narrow translation of recorded line calls."""

from __future__ import annotations

import pytest

from dataclasses import replace
from dataflow_next.harness.record import Call, Raised, Target
from dataflow_next.harness.replay import judge
from typing import Any

import finn.dataflow.schedule as line_schedule
import finn.dataflow.traversal as line_traversal
import finn.dataflow_next.schedule as schedule
import finn.dataflow_next.traversal as traversal
from finn.dataflow_next.timing import Kernel


def test_schedule_names_keep_iterations_separate_from_port_beats() -> None:
    a, b = schedule.Index("a"), schedule.Index("b")
    core = schedule.Schedule({a: 3, b: 4}, {b: 2})
    assert core.fold(b) == 2 and core.iterations == 6
    assert core.beat_times(reduces=(b,)) == (1, 3, 5)
    assert not hasattr(core, "beat_count") and not hasattr(core, "steps")
    assert not hasattr(traversal, "period") and not hasattr(traversal, "AxisStep")
    old_keyword: dict[str, Any] = {"view": (3, 4)}
    with pytest.raises(TypeError, match="view"):
        core.present((3, 4), (a, b), lanes=(b,), **old_keyword)
    assert Kernel("k", iterations=2, reads=((0, 1),), writes=()).iterations == 2
    marker = traversal.level_done(beats=6, inner=2)
    assert marker == traversal.LevelDone(beats=6, inner=2)
    assert repr(marker) == "LevelDone(beats=6, inner=2)"


def _present_call(shape: tuple[int, ...] = (2, 3)) -> Call:
    a, b = line_schedule.Index("a"), line_schedule.Index("b")
    core = line_schedule.Schedule({a: 2, b: 3})
    args = (core, shape, (a, b))
    kwargs: dict[str, Any] = {"view": (2, 3)}
    try:
        outcome: Any = core.present(shape, (a, b), **kwargs)
    except line_schedule.Refused as error:
        outcome = Raised.of(error)
    return Call(
        Target("method", "finn.dataflow.schedule", "Schedule", "present"),
        args,
        kwargs,
        outcome,
    )


def test_old_present_keyword_replays_without_mutating_the_record() -> None:
    call = _present_call()
    assert judge(call).verdict == "same"
    assert call.kwargs == {"view": (2, 3)}
    both = replace(call, kwargs={"view": (2, 3), "reshape": (6,)})
    assert judge(both).verdict == "untranslatable"


def test_old_form_name_replays_and_a_changed_result_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    form = line_traversal.vector_major((2, 3), 1).repeated(2)
    call = Call(
        Target("function", "finn.dataflow.traversal", "", "period"),
        (form,),
        {},
        line_traversal.period(form),
    )
    assert judge(call).verdict == "same"
    monkeypatch.setattr(traversal, "one_pass", lambda form: form)
    assert judge(call).verdict == "differs"


def test_both_reshape_refusals_keep_the_named_code_and_replay_the_line_message() -> None:
    call = _present_call((7,))
    assert judge(call).verdict == "same"
    a, b = line_schedule.Index("a"), line_schedule.Index("b")
    access = line_schedule.Access("x", (7,), (a, b), reshaped=True)
    args = ([access], {a: 2, b: 3})
    with pytest.raises(line_schedule.Refused) as old:
        line_schedule.bind_extents(*args)
    bound = Call(
        Target("function", "finn.dataflow.schedule", "", "bind_extents"),
        args,
        {},
        Raised.of(old.value),
    )
    assert judge(bound).verdict == "same"
    x, y = schedule.Index("a"), schedule.Index("b")
    with pytest.raises(schedule.Refused) as new:
        schedule.bind_extents([schedule.Access("x", (7,), (x, y), reshaped=True)], {x: 2, y: 3})
    assert new.value.findings[0].code == "reshape-size-mismatch"


@pytest.mark.parametrize("target", ["present", "bind_extents"])
@pytest.mark.parametrize("shape,reshaped", [((), (2,)), ((2,), ()), ((3,), (2,))])
def test_reshape_refusal_translation_includes_scalar_shapes(
    target: str, shape: tuple[int, ...], reshaped: tuple[int, ...]
) -> None:
    """Scalar source, scalar destination, and a non-scalar control at both targets."""
    a = line_schedule.Index("a")
    indices = (a,) if reshaped else ()
    kwargs: dict[str, Any] = {}
    if target == "present":
        core = line_schedule.Schedule({a: 2})
        args: tuple[Any, ...] = (core, shape, indices)
        kwargs = {"view": reshaped}
        called = Target("method", "finn.dataflow.schedule", "Schedule", target)
        with pytest.raises(line_schedule.Refused) as error:
            core.present(shape, indices, **kwargs)
    else:
        args = ([line_schedule.Access("x", shape, indices, reshaped=True)], {a: 2})
        called = Target("function", "finn.dataflow.schedule", "", target)
        with pytest.raises(line_schedule.Refused) as error:
            line_schedule.bind_extents(*args)
    assert error.value.findings[0].code == "view-size-mismatch"
    call = Call(called, args, kwargs, Raised.of(error.value))
    assert judge(call).verdict == "same"
    # Changing a shape in the recorded refusal must not disappear in translation.
    changed = replace(
        call, outcome=Raised(call.outcome.kind, "a (99,) tensor cannot be viewed as ()")
    )
    assert judge(changed).verdict == "differs"


@pytest.mark.parametrize(
    "code,owner,message",
    [
        ("other", "Schedule.present", "a (7,) tensor cannot be reshaped to (2, 3)"),
        ("reshape-size-mismatch", "other", "a (7,) tensor cannot be reshaped to (2, 3)"),
        ("reshape-size-mismatch", "Schedule.present", "a (8,) tensor cannot be reshaped to (2, 3)"),
        ("reshape-size-mismatch", "Schedule.present", "another refusal"),
    ],
)
def test_refusal_translation_does_not_hide_other_differences(
    monkeypatch: pytest.MonkeyPatch, code: str, owner: str, message: str
) -> None:
    call = _present_call((7,))

    def wrong(*args: Any, **kwargs: Any) -> Any:
        raise schedule.Refused(code, owner, message)

    monkeypatch.setattr(schedule.Schedule, "present", wrong)
    assert judge(call).verdict == "differs"


def test_the_changed_schedule_order_message_is_not_normalized() -> None:
    a = line_schedule.Index("a")
    with pytest.raises(ValueError) as error:
        line_schedule.Schedule({a: 2}, order=())
    call = Call(
        Target("method", "finn.dataflow.schedule", "Schedule", "__init__"),
        # Replay invokes the constructor on the translated instance; it refuses before writing.
        (line_schedule.Schedule({a: 2}), {a: 2}),
        {"order": ()},
        Raised.of(error.value),
    )
    assert judge(call).verdict == "differs"
