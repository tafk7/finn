# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How the kernel path runs a step inside one of its phases
(finn.builder.kernel_build_steps): the steps a KernelBuildConfig injects before and after
it, each step's model saved as an intermediate model, and each step's time recorded while
build_dataflow records them (``recorded_step_times``)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finn.builder.kernel_build_config import KernelBuildConfig

#: The seconds each step a phase runs took, by name, while a build records them
#: (``recorded_step_times``); None outside one.
_step_times: ContextVar[dict[str, float] | None] = ContextVar("_step_times", default=None)


@contextmanager
def recorded_step_times() -> Iterator[dict[str, float]]:
    """The seconds each step run through ``execute_step`` takes while the context is
    open, by step name (a step run twice adds up): the steps inside a phase, which the
    builder times only as the phase."""
    times: dict[str, float] = {}
    token = _step_times.set(times)
    try:
        yield times
    finally:
        _step_times.reset(token)


def _save_intermediate_model(model: ModelWrapper, step_name: str, cfg: KernelBuildConfig) -> None:
    """The model after ``step_name``, as intermediate_models/<step_name>.onnx."""
    intermediate_model_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(intermediate_model_dir, exist_ok=True)
    model.save(f"{intermediate_model_dir}/{step_name}.onnx")


def _timed(step_fn, model: ModelWrapper, cfg: KernelBuildConfig) -> ModelWrapper:
    """``step_fn`` run on the model, its time recorded if a build records step times."""
    started = time.time()
    model = step_fn(model, cfg)
    times = _step_times.get()
    if times is not None:
        name = step_fn.__name__
        times[name] = times.get(name, 0.0) + time.time() - started
    return model


def execute_step(step_fn, model: ModelWrapper, cfg: KernelBuildConfig) -> ModelWrapper:
    """Run ``step_fn`` inside a phase: the steps ``cfg.inject_steps_before`` names for it
    first and those ``cfg.inject_steps_after`` names after it, each one's model saved
    as an intermediate model if ``cfg.save_intermediate_models``, and each one timed
    (``recorded_step_times``)."""
    step_name = step_fn.__name__
    for injected_step in cfg.inject_steps_before.get(step_name, []):
        model = _timed(injected_step, model, cfg)
        if cfg.save_intermediate_models:
            _save_intermediate_model(model, injected_step.__name__, cfg)
    model = _timed(step_fn, model, cfg)
    if cfg.save_intermediate_models:
        _save_intermediate_model(model, step_name, cfg)
    for injected_step in cfg.inject_steps_after.get(step_name, []):
        model = _timed(injected_step, model, cfg)
        if cfg.save_intermediate_models:
            _save_intermediate_model(model, injected_step.__name__, cfg)
    return model


__all__ = ["execute_step", "recorded_step_times"]
