# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's phases, each a ``(model, DataflowBuildConfig) -> model`` function
that runs steps of finn.builder.build_dataflow_steps in order, registered by name in
``build_dataflow_phase_lookup`` for a DataflowBuildConfig's ``steps``; and how a phase runs
a step inside it (``_execute_step``): the steps a build configuration injects before and
after it, each step's model saved as an intermediate model, and each step's time
recorded while build_dataflow records them (``recorded_step_times``)."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import (
    step_kernel_bitfile,
    step_kernel_choices,
    step_kernel_deployment_package,
    step_kernel_driver,
    step_kernel_ops,
    step_kernel_partition,
    step_kernel_resources,
    step_kernel_stitched_ip,
    step_prepare_checkpoint,
    step_prepare_containers,
    step_prepare_import,
    step_prepare_io,
    step_prepare_quantization,
    step_prepare_streamlining,
    step_prepare_topology,
    step_verify_kernel_partition,
)

#: A step a phase runs, or one a build configuration injects.
_Step = Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]


#: The seconds each step a phase runs took, by name, while a build records them
#: (``recorded_step_times``); None outside one.
_step_times: ContextVar[dict[str, float] | None] = ContextVar("_step_times", default=None)


@contextmanager
def recorded_step_times() -> Iterator[dict[str, float]]:
    """The seconds each step run through ``_execute_step`` takes while the context is
    open, by step name (a step run twice adds up): the steps inside a phase, which the
    builder times only as the phase."""
    times: dict[str, float] = {}
    token = _step_times.set(times)
    try:
        yield times
    finally:
        _step_times.reset(token)


def _save_intermediate_model(model: ModelWrapper, step_name: str, cfg: DataflowBuildConfig) -> None:
    """The model after ``step_name``, as intermediate_models/<step_name>.onnx."""
    intermediate_model_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(intermediate_model_dir, exist_ok=True)
    model.save(f"{intermediate_model_dir}/{step_name}.onnx")


def _timed(step_fn: _Step, model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """``step_fn`` run on the model, its time recorded if a build records step times."""
    started = time.time()
    model = step_fn(model, cfg)
    times = _step_times.get()
    if times is not None:
        name = step_fn.__name__
        times[name] = times.get(name, 0.0) + time.time() - started
    return model


def _execute_step(step_fn: _Step, model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run ``step_fn`` inside a phase: the steps ``cfg.inject_steps_before`` names for it
    first and those ``cfg.inject_steps_after`` names after it, each one's model saved
    as an intermediate model if ``cfg.save_intermediate_models``, and each one timed
    (``recorded_step_times``). Injection names a phase or a step inside one:
    ``inject_steps_after={"step_kernel_choices": [my_func]}`` runs ``my_func`` after
    ``step_kernel_choices`` inside ``phase_kernel_path`` too."""
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


def phase_graph_preparation(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Phase: graph preparation, from a Brevitas export to the streamlined graph the
    kernel path converts, as ``cfg.preparation`` states it. It reads no target.

    Internal steps, in the order they run (the sub-phases of
    finn.transformation.prepare.phase):
    - step_prepare_import: P0, qonnx's cleanup
    - step_prepare_quantization: P2, Quant to MultiThreshold, weights to integers
    - step_prepare_io: P1, preprocessing merged, input datatype, label select
    - step_prepare_streamlining: P3, the streamlining recipe
    - step_prepare_topology: P4, the topology recipe
    - step_prepare_containers: P6, integer regions widened where float32 is not exact
    - step_prepare_checkpoint: P7, the prepared graph checked (and its equivalence
      with the export, as verify_steps asks)

    Writes report/graph_preparation.json."""
    for step in (
        step_prepare_import,
        step_prepare_quantization,
        step_prepare_io,
        step_prepare_streamlining,
        step_prepare_topology,
        step_prepare_containers,
        step_prepare_checkpoint,
    ):
        model = _execute_step(step, model, cfg)
    return model


def phase_kernel_path(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Phase: the kernel path, from a prepared model to a partition of KernelOps.

    Internal steps:
    - step_kernel_ops: State the build target in the model, rewrite to KernelOps,
      every tensor inferred as it goes
    - step_kernel_partition: The KernelOps cut once into one partition, the rest on the host
    - step_kernel_choices: Commit the partition's open choices by the configured strategies
    - step_verify_kernel_partition: Check the partition (and what verify_steps asks)

    Returns the parent graph: the host's nodes and the partition node, whose body
    holds the KernelOps, their choices committed."""
    model = _execute_step(step_kernel_ops, model, cfg)
    model = _execute_step(step_kernel_partition, model, cfg)
    model = _execute_step(step_kernel_choices, model, cfg)
    model = _execute_step(step_verify_kernel_partition, model, cfg)
    return model


def phase_kernel_outputs(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Phase: the outputs the configuration asks (generate_outputs).

    Internal steps (each but step_kernel_resources checks generate_outputs):
    - step_kernel_stitched_ip: The partition's IP, its description, resources, testbench
    - step_kernel_bitfile: The shell built around the partition, to a bitfile
    - step_kernel_driver: The driver of the shell's host runtime
    - step_kernel_resources: The resources per member of the shell
    - step_kernel_deployment_package: The bitfile and driver, packaged"""
    model = _execute_step(step_kernel_stitched_ip, model, cfg)
    model = _execute_step(step_kernel_bitfile, model, cfg)
    model = _execute_step(step_kernel_driver, model, cfg)
    model = _execute_step(step_kernel_resources, model, cfg)
    model = _execute_step(step_kernel_deployment_package, model, cfg)
    return model


#: The kernel path's phases by name, for a DataflowBuildConfig's ``steps``.
build_dataflow_phase_lookup = {
    phase.__name__: phase
    for phase in (
        phase_graph_preparation,
        phase_kernel_path,
        phase_kernel_outputs,
    )
}
