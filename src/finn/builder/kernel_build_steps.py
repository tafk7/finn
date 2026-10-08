# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's steps and phases, each a ``(model, KernelBuildConfig) -> model``
function, registered by name in ``kernel_build_step_lookup`` for a KernelBuildConfig's
``steps`` (finn.builder.kernel_build_config).

``phase_kernel_path`` takes a streamlined model to a partition of KernelOps
(finn.custom_op.kernels): the KernelOps bind kernels to RTL from the model's facts and
the build's target, so there is no specialization, folding config or per-node IP
generation. ``phase_kernel_outputs`` then makes what the configuration asks of the
target's shell (its bitfile, driver and deployment package)."""

import json
import numpy as np
import os
import shutil
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_phases import _execute_step
from finn.builder.build_dataflow_steps import collect_zynq_bitfile, deployment_package
from finn.builder.kernel_build_config import (
    KernelBuildConfig,
    KernelOutputType,
    KernelVerificationStepType,
)
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.base import read_target
from finn.platform import refuse_drift, shell_row
from finn.transformation.fpgadataflow.cut_kernel_partition import CutKernelPartition
from finn.transformation.fpgadataflow.kernel_partitions import (
    OUTPUT_BITFILE,
    OUTPUT_HOST_RUNTIME,
    OUTPUT_HWH,
    OUTPUT_PROJECT,
    OUTPUT_REPORTS,
    partition_body,
)
from finn.transformation.fpgadataflow.make_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild, kernel_link_graph
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    completion,
    explore_kernel_choices,
    kernel_choices_config,
    partition_bottleneck,
    strategy,
)
from finn.transformation.kernels.integration import VIVADO_BLOCK_DESIGN, integration
from finn.transformation.kernels.package import ElaboratePartition, configured_root


def _kernel_path_source(cfg: KernelBuildConfig) -> str:
    """Where step_kernel_ops keeps the model it converts, the reference of the
    partition's Python check."""
    return cfg.output_dir + "/intermediate_models/kernel_path_source.onnx"


def step_kernel_ops(model: ModelWrapper, cfg: KernelBuildConfig):
    """State the build's target in the model (``cfg.target``, resolved) and rewrite
    each node a KernelOp binds (MatMul, MultiThreshold) as one: ToKernelOps. When
    PARTITION_PYTHON is asked, the model it converts is kept as that check's
    reference (kernel_path_source.onnx)."""
    if KernelVerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        os.makedirs(os.path.dirname(_kernel_path_source(cfg)), exist_ok=True)
        model.save(_kernel_path_source(cfg))
    return model.transform(ToKernelOps(cfg._resolve_target()))


def step_infer_kernel_tensors(model: ModelWrapper, cfg: KernelBuildConfig):
    """Infer every tensor in graph order, the KernelOps answering from their kernels."""
    return model.transform(InferKernelTensors())


def step_kernel_choices(model: ModelWrapper, cfg: KernelBuildConfig):
    """Explore the KernelOps' open choices through the DSE seam by the strategies
    ``cfg.kernel_exploration`` lists, in order (finn.transformation.kernels.strategy:
    each a spec, {"strategy": name, **parameters}), and save them on their nodes
    (explore_kernel_choices): the choices strategies made, never a completed one. What
    no strategy chose stays open, and ``cfg.kernel_completion`` completes it wherever
    the partition is costed or built. A strategy carries its own objective. Writes
    report/kernel_exploration.json (the strategies, each with the choices it
    committed, attempts and time, and the completed values it read; every committed
    choice with the strategy that made it; the completion policy, every value it
    completes and the required choices it leaves open; whether FIFOs were sized; the
    dropped choices with why, per member cycles and buffering and the bottleneck, of
    the completed point) and kernel_choices.json (the nodes' choices, sparse,
    ApplyConfig's form, which a "pinned" strategy reads back), and logs what each
    strategy committed, what the completion completed (each value, for the debug
    placeholder), and whether FIFOs were sized."""
    strategies = [strategy(spec) for spec in cfg.kernel_exploration]
    explored = explore_kernel_choices(
        model,
        strategies,
        fresh=cfg.kernel_exploration_fresh,
        completion=completion(cfg.kernel_completion),
    )
    os.makedirs(cfg.output_dir + "/report", exist_ok=True)
    with open(cfg.output_dir + "/report/kernel_exploration.json", "w") as f:
        json.dump(explored.report, f, indent=2)
    with open(cfg.output_dir + "/kernel_choices.json", "w") as f:
        json.dump(kernel_choices_config(model), f, indent=2)
    report = explored.report
    for each in report["strategies"]:
        print(f"Kernel choices: {each['strategy']} committed {each['committed']}")
        if "read_completed" in each:
            print(f"Kernel choices: {each['strategy']} {each['read_completed']}")
    completed = [
        (f"{node}.{attribute}", entry)
        for node, held in report["completed"].items()
        for attribute, entry in held.items()
    ]
    policy = report["completion"]
    print(f"Kernel choices: {policy['policy']} completes {len(completed)}")
    for name, entry in completed:
        if entry["by"].startswith("DEBUG"):
            print(f"{entry['by']}: {name} = {entry['value']!r}")
    if "refused" in policy:
        print(f"Kernel choices: the {policy['policy']} completion refuses: {policy['refused']}")
    if policy.get("open"):
        print(f"Kernel choices: open, to choose before packaging: {', '.join(policy['open'])}")
    print(f"FIFOs: {report['fifos']}")
    return model


def _partition_directory(cfg: KernelBuildConfig) -> Path:
    """Where the kernel path keeps its partition's body (``partition.onnx``) and what its
    shell's integration links around it."""
    return Path(cfg.output_dir) / "partition"


def step_kernel_partition(model: ModelWrapper, cfg: KernelBuildConfig):
    """Cut the KernelOps once into one StreamingDataflowPartition, the nodes before and
    after it on the host (CutKernelPartition). The parent graph is the build's model
    from here on; the partition is named ``partition`` (its node, its body
    partition/partition.onnx, its IP and its block-design instance), and its body
    carries the target and states its boundary, the ends' facts (finn.partition),
    completed by ``cfg.kernel_completion``."""
    return model.transform(
        CutKernelPartition(_partition_directory(cfg), completion(cfg.kernel_completion))
    )


def verify_kernel_partition_python(model: ModelWrapper, cfg: KernelBuildConfig) -> bool:
    """The partition's own outputs, for each input ``verify_input_npy`` states: the
    parent graph ``model`` executed, its partition node running its body of KernelOps,
    against the model the kernel path started from (step_kernel_ops'
    kernel_path_source.onnx) on the same input, value for value. Each input's partition
    outputs are saved as
    verification_output/verify_kernel_partition_python_<index>_<SUCCESS|FAIL>.npz."""
    source_file = _kernel_path_source(cfg)
    if not os.path.isfile(source_file):
        raise FileNotFoundError(
            f"{source_file}: the kernel path's source, which step_kernel_ops keeps when "
            "PARTITION_PYTHON is asked, is not there (did the build start after it?)"
        )
    verify_out_dir = cfg.output_dir + "/verification_output"
    os.makedirs(verify_out_dir, exist_ok=True)
    node, _, _ = partition_body(model)
    source = ModelWrapper(source_file)
    source_input = source.graph.input[0].name
    ishape = tuple(source.get_tensor_shape(source_input))
    outputs = list(node.output)
    inputs = np.load(cfg.verify_input_npy)
    all_match = True
    for index in range(inputs.shape[0]):
        frame = inputs[index : index + 1].reshape((1,) + ishape[1:])
        built = execute_onnx(model, {model.graph.input[0].name: frame}, True)
        expected = execute_onnx(source, {source_input: frame}, True)
        mismatched = [name for name in outputs if not np.array_equal(built[name], expected[name])]
        for name in mismatched:
            print(
                f"kernel_partition_python: input {index}: {name} differs from the source's in "
                f"{int(np.sum(built[name] != expected[name]))} of {built[name].size} values"
            )
        result = "FAIL" if mismatched else "SUCCESS"
        all_match = all_match and not mismatched
        np.savez(
            f"{verify_out_dir}/verify_kernel_partition_python_{index}_{result}.npz",
            **{name: built[name] for name in outputs},
        )
    print(
        f"kernel_partition_python: the partition's outputs {outputs} on {inputs.shape[0]} "
        f"inputs of {cfg.verify_input_npy}, against the model the kernel path started from"
    )
    return all_match


def step_verify_kernel_partition(model: ModelWrapper, cfg: KernelBuildConfig):
    """Check the kernel path's partition before a shell builds it, its body opened
    through the parent graph's partition node.

    Always: its choices replay to a configured root, completed by
    ``cfg.kernel_completion`` (none stale, none left open, its ports its graph's
    inputs and outputs in order; configured_root), and it states the configuration's
    target (refuse_drift). As ``verify_steps`` asks: PARTITION_PYTHON compares the
    partition's own outputs, with the parent graph executed, against the model the
    kernel path started from, on each input verify_input_npy states
    (verify_kernel_partition_python); PARTITION_ELABORATION compiles and elaborates
    the partition's emitted RTL in XSim, through the build's toolchain."""
    node, body, _ = partition_body(model)
    configured_root(body, node.name, completion(cfg.kernel_completion))
    refuse_drift(read_target(body), cfg._resolve_target(), "build")
    if KernelVerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        matched = verify_kernel_partition_python(model, cfg)
        print("Verification for kernel_partition_python : " + ("SUCCESS" if matched else "FAIL"))
    if KernelVerificationStepType.PARTITION_ELABORATION in cfg.verify_steps:
        directory = cfg.output_dir + "/verification_output/kernel_partition_elaboration"
        body.transform(
            ElaboratePartition(
                directory=Path(directory),
                toolchain=cfg._resolve_toolchain(),
                completion=completion(cfg.kernel_completion),
            )
        )
        print("Verification for kernel_partition_elaboration : SUCCESS")
    return model


def _objective_fps(cfg: KernelBuildConfig):
    """The throughput the exploration's target_throughput strategy asks, if any."""
    asked = [
        spec["fps"]
        for spec in cfg.kernel_exploration
        if spec.get("strategy") == "target_throughput" and "fps" in spec
    ]
    return asked[0] if asked else None


def _integrating_row(cfg: KernelBuildConfig, output: KernelOutputType):
    """The target's shell row, which must integrate the partition to make ``output``."""
    target = cfg._resolve_target()
    row = shell_row(target.shell, target.board)
    if row.integration is None:
        raise ValueError(
            f"{output.value}: the {target.shell!r} shell does not integrate the partition; "
            "state a shell that does (pynq, for a board)"
        )
    return target, row


def step_kernel_bitfile(model: ModelWrapper, cfg: KernelBuildConfig):
    """Build the target's shell around the partition to a bitfile, by the shell's
    integration, if BITFILE is asked: the Zynq block design (ZynqBuild) from the
    partition's integration export (finn.transformation.kernels.integration, written as
    report/integration.json), the partition completed by ``cfg.kernel_completion``. The
    bitfile, its hardware handoff and reports go to the output directory
    (collect_zynq_bitfile); the delivered clock's report states the partition's
    bottleneck, ends included, and the throughput the exploration asked. The parent
    graph stays the build's model and states what was made (finn.outputs: the
    project, bitfile, hwh, reports and host runtime). Vivado launches as many runs at
    once as the toolchain's selection says (``Selection.vivado_jobs``)."""
    if KernelOutputType.BITFILE not in cfg.generate_outputs:
        print("BITFILE not in requested outputs, skipping step_kernel_bitfile.")
        return model
    target, row = _integrating_row(cfg, KernelOutputType.BITFILE)
    if row.integration != VIVADO_BLOCK_DESIGN:
        raise ValueError(f"bitfile: no build for the {row.integration!r} integration")
    kernel_completion = completion(cfg.kernel_completion)
    _, body, _ = partition_body(model)
    bottleneck = partition_bottleneck(body, kernel_completion)
    export = integration(model, kernel_completion)
    report_dir = cfg.output_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)
    with open(report_dir + "/integration.json", "w") as f:
        json.dump(export.report(), f, indent=2)
    toolchain = cfg._resolve_toolchain()
    link = model.transform(
        ZynqBuild(
            target.board,
            target.platform.period_ns,
            cfg.enable_hw_debug,
            partition_model_dir=str(_partition_directory(cfg) / "zynq_link"),
            toolchain=toolchain,
            vivado_jobs=toolchain.selection.vivado_jobs,
            completion=kernel_completion,
            integration=export,
        )
    )
    collect_zynq_bitfile(
        link,
        cfg.output_dir,
        target.platform.period_ns,
        None if bottleneck is None else bottleneck.cycles,
        _objective_fps(cfg) if bottleneck is not None else None,
    )
    model.set(OUTPUT_PROJECT, link.get_metadata_prop("vivado_pynq_proj"))
    model.set(OUTPUT_BITFILE, cfg.output_dir + "/bitfile/finn-accel.bit")
    model.set(OUTPUT_HWH, cfg.output_dir + "/bitfile/finn-accel.hwh")
    model.set(
        OUTPUT_REPORTS,
        {
            "synthesis": report_dir + "/post_synth_resources.xml",
            "timing": report_dir + "/post_route_timing.rpt",
            "delivered_clock": report_dir + "/delivered_clock.json",
            "integration": report_dir + "/integration.json",
        },
    )
    if row.host_runtime is not None:
        model.set(OUTPUT_HOST_RUNTIME, row.host_runtime)
    return model


def step_kernel_driver(model: ModelWrapper, cfg: KernelBuildConfig):
    """Write the driver the target's shell's host runtime runs (the PYNQ driver), if
    PYNQ_DRIVER is asked, from the Zynq block design's link graph of the partition's
    integration export (kernel_link_graph); the parent graph stays the build's model."""
    if KernelOutputType.PYNQ_DRIVER not in cfg.generate_outputs:
        print("PYNQ_DRIVER not in requested outputs, skipping step_kernel_driver.")
        return model
    _, row = _integrating_row(cfg, KernelOutputType.PYNQ_DRIVER)
    driver_dir = os.path.join(cfg.output_dir, "driver")
    export = integration(model, completion(cfg.kernel_completion))
    link = kernel_link_graph(model, export, _partition_directory(cfg) / "driver_link")
    link = link.transform(MakePYNQDriver(row.host_runtime))
    shutil.copytree(link.get_metadata_prop("pynq_driver_dir"), driver_dir, dirs_exist_ok=True)
    print("PYNQ Python driver written into " + driver_dir)
    return model


def step_kernel_deployment_package(model: ModelWrapper, cfg: KernelBuildConfig):
    """Package the bitfile and the driver for deployment, if DEPLOYMENT_PACKAGE is asked."""
    if KernelOutputType.DEPLOYMENT_PACKAGE not in cfg.generate_outputs:
        print("DEPLOYMENT_PACKAGE not in requested outputs, skipping.")
        return model
    _integrating_row(cfg, KernelOutputType.DEPLOYMENT_PACKAGE)
    deployment_package(cfg.output_dir)
    return model


def phase_kernel_path(model: ModelWrapper, cfg: KernelBuildConfig):
    """Phase: the kernel path, from a streamlined model to a partition of KernelOps.

    Internal steps:
    - step_kernel_ops: State the build target in the model, rewrite to KernelOps
    - step_infer_kernel_tensors: Infer every tensor from the kernels
    - step_kernel_choices: Commit the open choices by the configured strategies
    - step_kernel_partition: The KernelOps cut once into one partition, the rest on the host
    - step_verify_kernel_partition: Check the partition (and what verify_steps asks)

    Returns the parent graph: the host's nodes and the partition node, whose body
    holds the KernelOps, their choices committed."""
    model = _execute_step(step_kernel_ops, model, cfg)
    model = _execute_step(step_infer_kernel_tensors, model, cfg)
    model = _execute_step(step_kernel_choices, model, cfg)
    model = _execute_step(step_kernel_partition, model, cfg)
    model = _execute_step(step_verify_kernel_partition, model, cfg)
    return model


def phase_kernel_outputs(model: ModelWrapper, cfg: KernelBuildConfig):
    """Phase: what the configuration asks of the target's shell (generate_outputs).

    Internal steps (each checks generate_outputs):
    - step_kernel_bitfile: The shell built around the partition, to a bitfile
    - step_kernel_driver: The driver of the shell's host runtime
    - step_kernel_deployment_package: The bitfile and driver, packaged"""
    model = _execute_step(step_kernel_bitfile, model, cfg)
    model = _execute_step(step_kernel_driver, model, cfg)
    model = _execute_step(step_kernel_deployment_package, model, cfg)
    return model


#: The kernel path's steps and phases by name, for a KernelBuildConfig's ``steps``.
kernel_build_step_lookup = {
    step.__name__: step
    for step in (
        step_kernel_ops,
        step_infer_kernel_tensors,
        step_kernel_choices,
        step_kernel_partition,
        step_verify_kernel_partition,
        step_kernel_bitfile,
        step_kernel_driver,
        step_kernel_deployment_package,
        phase_kernel_path,
        phase_kernel_outputs,
    )
}
