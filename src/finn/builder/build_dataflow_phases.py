# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's phases: each a name for a list of steps of finn.builder.build_dataflow_steps,
run in that order. A DataflowBuildConfig's ``steps``, ``start_step``, ``stop_step`` and
``inject_steps_before``/``inject_steps_after`` may name a phase wherever they name a step;
build_dataflow expands each phase into its steps, so every step is run, injected around,
saved, timed, started from and stopped at as one step of the build."""

#: The build's phases by name, in the order a build runs them, each the names of its
#: steps in order.
build_dataflow_phase_lookup: dict[str, list[str]] = {
    # The export in FINN's form, cleaned for the compiler, its computation unchanged
    # (finn.transformation.prepare); the prepared graph checked.
    "phase_prepare_model": [
        "step_prepare_import",
        "step_prepare_quantization",
        "step_prepare_io",
        "step_prepare_streamlining",
        "step_prepare_topology",
        "step_prepare_containers",
        "step_prepare_checkpoint",
    ],
    # The target stated in the model; the graph's nodes rewritten to KernelOps.
    "phase_convert_to_hardware": ["step_convert_to_hw"],
    # The KernelOps cut once into one partition, the rest on the host.
    "phase_create_accelerator": ["step_create_dataflow_partition"],
    # The partition's choices explored by the configured strategies; the partition checked.
    "phase_optimize_hardware": ["step_explore", "step_verify_partition"],
    # The partition's IP: its HDL rendered and packaged.
    "phase_build_hardware": ["step_create_stitched_ip"],
    # The outputs the configuration asks: bitfile, driver, resources, deployment package.
    "phase_generate_outputs": [
        "step_synthesize_bitfile",
        "step_make_driver",
        "step_resources_report",
        "step_deployment_package",
    ],
}
