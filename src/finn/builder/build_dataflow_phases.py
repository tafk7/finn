# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's phases. Each phase names a list of steps of finn.builder.build_dataflow_steps,
run in that order. A configuration's ``steps``, ``start_step``, ``stop_step`` and
``inject_steps_before``/``inject_steps_after`` can name a phase wherever they name a step.
build_dataflow expands each phase into its steps, so each step is still run, injected
around, saved, timed, started from and stopped at on its own."""

#: The build's phases by name, in the order a build runs them, each with its steps in
#: order.
build_dataflow_phase_lookup: dict[str, list[str]] = {
    # Bring the export into FINN's form and clean it for the compiler without changing
    # its computation (finn.transformation.prepare), then check the prepared graph.
    "phase_prepare_model": [
        "step_prepare_import",
        "step_prepare_quantization",
        "step_prepare_io",
        "step_prepare_streamlining",
        "step_prepare_topology",
        "step_prepare_containers",
        "step_prepare_checkpoint",
    ],
    # Record the target in the model and convert the graph's nodes to KernelOps.
    "phase_convert_to_hardware": ["step_convert_to_hw"],
    # Cut the KernelOps into one partition; the other nodes stay on the host.
    "phase_create_accelerator": ["step_create_dataflow_partition"],
    # Choose the partition's parameters with the configured strategies, then check it.
    "phase_optimize_hardware": ["step_explore", "step_verify_partition"],
    # Generate the partition's HDL and package it as an IP.
    "phase_build_hardware": ["step_create_stitched_ip"],
    # Make the outputs the configuration asks for: bitfile, driver, resources report,
    # deployment package.
    "phase_generate_outputs": [
        "step_synthesize_bitfile",
        "step_make_driver",
        "step_resources_report",
        "step_deployment_package",
    ],
}
