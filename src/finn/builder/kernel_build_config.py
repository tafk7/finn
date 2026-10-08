# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's build configuration: a build of KernelOps (finn.custom_op.kernels)
from a streamlined model, through ``build_dataflow`` as a DataflowBuildConfig's build is.

It is composed of what the kernel path reads: its target (a board or a part, the clock
and the shell, stated once: ``finn.platform.TargetRequest``), its toolchain
(``finn.util.toolchain.Selection``), the outputs it makes, the strategies that explore
its choices and the policy that completes the rest. Nothing of the HWCustomOp flow's
configuration (folding, FIFO sizing, specialization, the Vitis and SLASH shells) is a
field of it: what the kernel path does not make, it cannot be asked for.
"""

from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin, Undefined, config
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from finn.builder.build_dataflow_config import declared
from finn.kernels.target import Target
from finn.platform import TargetRequest
from finn.util.toolchain import Selection, Toolchain, machine_selection


class KernelOutputType(str, Enum):
    """What a kernel-path build makes beside its partition and reports. Each needs a
    shell that integrates the partition (``finn.platform.ShellRow.integration``; not
    the ``ip`` shell's): the shell's bitfile, the driver its host runtime runs, and
    the deployment package of both. The names are DataflowOutputType's."""

    BITFILE = "bitfile"
    PYNQ_DRIVER = "pynq_driver"
    DEPLOYMENT_PACKAGE = "deployment_package"


class KernelVerificationStepType(str, Enum):
    """The checks step_verify_kernel_partition runs on the partition, as asked."""

    #: the partition's own outputs, the parent graph with the partition of KernelOps
    #: executed in Python, against the model the build started from, on each
    #: verify_input_npy input
    PARTITION_PYTHON = "kernel_partition_python"
    #: the partition's emitted RTL compiles and elaborates in XSim (xvlog, xelab);
    #: needs Vivado
    PARTITION_ELABORATION = "kernel_partition_elaboration"


#: The steps of a kernel-path build, from a streamlined model: the kernel-path phase
#: (the target, KernelOps, their choices, the partition, its verification), then the
#: outputs its shell makes.
default_kernel_build_steps = ["phase_kernel_path", "phase_kernel_outputs"]


# undefined=RAISE: a key the configuration does not declare is refused, named, when a
# configuration is read (from_json, from_dict), never dropped: a DataflowBuildConfig
# field stated here (target_fps, shell_flow_type, ...) is refused, not ignored.
@dataclass
class KernelBuildConfig(DataClassJsonMixin):
    """The configuration of a kernel-path build, passed to build_dataflow_cfg, or
    written as ``kernel_build_config.json`` beside ``model.onnx`` for
    build_dataflow_directory and the ``build_dataflow`` command. Serialized to and
    from JSON as DataflowBuildConfig is; reading one refuses a key it does not
    declare (``UndefinedParameterError``, naming the keys)."""

    dataclass_json_config = config(undefined=Undefined.RAISE)["dataclasses_json"]

    #: Directory where the build's outputs and reports are written
    output_dir: str

    #: The target: the clock period, a board or a part (a board gives its part; a
    #: part beside it is an assertion) and the shell, ``ip`` unless one is stated
    #: (finn.platform.shells: ``ip``, the packaged IP its user integrates; ``pynq``,
    #: the Zynq block design for a board). In JSON:
    #: {"period_ns": 5.0, "board": "Ultra96", "part": null, "shell": "pynq"}.
    target: TargetRequest = field(metadata=config(decoder=declared(TargetRequest, "target")))

    #: What the build makes beside the partition and its reports (KernelOutputType);
    #: each needs a shell that integrates the partition. None by default.
    generate_outputs: List[KernelOutputType] = field(default_factory=list)

    #: The exploration (step_kernel_choices): the strategies that choose the KernelOps'
    #: open choices through the DSE seam, run as written, each a spec with its own
    #: parameters (finn.transformation.kernels.KERNEL_STRATEGIES):
    #: ``{"strategy": "pinned", "path": ...}`` (a kernel_choices.json),
    #: ``{"strategy": "target_throughput", "fps": ..., "relax": true}`` (the least
    #: parallelism meeting fps at the target's clock), ``{"strategy": "size_fifos",
    #: "margin": 0}`` (each channel's FIFO from both ends' beat patterns). None by
    #: default: what no strategy chooses stays open, and kernel_completion completes it.
    kernel_exploration: List[Dict[str, Any]] = field(default_factory=list)

    #: Clear the KernelOps' saved choices before exploring: re-explore from scratch.
    #: Otherwise a saved choice is pinned and the exploration fills only open ones.
    kernel_exploration_fresh: bool = False

    #: The completion policy (finn.transformation.kernels.KERNEL_COMPLETIONS) that
    #: completes what no strategy chose, on a copy that is never saved, wherever the
    #: partition is costed or built: ``baseline``, every open choice at its kernel's
    #: baseline (its first viable case: the least parallelism, ``auto`` memories) and,
    #: when the partition is built, its FIFOs sized at that folding; a required choice
    #: (a FIFO's depth) is refused, named. ``placeholder``, for debugging, also takes
    #: the first case of a required choice. report/kernel_exploration.json lists every
    #: value completed.
    kernel_completion: str = "baseline"

    #: The checks step_verify_kernel_partition runs (KernelVerificationStepType).
    verify_steps: List[KernelVerificationStepType] = field(default_factory=list)

    #: The .npy file of inputs PARTITION_PYTHON runs the partition and the source on,
    #: one frame per index of its first axis.
    verify_input_npy: str = "input.npy"

    #: The AMD tool installation every tool step of the build runs in, and the
    #: environment of build_dataflow_directory's build process; unset (None), the
    #: machine's. As DataflowBuildConfig.toolchain. It also says how many runs Vivado
    #: launches at once when it builds the shell's bitfile (``vivado_jobs``).
    toolchain: Optional[Selection] = field(
        default=None, metadata=config(decoder=declared(Selection, "toolchain"))
    )

    #: Insert debug cores (ILA) in the shell's bitfile build (ZynqBuild's enable_debug).
    enable_hw_debug: bool = False

    #: If given, only run the steps in the list, each a step or phase name of the
    #: kernel path (finn.builder.kernel_build_steps) or a function called with
    #: (model, KernelBuildConfig); by default default_kernel_build_steps.
    steps: Optional[List[Any]] = None

    #: If given, start from this step or phase, loading the intermediate model saved
    #: before it (save_intermediate_models must be enabled).
    start_step: Optional[str] = None

    #: If given, stop after this step or phase.
    stop_step: Optional[str] = None

    #: Whether each step's model is saved under intermediate_models.
    save_intermediate_models: bool = True

    #: Whether pdb postmortem debugging is launched when the build fails.
    enable_build_pdb_debug: bool = True

    #: Whether every step's output is printed to stdout, not only to the build log.
    verbose: bool = False

    #: Functions to run after named steps or phases, as DataflowBuildConfig's.
    inject_steps_after: Dict[str, List[Callable]] = field(default_factory=dict)

    #: Functions to run before named steps or phases, as DataflowBuildConfig's.
    inject_steps_before: Dict[str, List[Callable]] = field(default_factory=dict)

    def _resolve_target(self) -> Target:
        """The build's target (finn.platform.resolve_target), every refusal named."""
        return self.target.resolve()

    def _resolve_selection(self) -> Selection:
        """The selection this build runs its tools by: ``toolchain``, or the machine's."""
        return machine_selection() if self.toolchain is None else self.toolchain

    def _resolve_toolchain(self) -> Toolchain:
        """The prepared toolchain every tool step of this build runs in, prepared once
        and kept on the instance (not a field: it is prepared, not configured)."""
        toolchain = getattr(self, "_toolchain", None)
        if toolchain is None:
            toolchain = self._toolchain = self._resolve_selection().prepare()
        return toolchain
