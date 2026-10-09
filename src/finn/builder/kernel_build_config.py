# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's build configuration: a build of KernelOps (finn.custom_op.kernels)
from a Brevitas export, through ``build_dataflow``.

It is composed of what the kernel path reads: how the export is prepared
(``finn.transformation.prepare.GraphPreparation``), its target (a board or a part, the clock
and the shell, stated once: ``finn.platform.TargetRequest``), its toolchain
(``finn.util.toolchain.Selection``), the outputs it makes, the strategies that explore
its choices and the policy that completes the rest. What the kernel path does not make,
it cannot be asked for.
"""

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin, Undefined, config
from dataclasses_json.undefined import UndefinedParameterError
from qonnx.core.modelwrapper import ModelWrapper

from finn.kernels.target import Target
from finn.platform import VIVADO_BLOCK_DESIGN, TargetRequest, shell_row
from finn.shells.pynq.runner import PynqOptions
from finn.transformation.prepare import GraphPreparation
from finn.util.toolchain import Selection, Toolchain, machine_selection


def declared(cls: type, name: str) -> Callable[[Any], Any]:
    """The decoder of the nested dataclass ``cls`` a configuration states as ``name``,
    refusing keys ``cls`` does not declare (dataclasses_json would drop them, as it
    does a nested dataclass's), naming them."""

    def decode(stated: Any) -> Any:
        if stated is None or isinstance(stated, cls):
            return stated
        unknown = sorted(set(stated) - {item.name for item in fields(cls)})
        if unknown:
            raise UndefinedParameterError(
                f"{name}: keys {cls.__name__} does not declare: {unknown}"
            )
        return cls(**stated)

    return decode


class KernelOutputType(str, Enum):
    """What a kernel-path build makes beside its partition and reports. The values are
    what ``finn.outputs`` records.

    The partition's own, on every shell: ``stitched_ip``, the shell root's packaged IP
    with its interface description and its XSim testbench, written and not run (the
    verification step STITCHED_IP_TESTBENCH runs it); ``ooc_synth``, the IP
    synthesized out of context, with its resources per member of the shell root.

    The shell's, each needing a shell that integrates the partition
    (``finn.platform.ShellRow.integration``; not the ``ip`` shell's): its bitfile, the
    driver its host runtime runs, and the deployment package of both
    (``SHELL_OUTPUTS``). The driver sets the clock its bitfile delivers, and the
    deployment ships both: each needs the outputs it is made from asked with it
    (``OUTPUT_NEEDS``)."""

    STITCHED_IP = "stitched_ip"
    OOC_SYNTH = "ooc_synth"
    BITFILE = "bitfile"
    PYNQ_DRIVER = "pynq_driver"
    DEPLOYMENT_PACKAGE = "deployment_package"


#: The outputs only a shell that integrates the partition makes.
SHELL_OUTPUTS = (
    KernelOutputType.BITFILE,
    KernelOutputType.PYNQ_DRIVER,
    KernelOutputType.DEPLOYMENT_PACKAGE,
)

#: The outputs each output is made from, which a build asking for it asks for too: the
#: driver from its bitfile's delivered clock, the deployment from the bitfile and the
#: driver of the same build.
OUTPUT_NEEDS = {
    KernelOutputType.PYNQ_DRIVER: (KernelOutputType.BITFILE,),
    KernelOutputType.DEPLOYMENT_PACKAGE: (KernelOutputType.BITFILE, KernelOutputType.PYNQ_DRIVER),
}


class KernelVerificationStepType(str, Enum):
    """The checks a build runs beside those it always runs, as asked: none by
    default, since no build re-verifies the partition's computation (its KernelOps'
    ``execute_node`` is the reference, and the harness checks the hardware against
    it). step_prepare_checkpoint runs the prepared graph's; step_verify_kernel_partition
    the partition's; step_kernel_stitched_ip its testbench's. The Python checks
    (GRAPH_PREPARATION_PYTHON, PARTITION_PYTHON) report a failure as FAIL and the
    build continues, but for a prepared annotation a drawn value breaks, which
    refuses the graph; the RTL checks (PARTITION_ELABORATION, STITCHED_IP_TESTBENCH)
    stop it."""

    #: the prepared graph's outputs against the export's, its inputs and outputs as
    #: the preparation states them, on seeded draws from the prepared input's
    #: annotation (finn.harness.preparation), and its integer annotations' soundness
    #: on what it computes there
    GRAPH_PREPARATION_PYTHON = "graph_preparation_python"

    #: the partition's own outputs, the parent graph with the partition of KernelOps
    #: executed in Python, against the model the build started from, on each
    #: verify_input_npy input
    PARTITION_PYTHON = "kernel_partition_python"
    #: the partition's emitted RTL compiles and elaborates in XSim (xvlog, xelab);
    #: needs Vivado
    PARTITION_ELABORATION = "kernel_partition_elaboration"
    #: the packaged IP's XSim testbench (STITCHED_IP's, its run.sh) run once, through
    #: the build's toolchain: its module's outputs on the testbench's frame against the
    #: partition's in Python; needs STITCHED_IP and Vivado
    STITCHED_IP_TESTBENCH = "stitched_ip_testbench"


#: The steps each verification needs to run, each a step or its phase: the one that
#: keeps its reference, and the one that checks.
VERIFIED_BY = {
    KernelVerificationStepType.GRAPH_PREPARATION_PYTHON: (
        ("phase_graph_preparation", "step_prepare_import"),
        ("phase_graph_preparation", "step_prepare_checkpoint"),
    ),
    KernelVerificationStepType.PARTITION_PYTHON: (
        ("phase_kernel_path", "step_kernel_ops"),
        ("phase_kernel_path", "step_verify_kernel_partition"),
    ),
    KernelVerificationStepType.PARTITION_ELABORATION: (
        ("phase_kernel_path", "step_verify_kernel_partition"),
    ),
    KernelVerificationStepType.STITCHED_IP_TESTBENCH: (
        ("phase_kernel_outputs", "step_kernel_stitched_ip"),
    ),
}


#: The steps of a kernel-path build, from a Brevitas export: the graph-preparation
#: phase (the export to a streamlined graph, checked), the kernel-path phase (the
#: target, KernelOps, their choices, the partition, its verification), then the
#: outputs its shell makes.
default_kernel_build_steps = [
    "phase_graph_preparation",
    "phase_kernel_path",
    "phase_kernel_outputs",
]


# undefined=RAISE: a key the configuration does not declare is refused, named, when a
# configuration is read (from_json, from_dict), never dropped.
@dataclass
class KernelBuildConfig(DataClassJsonMixin):
    """The configuration of a kernel-path build, passed to build_dataflow_cfg, or
    written as ``kernel_build_config.json`` beside ``model.onnx`` for
    build_dataflow_directory and the ``build_dataflow`` command. Serialized to and
    from JSON; reading one refuses a key it does not declare
    (``UndefinedParameterError``, naming the keys)."""

    # dataclasses_json types the hook None, the value its dataclass_json decorator sets.
    dataclass_json_config = config(undefined=Undefined.RAISE)["dataclasses_json"]  # type: ignore[assignment]

    #: Directory where the build's outputs and reports are written
    output_dir: str

    #: The target: the clock period, a board or a part (a board gives its part; a
    #: part beside it is an assertion) and the shell, ``ip`` unless one is stated
    #: (finn.platform.shells: ``ip``, the packaged IP its user integrates; ``pynq``,
    #: the Zynq block design for a board). In JSON:
    #: {"period_ns": 5.0, "board": "Ultra96", "part": null, "shell": "pynq"}.
    target: TargetRequest = field(metadata=config(decoder=declared(TargetRequest, "target")))

    #: How the export is prepared (phase_graph_preparation), a
    #: finn.transformation.prepare.GraphPreparation: P0's ``override_inpsize``, P2's
    #: ``max_multithreshold_bit_width``, P1's ``preprocessing`` (an ONNX model merged
    #: ahead of the network), ``input_datatype`` and ``topk``, the ``streamlining`` and
    #: ``topology`` recipes. In JSON, for TFC: {"preprocessing": "preproc.onnx",
    #: "input_datatype": "UINT8", "topk": 1}. By default the export's own inputs and
    #: outputs, the default recipes.
    preparation: GraphPreparation = field(
        default_factory=GraphPreparation,
        metadata=config(decoder=declared(GraphPreparation, "preparation")),
    )

    #: What the build makes beside the partition and its reports (KernelOutputType):
    #: the partition's IP and its out-of-context resources on any shell; the bitfile,
    #: driver and deployment package on a shell that integrates the partition. None by
    #: default.
    generate_outputs: List[KernelOutputType] = field(default_factory=list)

    #: The exploration (step_kernel_choices): the strategies that choose the KernelOps'
    #: open choices through the DSE seam, run as written, each a spec with its own
    #: parameters (finn.transformation.kernels.KERNEL_STRATEGIES):
    #: ``{"strategy": "pinned", "path": ...}`` (a kernel_choices.json),
    #: ``{"strategy": "target_throughput", "fps": ..., "relax": true}`` (the least
    #: parallelism meeting fps at the target's clock), ``{"strategy":
    #: "max_throughput", "within": {"lut": 0.5}}`` (the fewest cycles a frame whose
    #: shell's resources stay within the fractions named of the part's; ``within`` is
    #: required), ``{"strategy": "size_fifos", "margin": 0}`` (each channel's FIFO from
    #: both ends' beat patterns). None by default: what no strategy chooses stays open,
    #: and kernel_completion completes it.
    kernel_exploration: List[Dict[str, Any]] = field(default_factory=list)

    #: Clear the KernelOps' saved choices before exploring: re-explore from scratch.
    #: Otherwise a saved choice is pinned and the exploration fills only open ones.
    kernel_exploration_fresh: bool = False

    #: The completion policy (finn.transformation.kernels.KERNEL_COMPLETIONS) that
    #: completes what no strategy chose, on a copy that is never saved, wherever the
    #: partition is costed or built: ``baseline``, every open choice at its kernel's
    #: baseline (its first viable case: the least parallelism, each memory in the
    #: explicit style its size orders first, a FIFO in FinnLib's ``auto``) and,
    #: when the partition is built, its FIFOs sized at that folding; a required choice
    #: (a FIFO's depth) is refused, named. ``placeholder``, for debugging, also takes
    #: the first case of a required choice. report/kernel_exploration.json lists every
    #: value completed.
    kernel_completion: str = "baseline"

    #: The checks the build runs beside those it always runs
    #: (KernelVerificationStepType), none by default: step_prepare_checkpoint's,
    #: step_verify_kernel_partition's, and step_kernel_stitched_ip's testbench.
    verify_steps: List[KernelVerificationStepType] = field(default_factory=list)

    #: The .npy file of inputs PARTITION_PYTHON runs the partition and the source on,
    #: one frame per index of its first axis. STITCHED_IP's testbench streams its first
    #: frame when the file exists, a generated frame otherwise, and
    #: STITCHED_IP_TESTBENCH runs it.
    verify_input_npy: str = "input.npy"

    #: The AMD tool installation every tool step of the build runs in, and the
    #: environment of build_dataflow_directory's build process: the machine's
    #: (finn.util.toolchain.machine_selection), with what this selection states laid
    #: over it, each field it states winning. How many runs Vivado launches at once
    #: when it builds the shell's bitfile is the selection's ``vivado_jobs``, a machine
    #: setting (``FINN_VIVADO_JOBS``) a build may state over.
    toolchain: Optional[Selection] = field(
        default=None, metadata=config(decoder=declared(Selection, "toolchain"))
    )

    #: The options of the target's shell's build, as that build states them; a shell
    #: without a build (``ip``) takes none. The ``pynq`` shell's
    #: (finn.shells.pynq.runner.PynqOptions): ``enable_hw_debug``,
    #: integrated logic analyzers on the ends' streams. In JSON:
    #: {"enable_hw_debug": true}. None by default.
    shell_options: Dict[str, Any] = field(default_factory=dict)

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

    #: Whether pdb postmortem debugging is launched when the build fails and stdin is
    #: a terminal.
    enable_build_pdb_debug: bool = True

    #: Whether every step's output is printed to stdout, not only to the build log.
    verbose: bool = False

    #: Functions to run after named steps or phases.
    inject_steps_after: Dict[
        str, List[Callable[[ModelWrapper, "KernelBuildConfig"], ModelWrapper]]
    ] = field(default_factory=dict)

    #: Functions to run before named steps or phases.
    inject_steps_before: Dict[
        str, List[Callable[[ModelWrapper, "KernelBuildConfig"], ModelWrapper]]
    ] = field(default_factory=dict)

    def _resolve_selection(self) -> Selection:
        """The selection this build runs its tools by: ``toolchain`` laid over the
        machine's (``machine_selection``), each field it states winning; unset, the
        machine's."""
        return machine_selection(stated=self.toolchain)

    def _resolve_toolchain(self) -> Toolchain:
        """The prepared toolchain every tool step of this build runs in: the resolved
        selection, prepared by the first step that asks and then the same object for
        every later step. Kept on the instance, not a field: it is prepared, not
        configured, and is not serialized with the build configuration."""
        toolchain: Optional[Toolchain] = getattr(self, "_toolchain", None)
        if toolchain is None:
            toolchain = self._resolve_selection().prepare()
            self._toolchain = toolchain
        return toolchain

    def _resolve_target(self) -> Target:
        """The build's target (finn.platform.resolve_target), every refusal named."""
        return self.target.resolve()

    def _resolve_shell_options(self) -> Optional[PynqOptions]:
        """The target's shell's build options (``shell_options``), as its build reads
        them; None for a shell without a build, which refuses any. Each refusal is a
        ValueError, named."""
        target = self._resolve_target()
        if shell_row(target.shell, target.board).integration == VIVADO_BLOCK_DESIGN:
            return PynqOptions.from_dict(self.shell_options)
        if self.shell_options:
            raise ValueError(
                f"shell_options: the {target.shell!r} shell has no build to take "
                f"{', '.join(sorted(self.shell_options))}"
            )
        return None
