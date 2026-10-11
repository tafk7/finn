# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's build configuration: a build of KernelOps (finn.custom_op.kernels)
from a Brevitas export, through ``build_dataflow``.

It is composed of how the build runs, which every step may read: its target (a board or
a part, the clock and the shell, stated once: ``finn.platform.TargetRequest``), its
toolchain (``finn.toolchain.launcher.Selection``), the outputs it makes, the checks it
runs, which steps run; and each step's own options (``step_options``, keyed by step name,
each step declaring its options type beside it: how the export is prepared, the
strategies that replace the baselines its choices are created with). What the kernel
path does not make, it cannot be asked for.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dataclasses_json import DataClassJsonMixin, Undefined, config
from dataclasses_json.undefined import UndefinedParameterError
from qonnx.core.modelwrapper import ModelWrapper

from finn.kernels.target import Target
from finn.platform import VIVADO_BLOCK_DESIGN, TargetRequest, shell_row
from finn.shells.pynq.runner import PynqOptions
from finn.toolchain.launcher import Selection, Toolchain, host_selection

#: A step's options type (finn.builder.build_dataflow_steps.build_dataflow_step_options).
StepOptions = TypeVar("StepOptions")


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


class DataflowOutputType(str, Enum):
    """What a kernel-path build makes beside its partition and reports. The model
    records what a later step consumes (``finn.outputs``: the packaged IP, the bitfile);
    the build's manifest indexes the run (``report/build.json``).

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
    DataflowOutputType.BITFILE,
    DataflowOutputType.PYNQ_DRIVER,
    DataflowOutputType.DEPLOYMENT_PACKAGE,
)

#: The outputs each output is made from, which a build asking for it asks for too: the
#: driver from its bitfile's delivered clock, the deployment from the bitfile and the
#: driver of the same build.
OUTPUT_NEEDS = {
    DataflowOutputType.PYNQ_DRIVER: (DataflowOutputType.BITFILE,),
    DataflowOutputType.DEPLOYMENT_PACKAGE: (
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
    ),
}


class VerificationStepType(str, Enum):
    """The checks a build runs beside those it always runs, as asked: none by
    default, since no build re-verifies the partition's computation (its KernelOps'
    ``execute_node`` is the reference, and the harness checks the hardware against
    it). step_prepare_checkpoint runs the prepared graph's; step_verify_partition
    the partition's; step_create_stitched_ip its testbench's. The Python checks
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


#: The steps each verification needs to run: the one that keeps its reference, and
#: the one that checks.
VERIFIED_BY = {
    VerificationStepType.GRAPH_PREPARATION_PYTHON: (
        "step_prepare_import",
        "step_prepare_checkpoint",
    ),
    VerificationStepType.PARTITION_PYTHON: ("step_convert_to_hw", "step_verify_partition"),
    VerificationStepType.PARTITION_ELABORATION: ("step_verify_partition",),
    VerificationStepType.STITCHED_IP_TESTBENCH: ("step_create_stitched_ip",),
}


#: The steps of a kernel-path build, from a Brevitas export: its phases
#: (finn.builder.build_dataflow_phases), in order.
default_build_dataflow_steps = [
    "phase_prepare_model",
    "phase_convert_to_hardware",
    "phase_create_accelerator",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


# undefined=RAISE: a key the configuration does not declare is refused, named, when a
# configuration is read (from_json, from_dict), never dropped.
@dataclass
class DataflowBuildConfig(DataClassJsonMixin):
    """The configuration of a kernel-path build, passed to build_dataflow_cfg, or
    written as ``dataflow_build_config.json`` beside ``model.onnx`` for
    build_dataflow_directory and the ``build_dataflow`` command, where its relative
    paths name files in that directory (read_build_config). Its fields are how the
    build runs, and ``step_options`` each step's own. Serialized to and from JSON;
    reading one, or making one in Python, refuses a key it does not declare, a step in
    ``step_options`` that takes no options and an option a step does not declare
    (``UndefinedParameterError``, naming them)."""

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

    #: What the build makes beside the partition and its reports (DataflowOutputType):
    #: the partition's IP and its out-of-context resources on any shell; the bitfile,
    #: driver and deployment package on a shell that integrates the partition. None by
    #: default.
    generate_outputs: List[DataflowOutputType] = field(default_factory=list)

    #: The checks the build runs beside those it always runs
    #: (VerificationStepType), none by default: step_prepare_checkpoint's,
    #: step_verify_partition's, and step_create_stitched_ip's testbench.
    verify_steps: List[VerificationStepType] = field(default_factory=list)

    #: The .npy file of inputs PARTITION_PYTHON runs the partition and the source on,
    #: one frame per index of its first axis. STITCHED_IP's testbench streams its first
    #: frame when the file exists, a generated frame otherwise, and
    #: STITCHED_IP_TESTBENCH runs it.
    verify_input_npy: str = "input.npy"

    #: The AMD tool installation every tool step of the build runs in: the machine's
    #: (finn.toolchain.launcher.host_selection), with what this selection states laid
    #: over it, each field it states winning. How many runs Vivado launches at once
    #: when it builds the shell's bitfile is the selection's ``vivado_jobs``, a machine
    #: setting (``FINN_VIVADO_JOBS``) a build may state over.
    toolchain: Optional[Selection] = field(
        default=None, metadata=config(decoder=declared(Selection, "toolchain"))
    )

    #: Whether the target's shell's build inserts integrated logic analyzers on the
    #: ends' streams (the ``pynq`` shell's, finn.shells.pynq.runner.PynqOptions); a
    #: shell without a build (``ip``) refuses it. The bitfile's record states it
    #: (finn.outputs ``bitfile_options``).
    enable_hw_debug: bool = False

    #: If given, only run the steps in the list, each a step or phase name of the
    #: kernel path (finn.builder.build_dataflow_phases, finn.builder.build_dataflow_steps)
    #: or a function called with (model, DataflowBuildConfig); by default
    #: default_build_dataflow_steps.
    steps: Optional[List[Any]] = None

    #: If given, start from this step, or a phase's first step, loading the
    #: intermediate model saved before it (save_intermediate_models must be enabled).
    start_step: Optional[str] = None

    #: If given, stop after this step, or a phase's last step.
    stop_step: Optional[str] = None

    #: Whether each step's model is saved under intermediate_models.
    save_intermediate_models: bool = True

    #: Whether pdb postmortem debugging is launched when the build fails and stdin is
    #: a terminal.
    enable_build_pdb_debug: bool = True

    #: Whether the console shows everything the build log has; otherwise the runner's
    #: lines (each step it runs, the outcome), warnings, refusals and failures.
    verbose: bool = False

    #: Functions to run after named steps or phases.
    inject_steps_after: Dict[
        str, List[Callable[[ModelWrapper, "DataflowBuildConfig"], ModelWrapper]]
    ] = field(default_factory=dict)

    #: Functions to run before named steps or phases.
    inject_steps_before: Dict[
        str, List[Callable[[ModelWrapper, "DataflowBuildConfig"], ModelWrapper]]
    ] = field(default_factory=dict)

    #: Each step's own options, by step name: the options type the step declares
    #: beside it (finn.builder.build_dataflow_steps.build_dataflow_step_options), as an
    #: instance, or in JSON an object of its fields; a step not named here takes its
    #: type's defaults, and each step reads its own (``options``). A step that takes no
    #: options, or an option its type does not declare, is refused by name. In JSON,
    #: for TFC: {"step_prepare_io": {"preprocessing": "preproc.onnx", "input_datatype":
    #: "UINT8", "topk": 1}, "step_explore": {"kernel_exploration": [{"strategy":
    #: "target_throughput", "fps": 1000000}, {"strategy": "size_fifos"}]}}.
    step_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step_options = {
            step: _step_options(step, stated) for step, stated in self.step_options.items()
        }

    def options(self, step: str, kind: type[StepOptions]) -> StepOptions:
        """The options of the step named ``step``, of the type it declares (``kind``):
        as ``step_options`` states them, else that type's defaults."""
        stated = _step_options(step, self.step_options.get(step))
        if not isinstance(stated, kind):
            raise TypeError(
                f"{step} declares {type(stated).__name__} as its options, not {kind.__name__}"
            )
        return stated

    def _resolve_selection(self) -> Selection:
        """The selection this build runs its tools by: ``toolchain`` laid over the
        machine's (``host_selection``), each field it states winning; unset, the
        machine's."""
        return host_selection(stated=self.toolchain)

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

    def _resolve_shell_build_options(self) -> Optional[PynqOptions]:
        """The target's shell's build options (``enable_hw_debug``), as its build reads
        them; None for a shell without a build, which refuses ``enable_hw_debug``. Each
        refusal is a ValueError, named."""
        if type(self.enable_hw_debug) is not bool:
            raise ValueError(f"enable_hw_debug is a bool, not {self.enable_hw_debug!r}")
        target = self._resolve_target()
        if shell_row(target.shell, target.board).integration == VIVADO_BLOCK_DESIGN:
            return PynqOptions(enable_hw_debug=self.enable_hw_debug)
        if self.enable_hw_debug:
            raise ValueError(f"enable_hw_debug: the {target.shell!r} shell has no build to take it")
        return None


def _step_options(step: str, stated: Any) -> Any:
    """The options ``stated`` for the step named ``step`` as the type it declares
    (finn.builder.build_dataflow_steps.build_dataflow_step_options): an instance as it
    is, an object of its fields decoded, None as the type's defaults. A step that takes
    no options, a key its type does not declare (``UndefinedParameterError``) and a value
    its type refuses (``ValueError``) are refused, named."""
    # Imported here: the steps module imports this one.
    from finn.builder.build_dataflow_steps import (  # noqa: PLC0415
        build_dataflow_step_lookup,
        build_dataflow_step_options,
    )

    kind = build_dataflow_step_options.get(step)
    if kind is None:
        what = "takes no options" if step in build_dataflow_step_lookup else "is no step"
        raise UndefinedParameterError(
            f"step_options: {step} {what}; the steps that take options: "
            f"{', '.join(build_dataflow_step_options)}"
        )
    if stated is None:
        return kind()
    if isinstance(stated, kind):
        return stated
    if not isinstance(stated, Mapping):
        raise ValueError(
            f"step_options.{step}: an object of {kind.__name__}'s fields, not {stated!r}"
        )
    try:
        return declared(kind, f"step_options.{step}")(stated)
    except ValueError as refused:
        raise ValueError(f"step_options.{step}: {refused}") from refused
