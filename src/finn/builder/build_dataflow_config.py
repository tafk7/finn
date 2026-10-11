# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build configuration for a build of KernelOps (finn.custom_op.kernels) from a Brevitas
export, run by ``build_dataflow``.

Its fields are what every step may read: the target (``finn.platform.TargetRequest``),
the toolchain (``finn.toolchain.launcher.Selection``), the outputs, the checks and which
steps run. Each step's own options are in ``step_options``. It has an option only for
what the build actually makes.
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
    """A decoder for the nested dataclass ``cls`` at the configuration key ``name``. It
    refuses, by name, any key ``cls`` does not declare, which dataclasses_json would
    otherwise drop silently in a nested dataclass."""

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
    """What a build makes besides the partition and its reports. The model records what
    a later step uses (``finn.outputs``: the packaged IP, the bitfile); the build's
    manifest indexes the run (``report/build.json``). stitched_ip and ooc_synth work on
    any shell; the rest need a shell that integrates the partition (SHELL_OUTPUTS), so
    not ``ip``."""

    #: the packaged IP, with its interface description and an XSim testbench (written,
    #: not run: STITCHED_IP_TESTBENCH runs it)
    STITCHED_IP = "stitched_ip"
    #: the IP synthesized out of context, with its resources per member of the shell
    OOC_SYNTH = "ooc_synth"
    #: the bitfile of the target's shell
    BITFILE = "bitfile"
    #: the driver the shell's host runtime runs
    PYNQ_DRIVER = "pynq_driver"
    #: the bitfile and the driver, packaged for deployment
    DEPLOYMENT_PACKAGE = "deployment_package"


#: The outputs only a shell that integrates the partition makes.
SHELL_OUTPUTS = (
    DataflowOutputType.BITFILE,
    DataflowOutputType.PYNQ_DRIVER,
    DataflowOutputType.DEPLOYMENT_PACKAGE,
)

#: The outputs each output is made from, which must be asked for with it: the driver
#: needs the bitfile's delivered clock, and the deployment ships the bitfile and the
#: driver of the same build.
OUTPUT_NEEDS = {
    DataflowOutputType.PYNQ_DRIVER: (DataflowOutputType.BITFILE,),
    DataflowOutputType.DEPLOYMENT_PACKAGE: (
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
    ),
}


class VerificationStepType(str, Enum):
    """Optional checks a build can add. None run by default: each KernelOp's
    ``execute_node`` is the reference, and the test harness checks the hardware against
    it. A failed Python check is reported and the build continues, unless a value breaks
    its annotation, which stops it. A failed RTL check stops the build."""

    #: compare the prepared graph with the export on random inputs drawn from the
    #: input's annotated range, and check that the integer annotations hold
    #: (finn.harness.preparation)
    GRAPH_PREPARATION_PYTHON = "graph_preparation_python"

    #: execute the parent graph with the partition's KernelOps in Python on each input
    #: of verify_input_npy, and compare with the model before conversion
    PARTITION_PYTHON = "kernel_partition_python"
    #: compile and elaborate the partition's RTL in XSim (xvlog, xelab); needs Vivado
    PARTITION_ELABORATION = "kernel_partition_elaboration"
    #: run the packaged IP's XSim testbench (its run.sh) once in the build's toolchain,
    #: comparing the module's outputs with the partition's in Python; needs STITCHED_IP
    #: and Vivado
    STITCHED_IP_TESTBENCH = "stitched_ip_testbench"


#: The steps each verification needs to run: the one that saves its reference, and
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


#: The phases a build runs by default, in order (finn.builder.build_dataflow_phases).
default_build_dataflow_steps = [
    "phase_prepare_model",
    "phase_convert_to_hardware",
    "phase_create_accelerator",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


@dataclass
class DataflowBuildConfig(DataClassJsonMixin):
    """The configuration of a build. Pass it to build_dataflow_cfg, or write it as
    ``dataflow_build_config.json`` beside ``model.onnx`` for build_dataflow_directory and
    the ``build_dataflow`` command; there, relative paths name files in that directory
    (read_build_config). Serialized to and from JSON. Reading or constructing one
    refuses, by name (``UndefinedParameterError``), a key it does not declare, a
    ``step_options`` entry for a step that takes no options, and an option the step
    does not declare."""

    # The type is ignored: dataclasses_json declares this attribute as None, the value
    # its decorator sets.
    dataclass_json_config = config(undefined=Undefined.RAISE)["dataclasses_json"]  # type: ignore[assignment]

    #: Directory where the build's outputs and reports are written
    output_dir: str

    #: The target: a clock period, a board or a part (a board implies its part; a part
    #: given with it must match), and a shell, ``ip`` unless given. ``ip`` is the
    #: packaged IP for its user to integrate; ``pynq`` the Zynq block design for a
    #: board (finn.platform.shells). In JSON:
    #: {"period_ns": 5.0, "board": "Ultra96", "part": null, "shell": "pynq"}.
    target: TargetRequest = field(metadata=config(decoder=declared(TargetRequest, "target")))

    #: What the build makes besides the partition and its reports (DataflowOutputType).
    #: None by default.
    generate_outputs: List[DataflowOutputType] = field(default_factory=list)

    #: Optional checks to run (VerificationStepType). None by default.
    verify_steps: List[VerificationStepType] = field(default_factory=list)

    #: The .npy file of inputs for PARTITION_PYTHON, one frame per index of its first
    #: axis. STITCHED_IP's testbench streams its first frame if the file exists, else a
    #: generated frame.
    verify_input_npy: str = "input.npy"

    #: The AMD tool installation the build's tools run in: the machine's
    #: (finn.toolchain.launcher.host_selection), with any field given here taking
    #: precedence.
    toolchain: Optional[Selection] = field(
        default=None, metadata=config(decoder=declared(Selection, "toolchain"))
    )

    #: Whether the shell's build inserts integrated logic analyzers on the ends' streams
    #: (``pynq`` only, finn.shells.pynq.runner.PynqOptions; ``ip`` has no build and
    #: refuses it). The bitfile's record notes it (finn.outputs ``bitfile_options``).
    enable_hw_debug: bool = False

    #: If given, run only these steps, each the name of a step or phase
    #: (finn.builder.build_dataflow_phases, finn.builder.build_dataflow_steps) or a
    #: function taking (model, DataflowBuildConfig). By default,
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

    #: Whether the console shows everything in the build log; otherwise only the
    #: runner's lines (each step it runs, the outcome), warnings and errors.
    verbose: bool = False

    #: Functions to run after named steps or phases.
    inject_steps_after: Dict[
        str, List[Callable[[ModelWrapper, "DataflowBuildConfig"], ModelWrapper]]
    ] = field(default_factory=dict)

    #: Functions to run before named steps or phases.
    inject_steps_before: Dict[
        str, List[Callable[[ModelWrapper, "DataflowBuildConfig"], ModelWrapper]]
    ] = field(default_factory=dict)

    #: Each step's own options, by step name: an instance of the options type the step
    #: declares (finn.builder.build_dataflow_steps.build_dataflow_step_options), or in
    #: JSON an object of its fields. A step not named here uses its type's defaults.
    #: Naming a step that takes no options, or an option its type does not declare, is
    #: refused. In JSON, for TFC: {"step_prepare_io": {"preprocessing": "preproc.onnx",
    #: "input_datatype": "UINT8", "topk": 1}, "step_explore": {"kernel_exploration":
    #: [{"strategy": "target_throughput", "fps": 1000000}, {"strategy": "size_fifos"}]}}.
    step_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step_options = {
            step: _step_options(step, stated) for step, stated in self.step_options.items()
        }

    def options(self, step: str, kind: type[StepOptions]) -> StepOptions:
        """The options of the step ``step``, of its declared type ``kind``: from
        ``step_options``, else that type's defaults."""
        stated = _step_options(step, self.step_options.get(step))
        if not isinstance(stated, kind):
            raise TypeError(
                f"{step} declares {type(stated).__name__} as its options, not {kind.__name__}"
            )
        return stated

    def _resolve_selection(self) -> Selection:
        """The selection this build's tools run by: ``toolchain`` laid over the
        machine's (``host_selection``)."""
        return host_selection(stated=self.toolchain)

    def _resolve_toolchain(self) -> Toolchain:
        """The prepared toolchain every tool step of this build runs in. The first step
        that asks prepares it, and later steps get the same object. It is kept on the
        instance, not as a field, because it is prepared, not configured, and is not
        serialized."""
        toolchain: Optional[Toolchain] = getattr(self, "_toolchain", None)
        if toolchain is None:
            toolchain = self._resolve_selection().prepare()
            self._toolchain = toolchain
        return toolchain

    def _resolve_target(self) -> Target:
        """The build's resolved target (finn.platform.resolve_target)."""
        return self.target.resolve()

    def _resolve_shell_build_options(self) -> Optional[PynqOptions]:
        """The options of the shell's build (``enable_hw_debug``), or None for a shell
        without a build. Raises ValueError if ``enable_hw_debug`` is not a bool, or is
        set for a shell without a build."""
        if type(self.enable_hw_debug) is not bool:
            raise ValueError(f"enable_hw_debug is a bool, not {self.enable_hw_debug!r}")
        target = self._resolve_target()
        if shell_row(target.shell, target.board).integration == VIVADO_BLOCK_DESIGN:
            return PynqOptions(enable_hw_debug=self.enable_hw_debug)
        if self.enable_hw_debug:
            raise ValueError(f"enable_hw_debug: the {target.shell!r} shell has no build to take it")
        return None


def _step_options(step: str, stated: Any) -> Any:
    """``stated`` as an instance of the options type the step ``step`` declares
    (finn.builder.build_dataflow_steps.build_dataflow_step_options): an instance is
    kept, an object of its fields is decoded, and None gives the defaults. Raises
    ``UndefinedParameterError`` for a step without options or an undeclared key, and
    ``ValueError`` for a value the type refuses."""
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
