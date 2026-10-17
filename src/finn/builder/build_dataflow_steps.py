# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's steps, and at the end its phases (``build_dataflow_phases``) and premade
lists (``build_dataflow_lists``).

A step is a function ``(model, cfg, *, settings...) -> model``; its keyword-only
parameters are its settings. A phase names a list of steps, and so does a premade list,
from the export to an output; a build without ``steps`` runs the default list
(``default_steps``). A step refuses a setting it cannot take, or a model it cannot work
on, when it runs, saying what to change: a setting's refusal is a
``ConfigurationRefused`` naming the setting.

The steps prepare a Brevitas export into a streamlined graph; convert its nodes to
KernelOps and cut them into one partition; choose the partition's parameters, one
strategy step at a time; and make the outputs. Each output is a step, and a list makes
the outputs whose steps it names: the packaged IP, the bitfile, the driver and the
deployment package. A step whose input an earlier step makes refuses when that input is
missing, naming the step to list before it. Each step only calls the transformations
that do its work.
"""

import logging
from pathlib import Path

from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup_model

import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    StepFunction,
    config_finding,
)
from finn.builder.build_dataflow_resources import (
    RESOURCES_FILE,
    built_reports,
    shell_resources_report,
)
from finn.builder.build_dataflow_testbench import TESTBENCH_DIR, generated_frame, write_testbench
from finn.core.graph.annotations.choices import OBJECTIVE_FPS
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import partition_body
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.reporting.manifest import note, reported, write_report
from finn.custom_op.kernels.network import stale_text
from finn.custom_op.kernels.shell import verify_partition
from finn.kernels.explore import (
    FIFOS_BY_MODEL,
    DebugPlaceholder,
    Explorer,
    MaxThroughput,
    Rebaseline,
    SizeFifos,
    TargetThroughput,
)
from finn.platform.resolve import refuse_drift
from finn.platform.shells import VIVADO_BLOCK_DESIGN, shell_row
from finn.shells.pynq.deploy import MakePynqDeployment
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT, MakePynqDriver
from finn.shells.pynq.runner import REPORTS, BuildPynqBitfile
from finn.transformation.kernels import (
    Explored,
    ToKernelOps,
    exploration_summary,
    explore_partition,
    kernel_choices_config,
    kernel_ops_report,
    kernel_ops_summary,
    refuse_host_between,
)
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import PackagePartition, partition_utilization_report
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.transformation.streamline import Streamline

#: The steps' logger.
_LOG = logging.getLogger(__name__)

#: Where step_package_partition packages the partition, under the output directory.
STITCHED_IP_DIR = "stitched_ip"


def _cleaned(model: ModelWrapper, override_inpsize: int | list[int] | None = None) -> ModelWrapper:
    """``model`` through qonnx's cleanup (``cleanup_model``), its Quant nodes kept."""
    inpsize = tuple(override_inpsize) if isinstance(override_inpsize, list) else override_inpsize
    cleaned: ModelWrapper = cleanup_model(model, override_inpsize=inpsize)  # type: ignore[no-untyped-call]
    return cleaned


def step_cleanup(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    override_inpsize: int | list[int] | None = None,
) -> ModelWrapper:
    """Clean the export with qonnx's cleanup, its Quant nodes kept.

    ``override_inpsize``: the batch size (an int) or the whole input shape (a list) to
    set; None keeps the export's."""
    return _cleaned(model, override_inpsize)


def step_add_pre_post_processing(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    preprocessing: Path | None = None,
    input_datatype: str | None = None,
    topk: int | None = None,
) -> ModelWrapper:
    """State the network's inputs and outputs, on the graph as exported; by default,
    the export's own.

    - ``preprocessing``: an ONNX model (a QONNX export, such as one of
      ``finn.util.pytorch.ToTensor``) merged ahead of the network; its input becomes the
      graph's input.
    - ``input_datatype``: the qonnx name of the graph input's datatype, the range of
      values the network is prepared for. None leaves the input unannotated (FLOAT32).
    - ``topk``: append a TopK of the ``topk`` largest outputs; its indices become the
      graph's output. A graph that already ends in a TopK is left alone."""
    if input_datatype is not None:
        try:
            DataType[input_datatype]
        except KeyError:
            message = f"{input_datatype!r} is no qonnx datatype; name one, such as UINT8 or INT4"
            place = "configuration.input_datatype"
            raise ConfigurationRefused(
                [config_finding("setting-invalid", message, place)]
            ) from None
    if topk is not None and topk < 1:
        message = f"{topk}; give at least 1, or no topk for the network's own outputs"
        raise ConfigurationRefused(
            [config_finding("setting-invalid", message, "configuration.topk")]
        )
    if preprocessing is not None:
        model = model.transform(
            MergeONNXModels(_cleaned(ModelWrapper(str(preprocessing))))  # type: ignore[no-untyped-call]
        )
        # MergeONNXModels imports the standard domain only; the custom ops' domains are
        # imported again, so their nodes read without a fallback-version warning.
        imported_domains = {opset.domain for opset in model.model.opset_import}
        for domain in sorted({node.domain for node in model.graph.node} - imported_domains):
            model.model.opset_import.append(helper.make_opsetid(domain, 1))
    if input_datatype is not None:
        model.set_tensor_datatype(model.get_first_global_in(), DataType[input_datatype])
    if topk is not None:
        model = model.transform(InsertTopK(k=topk))  # type: ignore[no-untyped-call]
        # InsertTopK drops the former output's shape.
        model = model.transform(InferShapes())
    return model


def step_qonnx_to_finn(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, max_multithreshold_bit_width: int = 8
) -> ModelWrapper:
    """Lower the graph's Quant nodes: activations to MultiThresholds, weights folded to
    integers.

    ``max_multithreshold_bit_width``: the widest Quant activation lowered to a
    MultiThreshold; a wider one stays a Quant."""
    return model.transform(
        ConvertQONNXtoFINN(  # type: ignore[no-untyped-call]
            filter_function=default_filter_function_generator(  # type: ignore[no-untyped-call]
                max_multithreshold_bit_width=max_multithreshold_bit_width
            )
        )
    )


def step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Tidy the graph: shapes inferred, constants folded, nodes and tensors named,
    datatypes inferred and static inputs removed."""
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())  # type: ignore[no-untyped-call]
    model = model.transform(GiveUniqueNodeNames())  # type: ignore[no-untyped-call]
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())  # type: ignore[no-untyped-call]
    return model.transform(RemoveStaticGraphInputs())


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Streamline the graph: scales and biases absorbed into thresholds, convolutions
    lowered to Im2Col and MatMul, bipolar MatMuls rewritten to XnorPopcountMatMul, and
    the scalar scale and bias ahead of a label select absorbed into its TopK."""
    model = model.transform(absorb.AbsorbScalarBiasIntoMultiThreshold())  # type: ignore[no-untyped-call]
    model = model.transform(reorder.MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    if model.get_nodes_by_op_type("Conv"):
        model = model.transform(LowerConvsToMatMul())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    return model.transform(absorb.AbsorbScalarMulAddIntoTopK())


def step_convert_to_channels_last(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Make MaxPool NHWC and absorb the transposes around it, then tidy the graph again
    (the transforms since the last tidy-up name what they add at random), infer the data
    layouts and remove the tensors no node reads."""
    model = model.transform(reorder.MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(reorder.MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = step_tidy_up(model, cfg)
    model = model.transform(InferDataLayouts())
    return model.transform(RemoveUnusedTensors())


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Record the build's target in the model, and convert the nodes the KernelOps
    cover (MatMul, MultiThreshold, and Im2Col with the MatMul that reads its patches)
    into KernelOps. Writes report/kernel_ops.json: what each node became, or why it
    stayed on the host. Refuses a host node between KernelOps: they form one
    partition."""
    conversion = ToKernelOps(cfg.resolved_target)
    model = model.transform(conversion)
    report = kernel_ops_report(model, conversion.outcomes)
    write_report(Path(cfg.output_dir) / "report" / "kernel_ops.json", report)
    for line in kernel_ops_summary(conversion.outcomes):
        _LOG.info(line)
    refuse_host_between(model, conversion.outcomes)
    return model


def _explore(
    model: ModelWrapper, cfg: DataflowBuildConfig, step: str, strategy: Explorer
) -> Explored:
    """Explore the partition by ``strategy``, the graph's holds kept, and save it; write the
    step's report, report/<step>.json, log its summary and note each limitation found as
    a warning."""
    explored = explore_partition(model, strategy)
    write_report(Path(cfg.output_dir) / "report" / f"{step}.json", explored.report)
    for line in exploration_summary(explored, step):
        _LOG.info(line)
    for finding in explored.findings:
        note(finding)
    return explored


def step_target_throughput(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    target_fps: float,
    relax_target_fps: bool = True,
) -> ModelWrapper:
    """Fold the partition's kernels to the least parallelism that meets ``target_fps``
    frames a second at the target's clock, the holds kept. With ``relax_target_fps``, a
    throughput the folding cannot reach is relaxed to the fastest it reaches. Writes
    report/step_target_throughput.json. The partition's body records ``target_fps``, the
    objective step_build_bitfile reports the delivered clock against, until a later step
    that may change the folding removes it."""
    strategy = TargetThroughput(target_fps, relax=relax_target_fps)
    _explore(model, cfg, "step_target_throughput", strategy)
    return model


def step_max_throughput(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, max_throughput_within: dict[str, float]
) -> ModelWrapper:
    """Fold the partition's kernels to the most throughput whose shell's resources stay
    within ``max_throughput_within``, a fraction of the part's for each resource named
    (``{"lut": 0.5}``), the holds kept. A budget even the least parallelism exceeds is a
    limitation, and that point is kept. Writes report/step_max_throughput.json."""
    explored = _explore(model, cfg, "step_max_throughput", MaxThroughput(max_throughput_within))
    each = explored.report["strategy"]
    _LOG.info(
        f"Kernel choices: max_throughput reached {each['bottleneck']['cycles']} cycles a "
        f"frame within {each['budget']} ({'fits' if each['fits'] else 'does not fit'}; "
        f"binding: {each['binding']}; {len(each['tried'])} budgets tried)"
    )
    return model


def step_size_fifos(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    fifo_margin: int = 0,
    fifo_ram_style: str | None = None,
) -> ModelWrapper:
    """Size each channel of the partition from the beat patterns at both its ends, at the
    current folding: a direct connection, or a FIFO of the least depth that keeps its
    producer within its idle time, plus ``fifo_margin`` words. A held channel is kept.
    ``fifo_ram_style`` commits a memory style with each FIFO's depth; None leaves it to
    FinnLib (``auto``). Writes report/step_size_fifos.json."""
    strategy = SizeFifos(margin=fifo_margin, ram_style=fifo_ram_style)
    _explore(model, cfg, "step_size_fifos", strategy)
    return model


def step_baseline_choices(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Set every choice of the partition that is not held back to its baseline, so a
    resumed build starts over. Writes report/step_baseline_choices.json."""
    _explore(model, cfg, "step_baseline_choices", Rebaseline())
    return model


def step_placeholder_choices(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """For debugging: set every open required choice of the partition to its first
    allowed value, so a design can be generated where no strategy makes that choice. A
    FIFO's depth has no first value and is refused (run step_size_fifos). Writes
    report/step_placeholder_choices.json."""
    _explore(model, cfg, "step_placeholder_choices", DebugPlaceholder())
    return model


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Cut the KernelOps into one StreamingDataflowPartition, leaving the nodes before
    and after it on the host; the build's model is the parent graph from here on. The
    partition is named ``partition``: its node, its body (partition/partition.onnx), its
    IP and its block-design instance. Each channel's choices are created here, and each
    KernelOp's checked again in context: one made stale by an edit since conversion is
    reset to its baseline, noted as a limitation (``choice-stale``, owned by the choice, the
    kernel's findings its causes)."""
    cut = CutKernelPartition(Path(cfg.output_dir) / "partition")
    model = model.transform(cut)
    for key, found in cut.dropped.items():
        message = f"dropped as stale at the cut and re-baselined: {stale_text(key, found)}"
        note(Finding(FindingKind.LIMITATION, "choice-stale", key, message, causes=found))
    return model


def step_verify_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write the partition's choices to report/kernel_choices.json by node or tensor name,
    a log FINN never reads, then verify the partition before a shell builds it
    (``verify_partition``): its choices are complete and current, its ports match its
    graph's inputs and outputs, its target is the configuration's, and its FIFOs keep their
    streams at its bottleneck period (the build's one check of them, by sizing's model). A
    FIFO not held that does not is refused (``fifo-stale``: run step_size_fifos); a held one
    (``fifo-held-stale``) and a ``direct`` that throttles its producer
    (``direct-throttles``) are noted as warnings. Writes report/step_verify_partition.json,
    what the check found. The body records the design it verified: packaging, the shell's
    integration and the testbench refuse one not verified, or changed since."""
    node, body, body_file = partition_body(model)
    write_report(
        Path(cfg.output_dir) / "report" / "kernel_choices.json", kernel_choices_config(body)
    )
    _, check = verify_partition(body, node.name)
    stated = {"fifos_judged": FIFOS_BY_MODEL, "fifo_check": check.report()}
    write_report(Path(cfg.output_dir) / "report" / "step_verify_partition.json", stated)
    for finding in check.findings:
        note(finding)
    refuse_drift(read_target(body), cfg.resolved_target, "build")
    body.save(body_file)
    return model


def _refuse_drift(model: ModelWrapper, cfg: DataflowBuildConfig, output: str) -> None:
    """Refuse to make ``output`` if the partition's target is not the configuration's."""
    _, body, _ = partition_body(model)
    refuse_drift(read_target(body), cfg.resolved_target, f"{output} build")


def _refuse_shell(cfg: DataflowBuildConfig, step: str) -> None:
    """Refuse ``step`` where the target's shell has no bitfile build: the bitfile, the
    driver and the deployment are that build's."""
    target = cfg.resolved_target
    if shell_row(target.shell, target.board).integration == VIVADO_BLOCK_DESIGN:
        return
    message = (
        f"the {target.shell!r} shell has no bitfile build, which {step} needs; state a "
        f"shell that has one (pynq, for a board), or leave {step} out of the steps"
    )
    raise ConfigurationRefused(
        [config_finding("shell-without-bitfile", message, "configuration.target.shell")]
    )


def step_package_partition(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, ooc_synth: bool = False
) -> ModelWrapper:
    """Package the partition as an IP named ``partition`` into ``stitched_ip/``, with its
    interface description (``interface.json``) and an XSim testbench in
    ``stitched_ip/testbench``, which streams a frame of the partition's inputs drawn from
    their datatypes. The partition's body records the IP, so a shell's integration
    reuses it.

    ``ooc_synth``: synthesize the IP out of context; its utilization report is listed
    under this step, and step_resources_report reads it. An IP the store holds already
    is reused, not packaged again, and its line says so."""
    directory = Path(cfg.output_dir) / STITCHED_IP_DIR
    toolchain = cfg.step_toolchain("step_package_partition")
    node, body, body_file = partition_body(model)
    packaging = PackagePartition(
        node.name,
        run_synth=ooc_synth,
        directory=directory,
        toolchain=toolchain,
    )
    body = body.transform(packaging)
    body.save(body_file)
    assert packaging.packaged is not None
    if packaging.packaged.built:
        _LOG.info(f"Packaged IP {node.name} and its interface description in {directory}")
    else:
        _LOG.info(
            f"Reused the stored IP {node.name} ({packaging.packaged.directory}), placed with "
            f"its interface description in {directory}"
        )
    if ooc_synth:
        reported(partition_utilization_report(directory))
    testbench = directory / TESTBENCH_DIR
    write_testbench(body, testbench, generated_frame(body), toolchain=toolchain, label=node.name)
    _LOG.info(f"XSim testbench on a generated frame written into {testbench} (run.sh runs it)")
    return model


def step_build_bitfile(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    hw_debug: bool = False,
) -> ModelWrapper:
    """Build the target's shell around the partition to a bitfile in the output
    directory; only the pynq shell has a bitfile build, and on another this step is
    refused. The shell's build packages the partition itself unless the IP
    step_package_partition packaged has the key the partition has now (its sources and
    package.tcl, Vivado and run_synth; package.ip_key). Refuses a partition built for
    another target than the configuration's. The model records the bitfile and its
    hardware handoff, what it was built from (the design and the IP) and with which
    options; the build's reports are collected under report/. A shortfall of the
    delivered clock is noted as a warning, against the throughput step_target_throughput
    folded to, where the partition's body records one.

    ``hw_debug``: insert integrated logic analyzers on the ends' streams."""
    _refuse_shell(cfg, "step_build_bitfile")
    _refuse_drift(model, cfg, "bitfile")
    _, body, _ = partition_body(model)
    building = BuildPynqBitfile(
        Path(cfg.output_dir),
        toolchain=cfg.step_toolchain("step_build_bitfile"),
        hw_debug=hw_debug,
        objective_fps=body.get(OBJECTIVE_FPS),
    )
    model = model.transform(building)
    for report in REPORTS.values():
        reported(Path(cfg.output_dir) / report)
    for finding in building.findings:
        note(finding)
    _LOG.info(f"Bitfile written into {building.bitfile.parent}")
    return model


def step_make_driver(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write the PYNQ driver into driver/ and its description into report/driver.json.
    The driver sets PL0 to the clock the bitfile delivers, and loads the build's bitfile
    by default: step_build_bitfile comes before it. Refuses a shell without a bitfile
    build, a missing bitfile, one built from another design than the current partition or
    from another IP than its body states, and a partition built for another target than
    the configuration's."""
    _refuse_shell(cfg, "step_make_driver")
    _refuse_drift(model, cfg, "pynq_driver")
    making = MakePynqDriver(Path(cfg.output_dir))
    model = model.transform(making)
    reported(Path(cfg.output_dir) / DRIVER_REPORT)
    description = making.description
    returned = ", ".join(
        f"{each['name']} ({each['element']}, {each['shape']})" for each in description["returns"]
    )
    driver_dir = Path(cfg.output_dir) / DRIVER_DIR
    _LOG.info(
        f"PYNQ Python driver written into {driver_dir}, at {description['fclk_mhz']} MHz; "
        f"it returns {returned}"
    )
    return model


def step_deployment_package(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Package the bitfile, the driver, the host-side model and the driver's description
    into deploy/, written afresh. Ships what this output directory's own steps made
    (step_build_bitfile and step_make_driver come before it): refuses a shell without a
    bitfile build, a missing or foreign bitfile or driver, one made from another design
    than the current partition or for another IP than its body states, and a partition
    built for another target than the configuration's."""
    _refuse_shell(cfg, "step_deployment_package")
    _refuse_drift(model, cfg, "deployment_package")
    return model.transform(MakePynqDeployment(Path(cfg.output_dir)))


def step_resources_report(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write report/resources.json: the resources of each member of the shell and their
    total, as modelled, as synthesized out of context and as placed, from the reports
    the steps before it made. A column the build did not make is marked absent."""
    toolchain = cfg.step_toolchain("step_resources_report")
    stated = shell_resources_report(model, built_reports(model, toolchain=toolchain))
    report = Path(cfg.output_dir) / RESOURCES_FILE
    write_report(report, stated)
    absent = ", ".join(f"{column} ({why})" for column, why in stated["absent"].items())
    _LOG.info(
        f"Resources per member written to {report}" + (f"; absent: {absent}" if absent else "")
    )
    return model


#: The build's phases, in order, each its steps in order. In a configuration's ``steps`` a
#: phase stands for its steps, as ``start_step`` for its first, as ``stop_step`` its last.
#: The last two are the outputs' steps: listing either makes each of its outputs.
build_dataflow_phases: dict[str, list[StepFunction]] = {
    "phase_prepare_model": [
        step_cleanup,
        step_add_pre_post_processing,
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_channels_last,
    ],
    "phase_convert_to_hardware": [step_convert_to_hw],
    "phase_create_accelerator": [step_create_dataflow_partition],
    # By default only the FIFOs are sized; a list adds the other strategy steps by name.
    "phase_optimize_hardware": [step_size_fifos, step_verify_partition],
    "phase_emit_hardware": [step_package_partition],
    "phase_generate_outputs": [
        step_build_bitfile,
        step_make_driver,
        step_resources_report,
        step_deployment_package,
    ],
}

#: The steps that choose the partition's parameters, one strategy each.
KERNEL_STRATEGY_STEPS: list[StepFunction] = [
    step_target_throughput,
    step_max_throughput,
    step_size_fifos,
    step_baseline_choices,
    step_placeholder_choices,
]

#: The build's steps by name.
build_dataflow_step_lookup: dict[str, StepFunction] = {
    step.__name__: step
    for step in [
        *(step for steps in build_dataflow_phases.values() for step in steps),
        *KERNEL_STRATEGY_STEPS,
    ]
}


#: The steps every premade list starts with: the partition from the export, its FIFOs
#: sized at the baseline folding, and checked.
_PARTITION: list[StepFunction] = [
    step
    for phase in (
        "phase_prepare_model",
        "phase_convert_to_hardware",
        "phase_create_accelerator",
        "phase_optimize_hardware",
    )
    for step in build_dataflow_phases[phase]
]

#: The premade lists, each from the export to the output it names, then the resources
#: report. In a configuration's ``steps`` a list stands for its steps, as a phase does.
build_dataflow_lists: dict[str, list[StepFunction]] = {
    # The partition, its choices (report/kernel_choices.json) and its modelled resources.
    "list_partition": [*_PARTITION, step_resources_report],
    # The packaged IP, its interface description and its testbench, on any shell.
    "list_ip": [*_PARTITION, step_package_partition, step_resources_report],
    # The bitfile of a shell with a bitfile build (pynq).
    "list_bitfile": [*_PARTITION, step_build_bitfile, step_resources_report],
    # The bitfile, its driver and the deployment package.
    "list_deployment": [
        *_PARTITION,
        step_build_bitfile,
        step_make_driver,
        step_resources_report,
        step_deployment_package,
    ],
}

#: The premade list a build runs without ``steps``.
DEFAULT_LIST = "list_partition"


def default_steps() -> list[str]:
    """The list a build runs without ``steps`` (``DEFAULT_LIST``), by step name."""
    return [step.__name__ for step in build_dataflow_lists[DEFAULT_LIST]]
