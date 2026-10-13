# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's steps, and at the end its phases (``build_dataflow_phases``).

A step is a function ``(model, cfg, *, settings...) -> model``. Its keyword-only
parameters are its settings: the build fills each from the configuration's setting of
that name, else it keeps its default (finn.builder.build_dataflow_plan). A phase is a
name for a list of steps; a build without ``steps`` runs every phase's steps in order.

The steps prepare a Brevitas export into the streamlined graph the kernel path converts;
convert its nodes to KernelOps (finn.custom_op.kernels) and cut them into one partition;
choose the partition's parameters, one strategy step at a time; and make the outputs the
configuration asks for: the partition's packaged IP on any shell, and the bitfile,
driver and deployment package on a shell that integrates it. The verifying steps
(VERIFYING_STEPS) are in no phase: a build asks for a check by listing its step.

A step is the transformations and calls it runs; what they do lives in the module that
owns it: the exploration in finn.transformation.kernels, the shell's outputs in
finn.shells, the checks in finn.builder.build_dataflow_verification,
finn.builder.build_dataflow_testbench and finn.transformation.kernels.package, the
resources report in finn.builder.build_dataflow_resources, and the logging of the
reports and verdicts the build's manifest collects in
finn.builder.build_dataflow_manifest."""

import json
import logging
from collections.abc import Callable
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
    DataflowBuildConfig,
    DataflowOutputType,
    StepFunction,
)
from finn.builder.build_dataflow_manifest import reported, verdict, write_report
from finn.builder.build_dataflow_resources import (
    RESOURCES_FILE,
    built_reports,
    shell_resources_report,
)
from finn.builder.build_dataflow_testbench import (
    TESTBENCH_DIR,
    run_testbench,
    testbench_frame,
    write_testbench,
)
from finn.builder.build_dataflow_verification import VERIFICATION_DIR, verify_partition_python
from finn.core.graph.annotations.choices import OBJECTIVE_FPS
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import partition_body
from finn.core.rtlsim_bench import SimulationFailed
from finn.custom_op.kernels.shell import configured_root
from finn.kernels.explore import (
    DebugPlaceholder,
    Explorer,
    MaxThroughput,
    Pinned,
    Rebaseline,
    SizeFifos,
    TargetThroughput,
)
from finn.platform.resolve import refuse_drift
from finn.shells.pynq.deploy import MakePynqDeployment
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT, MakePynqDriver
from finn.shells.pynq.runner import REPORTS, BuildPynqBitfile
from finn.toolchain.launcher import ToolFailed
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
from finn.transformation.kernels.package import (
    ElaboratePartition,
    PackagePartition,
    partition_utilization_report,
)
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.transformation.streamline import Streamline

#: The steps' logger. The build log has every line; the console shows warnings, such as
#: a failed verification the build continued past or a limitation, and with ``verbose``
#: every line.
_LOG = logging.getLogger(__name__)

#: Where step_package_partition packages the partition, under the output directory: its
#: IP, interface description and testbench, and with OOC_SYNTH its utilization report.
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
    """Clean the export with qonnx's cleanup: shapes inferred, constants folded with the
    Quant nodes kept, parameters made unique, unused tensors and static inputs removed,
    nodes and tensors named.

    ``override_inpsize``: the batch size (an int) or the whole input shape (a list) the
    cleanup sets; None keeps the export's."""
    return _cleaned(model, override_inpsize)


def step_add_pre_post_processing(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    preprocessing: Path | None = None,
    input_datatype: str | None = None,
    topk: int | None = None,
) -> ModelWrapper:
    """State the network's inputs and outputs, on the graph as exported (its Quant nodes
    not yet lowered); by default, the export's own.

    - ``preprocessing``: an ONNX model (a QONNX export, such as one of
      ``finn.util.pytorch.ToTensor``), cleaned and merged ahead of the network; its input
      becomes the graph's input. A graph that already holds its preprocessing is given
      none: merging composes it again.
    - ``input_datatype``: the qonnx name of the graph input's datatype, the range of
      values the network is prepared for. None leaves the input unannotated, which is
      read as FLOAT32.
    - ``topk``: append a TopK that selects the ``topk`` largest outputs; its indices
      become the graph's output. A graph that already ends in a TopK is left alone."""
    if input_datatype is not None:
        try:
            DataType[input_datatype]
        except KeyError:
            raise ValueError(f"input_datatype: {input_datatype!r} is no qonnx datatype") from None
    if topk is not None and topk < 1:
        raise ValueError(f"topk: {topk}; at least 1")
    if preprocessing is not None:
        model = model.transform(
            MergeONNXModels(_cleaned(ModelWrapper(str(preprocessing))))  # type: ignore[no-untyped-call]
        )
        # MergeONNXModels imports the standard domain only: the custom ops' domains are
        # stated again, so that reading their nodes warns of no fallback version.
        imported_domains = {opset.domain for opset in model.model.opset_import}
        for domain in sorted({node.domain for node in model.graph.node} - imported_domains):
            model.model.opset_import.append(helper.make_opsetid(domain, 1))
    if input_datatype is not None:
        model.set_tensor_datatype(model.get_first_global_in(), DataType[input_datatype])
    if topk is not None:
        model = model.transform(InsertTopK(k=topk))  # type: ignore[no-untyped-call]
        # InsertTopK drops the former output's shape with its graph output entry.
        model = model.transform(InferShapes())
    return model


def step_qonnx_to_finn(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, max_multithreshold_bit_width: int = 8
) -> ModelWrapper:
    """Lower the whole graph's Quant nodes, the preprocessing's too: activations to
    MultiThresholds, weights folded to integers (ConvertQONNXtoFINN).

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
    """Tidy the graph: shapes inferred, constants folded, nodes and tensors named by
    their place in the graph, every node's outputs annotated by qonnx's rule, and static
    inputs removed."""
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())  # type: ignore[no-untyped-call]
    model = model.transform(GiveUniqueNodeNames())  # type: ignore[no-untyped-call]
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())  # type: ignore[no-untyped-call]
    return model.transform(RemoveStaticGraphInputs())


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Streamline the graph: scales and biases absorbed into thresholds, convolutions
    lowered to Im2Col and MatMul if the graph has any, bipolar MatMuls rewritten to
    XnorPopcountMatMul (which reads the lowered convolutions' MatMuls too), and the
    scalar scale and bias ahead of a label select absorbed into its TopK."""
    model = model.transform(absorb.AbsorbScalarBiasIntoMultiThreshold())  # type: ignore[no-untyped-call]
    model = model.transform(reorder.MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    if model.get_nodes_by_op_type("Conv"):
        model = model.transform(LowerConvsToMatMul())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    return model.transform(absorb.AbsorbScalarMulAddIntoTopK())


def step_convert_to_channels_last(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Make MaxPool NHWC and absorb and collapse the transposes around it, then tidy the
    graph as step_tidy_up does (the transforms since the last tidy-up name what they add
    at random), infer every tensor's data layout and remove the tensors no node reads."""
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
    stayed on the host. Refuses a host node between KernelOps, since they must form one
    partition."""
    conversion = ToKernelOps(cfg.resolved_target)
    model = model.transform(conversion)
    report = kernel_ops_report(model, conversion.outcomes)
    write_report(Path(cfg.output_dir) / "report" / "kernel_ops.json", report)
    for line in kernel_ops_summary(conversion.outcomes):
        _LOG.info(line)
    refuse_host_between(model, conversion.outcomes)
    return model


#: Where step_verify_partition writes the partition's choices, by graph name
#: (kernel_choices_config), under the output directory: the file step_pin_choices pins.
KERNEL_CHOICES_FILE = "kernel_choices.json"


def _explore(
    model: ModelWrapper, cfg: DataflowBuildConfig, step: str, strategy: Explorer
) -> Explored:
    """Explore the partition by ``strategy``, its pins held and every value saved in its
    body (finn.transformation.kernels.explore_partition); write the step's report,
    report/<step>.json, and log its summary, each limitation found a warning, never an
    error."""
    explored = explore_partition(model, strategy)
    write_report(Path(cfg.output_dir) / "report" / f"{step}.json", explored.report)
    for level, line in exploration_summary(explored, step):
        _LOG.log(level, line)
    return explored


def step_pin_choices(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, kernel_choices_file: Path
) -> ModelWrapper:
    """Pin the partition's choices that ``kernel_choices_file`` states, by node or tensor
    name, in the form step_verify_partition writes (kernel_choices.json). The pins are
    set in one batch and recorded in the partition's body, so every later strategy step
    of the build holds them. A choice the partition does not have, a value it does not
    take, or one an earlier pin holds otherwise is refused, by name. Pin first, so that
    the strategies after it fold around the pins. Writes report/step_pin_choices.json."""
    _explore(model, cfg, "step_pin_choices", Pinned(kernel_choices_file))
    return model


def step_target_throughput(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    target_fps: float,
    relax_target_fps: bool = True,
) -> ModelWrapper:
    """Fold the partition's kernels to the least parallelism that meets ``target_fps``
    frames a second at the target's clock, the pins held. With ``relax_target_fps``, a
    throughput the folding cannot reach is relaxed to the fastest it reaches, and the
    kernels folded to that. Writes report/step_target_throughput.json, with the
    bottleneck reached and whether it meets the target. The partition's body records
    ``target_fps`` (finn.core.graph.annotations.choices.OBJECTIVE_FPS), the objective
    step_build_bitfile reports the delivered clock against, until a later step that may
    change the folding (FOLDING_STEPS) removes it."""
    strategy = TargetThroughput(target_fps, relax=relax_target_fps)
    _explore(model, cfg, "step_target_throughput", strategy)
    return model


def step_max_throughput(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, max_throughput_within: dict[str, float]
) -> ModelWrapper:
    """Fold the partition's kernels to the most throughput whose shell's resources stay
    within ``max_throughput_within``, a fraction of the part's for each resource named
    (``{"lut": 0.5}``), the pins held. A budget even the least parallelism exceeds is a
    limitation, logged, and that point is kept. Writes
    report/step_max_throughput.json, with every budget tried."""
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
    folding the steps before it chose: a direct connection, or a FIFO of the least depth
    that keeps its producer within its idle time, plus ``fifo_margin`` words. A pinned
    channel is kept. ``fifo_ram_style`` commits a memory style with each FIFO's depth;
    None leaves it to FinnLib's own selection (``auto``). Run it after the last step
    that changes the folding; the build's plan refuses a folding step after it. Writes
    report/step_size_fifos.json."""
    strategy = SizeFifos(margin=fifo_margin, ram_style=fifo_ram_style)
    _explore(model, cfg, "step_size_fifos", strategy)
    return model


def step_baseline_choices(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Set every choice of the partition that is not pinned back to its baseline: a
    build resumed from a partition explored before starts over with it. Writes
    report/step_baseline_choices.json."""
    _explore(model, cfg, "step_baseline_choices", Rebaseline())
    return model


def step_placeholder_choices(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """For debugging: set every open required choice of the partition to its first
    allowed value, so that a design can be generated while no strategy makes that choice
    yet; each is logged ``DEBUG: set by placeholder``. A FIFO's depth has no first value:
    it is refused, by name (run step_size_fifos). Writes
    report/step_placeholder_choices.json."""
    _explore(model, cfg, "step_placeholder_choices", DebugPlaceholder())
    return model


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Cut the KernelOps into one StreamingDataflowPartition, leaving the nodes before
    and after it on the host. From here on the build's model is the parent graph. The
    partition is named ``partition``: its node, its body (partition/partition.onnx), its
    IP and its block-design instance; partition/ also holds the scratch models the
    shell's integration generates the ends' IPs from (partition/ends/). Its body carries
    the target and every choice at its baseline. Each channel's choices are created
    here, and each KernelOp's are checked again in context: one that is stale because
    the model was edited since conversion is reset to its baseline, with a warning
    saying why."""
    cut = CutKernelPartition(Path(cfg.output_dir) / "partition")
    model = model.transform(cut)
    for key, why in cut.dropped.items():
        _LOG.warning(f"Kernel choices: {key} dropped as stale at the cut and re-baselined: {why}")
    return model


def step_verify_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write the partition's choices, as the strategy steps left them, to
    kernel_choices.json, by node or tensor name (kernel_choices_config; the file
    step_pin_choices pins). Then check the partition before a shell builds it: its
    choices are complete and current, its ports match its graph's inputs and outputs in
    order (configured_root), and its target is the configuration's (refuse_drift)."""
    output = Path(cfg.output_dir)
    node, body, _ = partition_body(model)
    output.mkdir(parents=True, exist_ok=True)
    (output / KERNEL_CHOICES_FILE).write_text(json.dumps(kernel_choices_config(body), indent=2))
    configured_root(body, node.name)
    refuse_drift(read_target(body), cfg.resolved_target, "build")
    return model


def _saved_before(cfg: DataflowBuildConfig, name: str) -> Path:
    """The model the build saved before the step ``name``, once in its list: the
    intermediate model of the step before it."""
    from finn.builder.build_dataflow_plan import resolve_plan  # noqa: PLC0415 (it imports this)

    plan = resolve_plan(cfg)
    before = plan.steps[plan.index(name) - 1].name
    return Path(cfg.output_dir) / "intermediate_models" / f"{before}.onnx"


def step_verify_partition_python(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, verify_input_npy: Path = Path("input.npy")
) -> ModelWrapper:
    """Check the partition in Python: execute the parent graph, its partition node
    running the KernelOps, on each input of ``verify_input_npy`` (an .npy file, one
    input a row), and compare the partition's outputs with the model's before
    conversion, the one the build saved before step_convert_to_hw
    (finn.builder.build_dataflow_verification). Records its verdict; on ``fail`` the
    build continues. The plan refuses it unless step_convert_to_hw is in the list once,
    after another step and before it, and the intermediate models are saved."""
    reference = _saved_before(cfg, "step_convert_to_hw")
    passed = verify_partition_python(model, reference, verify_input_npy, Path(cfg.output_dir))
    verdict("step_verify_partition_python", passed)
    return model


def _concluded(name: str, check: Callable[[], object]) -> None:
    """Run the RTL check ``check`` of the step ``name`` and record its verdict: ``fail``
    where a tool or the simulation fails, and the step fails with that error. A tool
    that is not available concludes nothing."""
    try:
        check()
    except (ToolFailed, SimulationFailed):
        verdict(name, False)
        raise
    verdict(name, True)


def step_elaborate_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Compile and elaborate the partition's RTL in XSim (ElaboratePartition), into
    verification_output/partition_elaboration, and record the verdict; a failure fails
    the build."""
    _, body, _ = partition_body(model)
    elaborating = ElaboratePartition(
        directory=Path(cfg.output_dir) / VERIFICATION_DIR / "partition_elaboration",
        toolchain=cfg.step_toolchain("step_elaborate_partition"),
    )
    _concluded("step_elaborate_partition", lambda: body.transform(elaborating))
    return model


def _refuse_drift(
    model: ModelWrapper, cfg: DataflowBuildConfig, output: DataflowOutputType
) -> None:
    """Refuse to make ``output`` if the partition's target is not the configuration's."""
    _, body, _ = partition_body(model)
    refuse_drift(read_target(body), cfg.resolved_target, f"{output.value} build")


def step_package_partition(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, verify_input_npy: Path = Path("input.npy")
) -> ModelWrapper:
    """Package the partition as an IP named ``partition`` into ``stitched_ip/``, with its
    interface description (``interface.json``), if STITCHED_IP or OOC_SYNTH is asked.
    The partition's body records the IP, so a shell's integration later in the build
    reuses it instead of packaging again.

    With OOC_SYNTH, the IP is synthesized out of context; its utilization report, in
    ``stitched_ip/``, is listed under this step in the build's manifest, and
    report/resources.json reads it there. With STITCHED_IP, an XSim testbench is written
    into ``stitched_ip/testbench`` (finn.builder.build_dataflow_testbench). It streams
    the partition's inputs for the first frame of ``verify_input_npy`` if that file
    exists, else a generated frame; step_run_testbench runs it."""
    stitched = DataflowOutputType.STITCHED_IP in cfg.generate_outputs
    ooc = DataflowOutputType.OOC_SYNTH in cfg.generate_outputs
    if not (stitched or ooc):
        _LOG.info("STITCHED_IP and OOC_SYNTH not in requested outputs, skipping.")
        return model
    directory = Path(cfg.output_dir) / STITCHED_IP_DIR
    toolchain = cfg.step_toolchain("step_package_partition")
    node, body, body_file = partition_body(model)
    body = body.transform(
        PackagePartition(
            node.name,
            run_synth=ooc,
            directory=directory,
            toolchain=toolchain,
        )
    )
    body.save(body_file)
    _LOG.info(f"Packaged IP {node.name} and its interface description in {directory}")
    if ooc:
        reported(partition_utilization_report(directory))
    if stitched:
        frame, stimulus = testbench_frame(model, body, verify_input_npy)
        testbench = directory / TESTBENCH_DIR
        write_testbench(body, testbench, frame, toolchain=toolchain, label=node.name)
        _LOG.info(f"XSim testbench on {stimulus} written into {testbench} (run.sh runs it)")
    return model


def step_run_testbench(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run the packaged partition's XSim testbench, the one step_package_partition
    writes with STITCHED_IP, once in the build's toolchain (run_testbench), and record
    the verdict; a mismatch fails the build (``SimulationFailed``). The plan refuses it
    without step_package_partition before it in the list, or without STITCHED_IP
    asked."""
    testbench = Path(cfg.output_dir) / STITCHED_IP_DIR / TESTBENCH_DIR
    toolchain = cfg.step_toolchain("step_run_testbench")
    _concluded("step_run_testbench", lambda: run_testbench(testbench, toolchain=toolchain))
    return model


def step_build_bitfile(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    hw_debug: bool = False,
) -> ModelWrapper:
    """Build the target's shell around the partition to a bitfile in the output
    directory, if BITFILE is asked. Only the pynq shell has a bitfile build
    (BuildPynqBitfile). Each tool's transcript goes into logs/step_build_bitfile.
    Refuses a partition built for another target than the configuration's. The model
    records the bitfile and its hardware handoff (finn.outputs), what the bitfile was
    built from and the options it was built with. The build's reports, collected beside
    the bitfile under report/ (finn.shells.pynq.runner.REPORTS), and the Vivado project
    are listed under this step in the build's manifest. Any shortfall of the delivered
    clock is logged as a warning.

    ``hw_debug``: insert integrated logic analyzers on the ends' streams. A shell
    without a build (``ip``) refuses it before the build starts.

    The objective report/delivered_clock.json states beside what the delivered clock
    gives is the throughput step_target_throughput folded the partition to, as the
    partition's body records it (OBJECTIVE_FPS); none where it records none."""
    if DataflowOutputType.BITFILE not in cfg.generate_outputs:
        _LOG.info("BITFILE not in requested outputs, skipping step_build_bitfile.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.BITFILE)
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
        _LOG.warning(finding.message)
    _LOG.info(f"Bitfile written into {building.bitfile.parent}")
    return model


def step_make_driver(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write the PYNQ driver into driver/ and its description into report/driver.json,
    if PYNQ_DRIVER is asked. The driver sets PL0 to the clock the bitfile actually
    delivers (report/delivered_clock.json), not the one requested, and loads the
    build's bitfile by default. Refuses a bitfile built from another design than the
    current partition, and a partition built for another target than the
    configuration's. The description records the design the driver was made from
    (``made_from``), which the deployment package checks."""
    if DataflowOutputType.PYNQ_DRIVER not in cfg.generate_outputs:
        _LOG.info("PYNQ_DRIVER not in requested outputs, skipping step_make_driver.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.PYNQ_DRIVER)
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
    """Package the bitfile, the driver, the host-side model (deploy/model/) and the
    driver's description (deploy/driver.json) into deploy/, if DEPLOYMENT_PACKAGE is
    asked; deploy/ is written afresh (finn.shells.pynq.deploy). Ships the bitfile and
    driver this output directory's own steps made. Refuses a model that records no
    bitfile, or another directory's; an output directory without a driver; a bitfile or
    driver made from another design than the current partition; and a partition built
    for another target than the configuration's."""
    if DataflowOutputType.DEPLOYMENT_PACKAGE not in cfg.generate_outputs:
        _LOG.info("DEPLOYMENT_PACKAGE not in requested outputs, skipping.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.DEPLOYMENT_PACKAGE)
    return model.transform(MakePynqDeployment(Path(cfg.output_dir)))


def step_resources_report(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write report/resources.json: the resources of each member of the shell (the
    partition, each end, each IP of the static region) and their total, as modelled, as
    synthesized out of context and as placed (finn.builder.build_dataflow_resources).
    It reads the reports the build made: those collected beside the bitfile
    (finn.shells.pynq.runner.REPORTS: ``placed``, ``out_of_context``) and, with
    OOC_SYNTH, the packaged partition's in stitched_ip/ (``ooc_synth``). A column the
    build did not make is marked absent."""
    ooc = DataflowOutputType.OOC_SYNTH in cfg.generate_outputs
    packaged = Path(cfg.output_dir) / STITCHED_IP_DIR if ooc else None
    stated = shell_resources_report(model, built_reports(model, packaged))
    report = Path(cfg.output_dir) / RESOURCES_FILE
    write_report(report, stated)
    absent = ", ".join(f"{column} ({why})" for column, why in stated["absent"].items())
    _LOG.info(
        f"Resources per member written to {report}" + (f"; absent: {absent}" if absent else "")
    )
    return model


#: The build's phases, in the order a build runs them, each its steps in order. A phase is
#: a name: in a configuration's ``steps`` it stands for its steps, as ``start_step`` for
#: its first and as ``stop_step`` for its last.
build_dataflow_phases: dict[str, list[StepFunction]] = {
    # Bring the export into FINN's form and clean it for the compiler without changing
    # its computation, its inputs and outputs as the configuration states them.
    "phase_prepare_model": [
        step_cleanup,
        step_add_pre_post_processing,
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_channels_last,
    ],
    # Record the target in the model and convert the graph's nodes to KernelOps.
    "phase_convert_to_hardware": [step_convert_to_hw],
    # Cut the KernelOps into one partition; the other nodes stay on the host.
    "phase_create_accelerator": [step_create_dataflow_partition],
    # Choose the partition's parameters, one strategy step at a time (by default its
    # FIFOs sized at the baseline folding), then check it. The other strategy steps are
    # added by name (KERNEL_STRATEGY_STEPS).
    "phase_optimize_hardware": [step_size_fifos, step_verify_partition],
    # Generate the partition's HDL and package it as an IP.
    "phase_emit_hardware": [step_package_partition],
    # Make the outputs the configuration asks for: bitfile, driver, resources report,
    # deployment package.
    "phase_generate_outputs": [
        step_build_bitfile,
        step_make_driver,
        step_resources_report,
        step_deployment_package,
    ],
}

#: The steps that choose the partition's parameters, one strategy each, which a
#: configuration adds to its list by name (by default, only step_size_fifos runs).
KERNEL_STRATEGY_STEPS: list[StepFunction] = [
    step_pin_choices,
    step_target_throughput,
    step_max_throughput,
    step_size_fifos,
    step_baseline_choices,
    step_placeholder_choices,
]

#: The strategy steps that may change the kernels' folding, which FIFO sizing reads (a
#: pin may too, and going back to the baselines does), and those that size the FIFOs at
#: the folding they find: a build's plan refuses a folding step after its last sizing
#: step (finn.builder.build_dataflow_plan).
FOLDING_STEPS: tuple[StepFunction, ...] = (
    step_pin_choices,
    step_target_throughput,
    step_max_throughput,
    step_baseline_choices,
)
FIFO_SIZING_STEPS: tuple[StepFunction, ...] = (step_size_fifos,)

#: The steps that check the partition, in no phase: a configuration lists each it asks
#: for, the Python check and the elaboration after the strategy steps, the testbench
#: after step_package_partition.
VERIFYING_STEPS: list[StepFunction] = [
    step_verify_partition_python,
    step_elaborate_partition,
    step_run_testbench,
]

#: The build's steps by name: the phases', the strategy and the verifying steps.
build_dataflow_step_lookup: dict[str, StepFunction] = {
    step.__name__: step
    for step in [
        *(step for steps in build_dataflow_phases.values() for step in steps),
        *KERNEL_STRATEGY_STEPS,
        *VERIFYING_STEPS,
    ]
}
