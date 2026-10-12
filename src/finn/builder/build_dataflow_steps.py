# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's steps, and at the end its phases (``build_dataflow_phases``).

A step is a function ``(model, cfg, *, settings...) -> model``. Its keyword-only
parameters are its settings: the build fills each from the configuration's setting of
that name, else it keeps its default (finn.builder.build_dataflow_plan). Each phase is a
list of steps; a build without ``steps`` runs every phase in order.

The steps prepare a Brevitas export into the streamlined graph the kernel path converts;
convert its nodes to KernelOps (finn.custom_op.kernels) and cut them into one partition;
choose the partition's parameters; and make the outputs the configuration asks for: the
partition's packaged IP on any shell, and the bitfile, driver and deployment package on
a shell that integrates it."""

import copy
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    Step,
    VerificationStepType,
)
from finn.builder.build_dataflow_manifest import REPORT, VERIFICATION
from finn.builder.build_dataflow_resources import RESOURCES_FILE, shell_resources_report
from finn.builder.build_dataflow_testbench import (
    TESTBENCH_DIR,
    generated_frame,
    partition_frame,
    run_testbench,
    write_testbench,
)
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.base import read_target
from finn.custom_op.kernels.shell import configured_root
from finn.custom_op.partition.kernel_partitions import (
    OUTPUT_BITFILE,
    OUTPUT_HWH,
    OUTPUT_IP,
    design_digest,
    partition_body,
)
from finn.platform import refuse_drift
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT, MakePynqDriver
from finn.shells.pynq.runner import (
    BITFILE,
    REPORTS,
    BuildPynqBitfile,
    bitfile_directory,
    pynq_row,
    refuse_stale_bitfile,
)
from finn.toolchain.launcher import Toolchain, ToolFailed
from finn.transformation.kernels import (
    EXPLORATION_REPORT,
    ExplorePartition,
    ToKernelOps,
    kernel_choices_config,
    kernel_ops_report,
    kernel_ops_summary,
    refuse_host_between,
    strategy,
)
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import (
    ElaboratePartition,
    PackagePartition,
    partition_utilization_report,
)
from finn.transformation.prepare import SUB_PHASES, GraphPreparation, census
from finn.transformation.prepare.phase import (
    SubPhase,
    imported,
    io_explicit,
    quantization_lowered,
    streamlined,
    topology_settled,
)

#: The steps' logger. The build log has every line; the console shows only warnings, such
#: as a failed verification the build continued past, or a limitation.
_LOG = logging.getLogger(__name__)

#: The graph-preparation phase's report, under the output directory.
PREPARATION_REPORT = "report/graph_preparation.json"


def _tools(cfg: DataflowBuildConfig, step: str) -> Toolchain:
    """The build's toolchain for ``step``, writing every tool's transcript into
    ``output_dir/logs/<step>``. The step passes it to each transformation that runs a
    tool."""
    return cfg.prepared_toolchain.transcripts_in(Path(cfg.output_dir) / "logs" / step)


def _verdict(name: str, passed: bool) -> None:
    """Log a verification's result, ``Verification for <name> : SUCCESS`` or ``FAIL``. A
    FAIL is a warning, because the build continues. The build's manifest records the
    result (finn.builder.build_dataflow_manifest.VERIFICATION)."""
    verdict = {VERIFICATION: {"name": name, "passed": passed}}
    if passed:
        _LOG.info(f"Verification for {name} : SUCCESS", extra=verdict)
    else:
        _LOG.warning(f"Verification for {name} : FAIL", extra=verdict)


def _reported(path: str | os.PathLike[str]) -> None:
    """Log ``wrote <path>`` for a report the step wrote; the build's manifest lists it
    under the step, with its findings (finn.builder.build_dataflow_manifest.REPORT)."""
    _LOG.info(f"wrote {path}", extra={REPORT: os.fspath(path)})


def _preparation_report(cfg: DataflowBuildConfig, fresh: bool = False) -> dict[str, Any]:
    """The graph-preparation report written so far, or an empty one if ``fresh`` or none
    exists."""
    path = Path(cfg.output_dir) / PREPARATION_REPORT
    if fresh or not path.is_file():
        return {"sub_phases": []}
    report: dict[str, Any] = json.loads(path.read_text())
    return report


def _write_preparation_report(cfg: DataflowBuildConfig, report: dict[str, Any]) -> None:
    path = Path(cfg.output_dir) / PREPARATION_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    _reported(path)


def _sub_phase(
    run: SubPhase,
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    options: GraphPreparation,
    fresh: bool = False,
) -> ModelWrapper:
    """Run one preparation sub-phase, ``run``, on ``model`` with ``options``, record the
    nodes it added and removed in the report under its name, and log a one-line summary.
    ``fresh`` starts the report afresh."""
    name = next(name for name, sub_phase in SUB_PHASES if sub_phase is run)
    before = copy.deepcopy(model)
    model = run(model, options)
    changed = census(before, model)
    report = _preparation_report(cfg, fresh)
    report["sub_phases"].append({"sub_phase": name, **changed})
    _write_preparation_report(cfg, report)
    counts = ", ".join(
        f"{sign}{count} {op}"
        for sign, side in (("-", "removed"), ("+", "added"))
        for op, count in changed[side].items()
    )
    _LOG.info(f"Graph preparation: {name}: {counts or 'no node changed'}")
    return model


def step_tidy_up(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    override_inpsize: int | list[int] | None = None,
) -> ModelWrapper:
    """Run qonnx's cleanup on the export. Starts report/graph_preparation.json afresh.

    ``override_inpsize``: the batch size (an int) or the whole input shape (a list) the
    cleanup sets; None keeps the export's."""
    preparation = GraphPreparation(override_inpsize=override_inpsize)
    return _sub_phase(imported, model, cfg, preparation, fresh=True)


def step_qonnx_to_finn(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, max_multithreshold_bit_width: int = 8
) -> ModelWrapper:
    """Lower Quant nodes: activations to MultiThresholds, weights to integers.

    ``max_multithreshold_bit_width``: the widest Quant activation lowered to a
    MultiThreshold; a wider one stays a Quant."""
    preparation = GraphPreparation(max_multithreshold_bit_width=max_multithreshold_bit_width)
    return _sub_phase(quantization_lowered, model, cfg, preparation)


def step_add_pre_post_processing(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    preprocessing: Path | None = None,
    input_datatype: str | None = None,
    topk: int | None = None,
    max_multithreshold_bit_width: int = 8,
) -> ModelWrapper:
    """State the network's inputs and outputs; by default, the export's own.

    - ``preprocessing``: an ONNX model (a QONNX export, such as one of
      ``finn.util.pytorch.ToTensor``) merged ahead of the network; its input becomes the
      graph's input. It is lowered on its own first, with step_qonnx_to_finn's
      ``max_multithreshold_bit_width``.
    - ``input_datatype``: the qonnx name of the graph input's datatype, the range of
      values the network is prepared for. None leaves the input unannotated, which is
      read as FLOAT32.
    - ``topk``: append a TopK that selects the ``topk`` largest outputs; its indices
      become the graph's output."""
    preparation = GraphPreparation(
        max_multithreshold_bit_width=max_multithreshold_bit_width,
        preprocessing=None if preprocessing is None else str(preprocessing),
        input_datatype=input_datatype,
        topk=topk,
    )
    return _sub_phase(io_explicit, model, cfg, preparation)


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Streamline the graph: scales and biases absorbed into thresholds, convolutions
    lowered to Im2Col and MatMul, bipolar MatMuls rewritten to XnorPopcountMatMul."""
    return _sub_phase(streamlined, model, cfg, GraphPreparation())


def step_convert_to_channels_last(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Absorb and collapse transposes, make MaxPool NHWC, and infer the data layouts."""
    return _sub_phase(topology_settled, model, cfg, GraphPreparation())


def _kernel_path_source(cfg: DataflowBuildConfig) -> str:
    """Where step_convert_to_hw saves the model it converts, the reference for the
    PARTITION_PYTHON check."""
    return cfg.output_dir + "/intermediate_models/kernel_path_source.onnx"


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Record the build's target in the model, and convert each node a KernelOp
    supports (MatMul, MultiThreshold) into that KernelOp. Writes report/kernel_ops.json:
    what each node became, or why it stayed on the host. Refuses a host node between
    KernelOps, since they must form one partition. With PARTITION_PYTHON, the model
    before conversion is saved as that check's reference."""
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        os.makedirs(os.path.dirname(_kernel_path_source(cfg)), exist_ok=True)
        model.save(_kernel_path_source(cfg))
    conversion = ToKernelOps(cfg.resolved_target)
    model = model.transform(conversion)
    report = kernel_ops_report(model, conversion.outcomes)
    os.makedirs(cfg.output_dir + "/report", exist_ok=True)
    with open(cfg.output_dir + "/report/kernel_ops.json", "w") as f:
        json.dump(report, f, indent=2)
    _reported(cfg.output_dir + "/report/kernel_ops.json")
    for line in kernel_ops_summary(conversion.outcomes):
        _LOG.info(line)
    refuse_host_between(model, conversion.outcomes)
    return model


#: step_explore's strategies by default: the baseline folding, its FIFOs sized.
DEFAULT_STRATEGIES: tuple[Mapping[str, Any], ...] = ({"strategy": "size_fifos"},)


def step_explore(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    *,
    strategies: Sequence[Mapping[str, Any]] = DEFAULT_STRATEGIES,
) -> ModelWrapper:
    """Run the strategies over the partition's KernelOps, in order, and save every value
    in the partition's body: a kernel's on its node, a channel's on its tensor. Each
    strategy carries its own objective; what no strategy decides keeps its baseline.
    Writes report/kernel_exploration.json (what each strategy changed, who set each
    value, the choices left open, and the limitations found) and kernel_choices.json
    (every choice by node or tensor name, which the ``pinned`` strategy reads back). A
    limitation, such as the shell's resources exceeding the part's, is logged as a
    warning, never an error.

    ``strategies``: each a spec with its own parameters
    (finn.transformation.kernels.KERNEL_STRATEGIES):

    - ``{"strategy": "pinned", "path": ...}``: the choices in a kernel_choices.json;
    - ``{"strategy": "target_throughput", "fps": ..., "relax": true}``: the least
      parallelism that meets ``fps`` at the target's clock;
    - ``{"strategy": "max_throughput", "within": {"lut": 0.5}}``: the fastest folding
      whose shell's resources stay within the given fractions of the part
      (``within`` is required);
    - ``{"strategy": "size_fifos", "margin": 0}``: each channel's FIFO sized from the
      beat patterns at both its ends;
    - ``{"strategy": "baseline"}``: every choice not pinned set back to its baseline;
    - ``{"strategy": "placeholder"}``: for debugging, every open required choice set
      to its first allowed value.

    Every choice starts at its baseline, set when it is created (a KernelOp's at
    conversion, a channel's at the cut). Each strategy replaces the values it decides,
    and every value is saved. Pins go first, and FIFO sizing after the last strategy
    that changes the folding; a list that folds after its last size_fifos is refused.
    To start over from the baselines, put ``{"strategy": "baseline"}`` first.

    The default keeps the baseline folding and sizes its FIFOs. A list you give
    replaces the default and runs as written: without size_fifos, every channel keeps
    its baseline, a direct connection. The partition is packaged and built from exactly
    the saved values, and a required choice no strategy set is refused there, by
    name."""
    exploring = ExplorePartition([strategy(spec) for spec in strategies])
    model = model.transform(exploring)
    explored, body = exploring.explored, exploring.body
    assert explored is not None and body is not None
    os.makedirs(cfg.output_dir + "/report", exist_ok=True)
    with open(os.path.join(cfg.output_dir, EXPLORATION_REPORT), "w") as f:
        json.dump(explored.report, f, indent=2)
    _reported(os.path.join(cfg.output_dir, EXPLORATION_REPORT))
    with open(cfg.output_dir + "/kernel_choices.json", "w") as f:
        json.dump(kernel_choices_config(body), f, indent=2)
    report = explored.report
    for each in report["strategies"]:
        _LOG.info(f"Kernel choices: {each['strategy']} changed {len(each['changed'])}")
        if each["strategy"] == "max_throughput":
            reached = each["bottleneck"]["cycles"]
            _LOG.info(
                f"Kernel choices: max_throughput reached {reached} cycles a frame within "
                f"{each['budget']} ({'fits' if each['fits'] else 'does not fit'}; binding: "
                f"{each['binding']}; {len(each['tried'])} budgets tried)"
            )
    for finding in explored.findings:
        _LOG.warning(f"Kernel choices: {finding.message}")
    for node, held in report["choices"].items():
        for attribute, entry in held.items():
            if str(entry["by"]).startswith("DEBUG"):
                _LOG.info(f"{entry['by']}: {node}.{attribute} = {entry['value']!r}")
    if report["open"]:
        _LOG.info(f"Kernel choices: open, refused at packaging: {', '.join(report['open'])}")
    _LOG.info(f"FIFOs: {report['fifos']}")
    return model


def _partition_directory(cfg: DataflowBuildConfig) -> Path:
    """Where the build keeps the partition's body (``partition.onnx``) and the scratch
    models the shell's integration generates the ends' IPs from (``ends/``)."""
    return Path(cfg.output_dir) / "partition"


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Cut the KernelOps into one StreamingDataflowPartition, leaving the nodes before
    and after it on the host. From here on the build's model is the parent graph. The
    partition is named ``partition``: its node, its body (partition/partition.onnx), its
    IP and its block-design instance. Its body carries the target and every choice at
    its baseline. Each channel's choices are created here, and each KernelOp's are
    checked again in context: one that is stale because the model was edited since
    conversion is reset to its baseline, with a warning saying why."""
    cut = CutKernelPartition(_partition_directory(cfg))
    model = model.transform(cut)
    for key, why in cut.dropped.items():
        _LOG.warning(f"Kernel choices: {key} dropped as stale at the cut and re-baselined: {why}")
    return model


def _verify_partition_python(model: ModelWrapper, cfg: DataflowBuildConfig) -> bool:
    """Execute the parent graph ``model``, its partition node running the KernelOps, on
    each input in ``verify_input_npy``, and compare the partition's outputs value for
    value with those of the model saved before conversion (step_convert_to_hw). Saves
    each input's partition outputs as
    verification_output/verify_partition_python_<index>_<SUCCESS|FAIL>.npz, and
    returns whether all matched."""
    source_file = _kernel_path_source(cfg)
    if not os.path.isfile(source_file):
        raise FileNotFoundError(
            f"{source_file}: the kernel path's source, which step_convert_to_hw keeps when "
            "PARTITION_PYTHON is asked, is not there (did the build start after it?)"
        )
    verify_out_dir = cfg.output_dir + "/verification_output"
    os.makedirs(verify_out_dir, exist_ok=True)
    node, _, _ = partition_body(model)
    source = ModelWrapper(source_file)
    source_input = source.graph.input[0].name
    shape = source.get_tensor_shape(source_input)
    if shape is None:
        raise ValueError(f"partition_python: the source's input {source_input} states no shape")
    ishape = tuple(shape)
    outputs = list(node.output)
    inputs = np.load(cfg.verify_input_npy)
    all_match = True
    for index in range(inputs.shape[0]):
        frame = inputs[index : index + 1].reshape((1,) + ishape[1:])
        built = execute_onnx(model, {model.graph.input[0].name: frame}, True)
        expected = execute_onnx(source, {source_input: frame}, True)
        mismatched = [name for name in outputs if not np.array_equal(built[name], expected[name])]
        for name in mismatched:
            _LOG.warning(
                f"partition_python: input {index}: {name} differs from the source's in "
                f"{int(np.sum(built[name] != expected[name]))} of {built[name].size} values"
            )
        result = "FAIL" if mismatched else "SUCCESS"
        all_match = all_match and not mismatched
        np.savez(
            f"{verify_out_dir}/verify_partition_python_{index}_{result}.npz",
            **{name: built[name] for name in outputs},
        )
    _LOG.info(
        f"partition_python: the partition's outputs {outputs} on {inputs.shape[0]} "
        f"inputs of {cfg.verify_input_npy}, against the model the kernel path started from"
    )
    return all_match


def step_verify_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Check the partition before a shell builds it: its choices are complete and
    current, its ports match its graph's inputs and outputs in order (configured_root),
    and its target is the configuration's (refuse_drift). Then run the optional checks
    ``verify_steps`` asks for: PARTITION_PYTHON compares the partition's outputs with
    the model before conversion (_verify_partition_python), and PARTITION_ELABORATION
    compiles and elaborates its RTL in XSim."""
    node, body, _ = partition_body(model)
    configured_root(body, node.name)
    refuse_drift(read_target(body), cfg.resolved_target, "build")
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        matched = _verify_partition_python(model, cfg)
        _verdict("partition_python", matched)
    if VerificationStepType.PARTITION_ELABORATION in cfg.verify_steps:
        directory = cfg.output_dir + "/verification_output/partition_elaboration"
        body.transform(
            ElaboratePartition(
                directory=Path(directory),
                toolchain=_tools(cfg, "step_verify_partition"),
            )
        )
        _verdict("partition_elaboration", True)
    return model


def _refuse_drift(
    model: ModelWrapper, cfg: DataflowBuildConfig, output: DataflowOutputType
) -> None:
    """Refuse to make ``output`` if the partition's target is not the configuration's."""
    _, body, _ = partition_body(model)
    refuse_drift(read_target(body), cfg.resolved_target, f"{output.value} build")


def _stitched_ip_directory(cfg: DataflowBuildConfig) -> Path:
    """Where step_package_partition packages the partition: its IP, interface
    description and testbench, and with OOC_SYNTH its utilization report."""
    return Path(cfg.output_dir) / "stitched_ip"


def step_package_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Package the partition as an IP named ``partition`` into ``stitched_ip/``, with its
    interface description (``interface.json``), if STITCHED_IP or OOC_SYNTH is asked.
    The partition's body records the IP, so a shell's integration later in the build
    reuses it instead of packaging again.

    With OOC_SYNTH, the IP is synthesized out of context; its utilization report, in
    ``stitched_ip/``, is listed under this step in the build's manifest, and
    report/resources.json reads it there. With STITCHED_IP, an XSim testbench is written
    into ``stitched_ip/testbench`` (finn.builder.build_dataflow_testbench). It streams
    the partition's inputs for the first frame of ``verify_input_npy`` if that file
    exists, else a generated frame. The testbench is run only if STITCHED_IP_TESTBENCH is
    in ``verify_steps``, and the step then fails (``SimulationFailed``) if the module's
    outputs differ from the partition's in Python."""
    stitched = DataflowOutputType.STITCHED_IP in cfg.generate_outputs
    ooc = DataflowOutputType.OOC_SYNTH in cfg.generate_outputs
    if not (stitched or ooc):
        _LOG.info("STITCHED_IP and OOC_SYNTH not in requested outputs, skipping.")
        return model
    directory = _stitched_ip_directory(cfg)
    toolchain = _tools(cfg, "step_package_partition")
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
        _reported(partition_utilization_report(directory))
    if stitched:
        if os.path.isfile(cfg.verify_input_npy):
            frame = partition_frame(model, np.load(cfg.verify_input_npy)[0])
            stimulus = f"the first input of {cfg.verify_input_npy}"
        else:
            frame, stimulus = generated_frame(body), "a generated frame"
        testbench = directory / TESTBENCH_DIR
        write_testbench(body, testbench, frame, toolchain=toolchain, label=node.name)
        _LOG.info(f"XSim testbench on {stimulus} written into {testbench} (run.sh runs it)")
        if VerificationStepType.STITCHED_IP_TESTBENCH in cfg.verify_steps:
            run_testbench(testbench, toolchain=toolchain)
            _verdict("stitched_ip_testbench", True)
    return model


def step_build_bitfile(
    model: ModelWrapper, cfg: DataflowBuildConfig, *, hw_debug: bool = False
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
    without a build (``ip``) refuses it before the build starts."""
    if DataflowOutputType.BITFILE not in cfg.generate_outputs:
        _LOG.info("BITFILE not in requested outputs, skipping step_build_bitfile.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.BITFILE)
    building = BuildPynqBitfile(
        Path(cfg.output_dir),
        toolchain=_tools(cfg, "step_build_bitfile"),
        hw_debug=hw_debug,
    )
    model = model.transform(building)
    for report in REPORTS.values():
        _reported(Path(cfg.output_dir) / report)
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
    _reported(Path(cfg.output_dir) / DRIVER_REPORT)
    description = making.description
    returned = ", ".join(
        f"{each['name']} ({each['element']}, {each['shape']})" for each in description["returns"]
    )
    driver_dir = os.path.join(cfg.output_dir, DRIVER_DIR)
    _LOG.info(
        f"PYNQ Python driver written into {driver_dir}, at {description['fclk_mhz']} MHz; "
        f"it returns {returned}"
    )
    return model


#: The finn.outputs keys a deployed model drops: they hold paths on the build machine.
_MACHINE_PATHS = (OUTPUT_IP, OUTPUT_BITFILE, OUTPUT_HWH)


def _host_model(model: ModelWrapper, directory: Path) -> Path:
    """Write the parent graph ``model`` into ``directory`` as the deployment ships it:
    parent.onnx (the host's nodes and the partition node) and partition.onnx beside it.
    The node refers to its body by a relative path, and neither model keeps a path on
    the build machine (_MACHINE_PATHS). Returns the path of parent.onnx."""
    directory.mkdir(parents=True, exist_ok=True)
    node, body, _ = partition_body(model)
    shipped = ModelWrapper(copy.deepcopy(model.model))
    shipped_body = ModelWrapper(copy.deepcopy(body.model))
    for each in (shipped, shipped_body):
        for key in _MACHINE_PATHS:
            each.delete(key)
    (shipped_node,) = [each for each in shipped.graph.node if each.name == node.name]
    getCustomOp(shipped_node).set_nodeattr("model", "partition.onnx")
    shipped_body.save(str(directory / "partition.onnx"))
    shipped.save(str(directory / "parent.onnx"))
    return directory / "parent.onnx"


def step_deployment_package(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Package the bitfile, the driver, the host-side model (deploy/model/) and the
    driver's description (deploy/driver.json) into deploy/, if DEPLOYMENT_PACKAGE is
    asked; deploy/ is written afresh. Ships the bitfile and driver this output
    directory's own steps made. Refuses a model that records no bitfile, or another
    directory's; an output directory without a driver; and a bitfile or driver made from
    another design than the current partition."""
    if DataflowOutputType.DEPLOYMENT_PACKAGE not in cfg.generate_outputs:
        _LOG.info("DEPLOYMENT_PACKAGE not in requested outputs, skipping.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.DEPLOYMENT_PACKAGE)
    pynq_row(model, "deployment_package")
    output = Path(cfg.output_dir)
    stated = model.get(OUTPUT_BITFILE)
    if stated is None:
        raise ValueError(
            "deployment_package: the model states no bitfile (finn.outputs): the "
            "deployment ships the bitfile and the driver its build made; ask bitfile "
            "and pynq_driver with it"
        )
    if Path(stated) != output / BITFILE:
        raise ValueError(
            f"deployment_package: the model's bitfile is {stated}, not this output "
            f"directory's ({output / BITFILE})"
        )
    if not (output / DRIVER_REPORT).is_file():
        raise ValueError(
            f"deployment_package: no driver in this output directory ({output / DRIVER_REPORT}):"
            " the deployment ships the bitfile and the driver its build made; ask "
            "pynq_driver with it"
        )
    refuse_stale_bitfile(model, "deployment_package")
    _, body, _ = partition_body(model)
    made_from = json.loads((output / DRIVER_REPORT).read_text()).get("made_from")
    now = design_digest(body)
    if made_from != now:
        raise ValueError(
            f"deployment_package: the driver in this output directory ({output / DRIVER_DIR}) "
            f"was made from the design {made_from}, and the partition's design is now {now}; "
            "make the driver again (pynq_driver)"
        )
    deploy = output / "deploy"
    shutil.rmtree(deploy, ignore_errors=True)
    shutil.copytree(output / BITFILE.parent, deploy / BITFILE.parent)
    shutil.copytree(output / DRIVER_DIR, deploy / DRIVER_DIR, copy_function=shutil.copyfile)
    _host_model(model, deploy / "model")
    shutil.copy(output / DRIVER_REPORT, deploy / "driver.json")
    return model


def step_resources_report(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write report/resources.json: the resources of each member of the shell (the
    partition, each end, each IP of the static region) and their total, as modelled, as
    synthesized out of context and as placed (finn.builder.build_dataflow_resources).
    It reads the reports the build made: those collected beside the bitfile
    (finn.shells.pynq.runner.REPORTS: ``placed``, ``out_of_context``) and, with
    OOC_SYNTH, the packaged partition's in stitched_ip/ (``ooc_synth``). A column the
    build did not make is marked absent."""
    report = Path(cfg.output_dir) / RESOURCES_FILE
    report.parent.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Path] = {}
    built = bitfile_directory(model)
    if built is not None:
        reports.update({name: built / REPORTS[name] for name in ("placed", "out_of_context")})
    if DataflowOutputType.OOC_SYNTH in cfg.generate_outputs:
        try:
            reports["ooc_synth"] = partition_utilization_report(_stitched_ip_directory(cfg))
        except ToolFailed:
            pass  # not synthesized into this output directory: the column is absent
    stated = shell_resources_report(model, reports)
    report.write_text(json.dumps(stated, indent=2))
    _reported(report)
    absent = ", ".join(f"{column} ({why})" for column, why in stated["absent"].items())
    _LOG.info(
        f"Resources per member written to {report}" + (f"; absent: {absent}" if absent else "")
    )
    return model


#: The build's phases, in the order a build runs them, each its steps in order. A
#: configuration's lists name a phase wherever they name a step.
build_dataflow_phases: dict[str, list[Step]] = {
    # Bring the export into FINN's form and clean it for the compiler without changing
    # its computation (finn.transformation.prepare).
    "phase_prepare_model": [
        step_tidy_up,
        step_qonnx_to_finn,
        step_add_pre_post_processing,
        step_streamline,
        step_convert_to_channels_last,
    ],
    # Record the target in the model and convert the graph's nodes to KernelOps.
    "phase_convert_to_hardware": [step_convert_to_hw],
    # Cut the KernelOps into one partition; the other nodes stay on the host.
    "phase_create_accelerator": [step_create_dataflow_partition],
    # Choose the partition's parameters with the strategies, then check it.
    "phase_optimize_hardware": [step_explore, step_verify_partition],
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

#: The build's steps by name.
build_dataflow_step_lookup: dict[str, Step] = {
    step.__name__: step for steps in build_dataflow_phases.values() for step in steps
}

#: The entries a build without ``steps`` runs: every phase, in order.
default_build_dataflow_steps: list[str] = list(build_dataflow_phases)
